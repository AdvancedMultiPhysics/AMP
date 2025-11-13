#include "AMP/mesh/structured/StructuredGeometryMesh.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/MeshParameters.h"


namespace AMP::Mesh {


/****************************************************************
 * Constructors                                                 *
 ****************************************************************/
StructuredGeometryMesh::StructuredGeometryMesh( std::shared_ptr<const MeshParameters> params )
    : BoxMesh( params )
{
    // Check for valid inputs
    AMP_INSIST( params.get(), "Params must not be null" );
    auto db = params->getDatabase();
    AMP_INSIST( db.get(), "Database must exist" );
    AMP_INSIST( d_comm != AMP_MPI( AMP_COMM_NULL ), "Communicator must be set" );
    // Construct the geometry
    auto db2 = db->cloneDatabase();
    db2->erase( "x_offset", false );
    db2->erase( "y_offset", false );
    db2->erase( "z_offset", false );
    d_geometry  = AMP::Geometry::Geometry::buildGeometry( std::move( db2 ) );
    d_geometry2 = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( d_geometry );
    AMP_ASSERT( d_geometry2 );
    // Fill basic mesh information
    PhysicalDim = d_geometry2->getDim();
    GeomDim     = static_cast<AMP::Mesh::GeomType>( d_geometry2->getLogicalDim() );
    d_max_gcw   = db->getWithDefault<int>( "GCW", 2 );
    AMP_ASSERT( PhysicalDim == db->getWithDefault<int>( "dim", PhysicalDim ) );
    auto size = d_geometry2->getLogicalGridSize( db->getVector<size_t>( "Size" ) );
    AMP_ASSERT( size.ndim() == static_cast<size_t>( GeomDim ) );
    std::array<int, 3> size2 = { (int) size[0], (int) size[1], (int) size[2] };
    auto surfaceIds          = d_geometry2->getLogicalSurfaceIds();
    if ( db->keyExists( "surfaceIds" ) ) {
        auto ids2 = db->getVector<int>( "surfaceIds" );
        for ( size_t i = 0; i < std::min( ids2.size(), surfaceIds.size() ); i++ ) {
            if ( surfaceIds[i] >= 0 )
                surfaceIds[i] = ids2[i];
            else
                AMP_INSIST( surfaceIds[i] == ids2[i], "Warning surface ids are not set correctly" );
        }
    }
    // Initialize the logical mesh
    BoxMesh::initialize(
        size2, surfaceIds, db->getWithDefault<std::vector<int>>( "LoadBalanceMinSize", {} ) );
    BoxMesh::finalize( db->getString( "MeshName" ), getDisplacement( db ) );
}
StructuredGeometryMesh::StructuredGeometryMesh(
    std::shared_ptr<AMP::Geometry::LogicalGeometry> geom,
    const ArraySize &size,
    const AMP::AMP_MPI &comm )
    : BoxMesh(), d_geometry2( geom )
{
    // Set base Mesh variables
    setMeshID();
    d_comm      = comm;
    d_geometry  = geom;
    GeomDim     = geom->getGeomType();
    PhysicalDim = geom->getDim();
    d_max_gcw   = 2;
    // Initialize the logical mesh
    AMP_ASSERT( size.ndim() == static_cast<int>( GeomDim ) );
    std::array<int, 3> size2 = { (int) size[0], (int) size[1], (int) size[2] };
    auto surfaceIds          = d_geometry2->getLogicalSurfaceIds();
    BoxMesh::initialize( size2, surfaceIds );
    BoxMesh::finalize( geom->getName(), {} );
}

StructuredGeometryMesh::StructuredGeometryMesh( const StructuredGeometryMesh &mesh )
    : BoxMesh( mesh )
{
    d_geometry2 = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( d_geometry );
    d_pos_hash  = mesh.d_pos_hash;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void StructuredGeometryMesh::writeRestart( int64_t fid ) const
{
    BoxMesh::writeRestart( fid );
    IO::writeHDF5( fid, "d_pos_hash", d_pos_hash );
}
StructuredGeometryMesh::StructuredGeometryMesh( int64_t fid, AMP::IO::RestartManager *manager )
    : BoxMesh( fid, manager )
{
    IO::readHDF5( fid, "d_pos_hash", d_pos_hash );
    d_geometry2 = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( d_geometry );
}


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string StructuredGeometryMesh::meshClass() const { return "StructuredGeometryMesh"; }


/****************************************************************
 * Basic functions                                               *
 ****************************************************************/
Mesh::Movable StructuredGeometryMesh::isMeshMovable() const { return Mesh::Movable::Displace; }
uint64_t StructuredGeometryMesh::positionHash() const { return d_pos_hash; }
void StructuredGeometryMesh::displaceMesh( const std::vector<double> &x )
{
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] += x[i];
        d_box[2 * i + 1] += x[i];
        d_box_local[2 * i + 0] += x[i];
        d_box_local[2 * i + 1] += x[i];
    }
    d_geometry2->displace( x.data() );
    d_pos_hash++;
}
void StructuredGeometryMesh::displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> )
{
    AMP_ERROR( "displaceMesh (vector) violates StructuredGeometryMesh properties" );
}
AMP::Geometry::Point
StructuredGeometryMesh::physicalToLogical( const AMP::Geometry::Point &x ) const
{
    return d_geometry2->logical( x );
}
std::unique_ptr<Mesh> StructuredGeometryMesh::clone() const
{
    return std::make_unique<StructuredGeometryMesh>( *this );
}


/****************************************************************
 * Return the coordinates                                        *
 ****************************************************************/
void StructuredGeometryMesh::coord( const MeshElementIndex &index, double *pos ) const
{
    AMP_ASSERT( index.type() == AMP::Mesh::GeomType::Vertex );
    double x = static_cast<double>( index.index( 0 ) ) / static_cast<double>( d_globalSize[0] );
    double y = static_cast<double>( index.index( 1 ) ) / static_cast<double>( d_globalSize[1] );
    double z = static_cast<double>( index.index( 2 ) ) / static_cast<double>( d_globalSize[2] );
    auto tmp = d_geometry2->physical( AMP::Geometry::Point( x, y, z ) );
    for ( int d = 0; d < PhysicalDim; d++ )
        pos[d] = tmp[d];
}
std::array<AMP::Array<double>, 3> StructuredGeometryMesh::localCoord() const
{
    auto local = getLocalBlock( d_comm.getRank() );
    ArraySize size( { local[1] - local[0] + 2, local[3] - local[2] + 2, local[5] - local[4] + 2 },
                    static_cast<int>( GeomDim ) );
    AMP::Array<double> x( size ), y( size ), z( size );
    double Ng[3] = { static_cast<double>( d_globalSize[0] ),
                     static_cast<double>( d_globalSize[1] ),
                     static_cast<double>( d_globalSize[2] ) };
    for ( size_t k = 0; k < size[2]; k++ ) {
        for ( size_t j = 0; j < size[1]; j++ ) {
            for ( size_t i = 0; i < size[0]; i++ ) {
                double x2    = static_cast<double>( i + local[0] ) / Ng[0];
                double y2    = static_cast<double>( j + local[2] ) / Ng[1];
                double z2    = static_cast<double>( k + local[4] ) / Ng[2];
                auto tmp     = d_geometry2->physical( AMP::Geometry::Point( x2, y2, z2 ) );
                x( i, j, k ) = tmp.x();
                y( i, j, k ) = tmp.y();
                z( i, j, k ) = tmp.z();
            }
        }
    }
    if ( PhysicalDim == 1 )
        return { x, {}, {} };
    else if ( PhysicalDim == 1 )
        return { x, y, {} };
    else
        return { x, y, z };
}
std::array<AMP::Array<double>, 3> StructuredGeometryMesh::globalCoord() const
{
    ArraySize size( { d_globalSize[0] + 1, d_globalSize[1] + 1, d_globalSize[2] + 1 },
                    static_cast<int>( GeomDim ) );
    AMP::Array<double> x( size ), y( size ), z( size );
    double Ng[3] = { static_cast<double>( d_globalSize[0] ),
                     static_cast<double>( d_globalSize[1] ),
                     static_cast<double>( d_globalSize[2] ) };
    for ( size_t k = 0; k < size[2]; k++ ) {
        for ( size_t j = 0; j < size[1]; j++ ) {
            for ( size_t i = 0; i < size[0]; i++ ) {
                double x2    = static_cast<double>( i ) / Ng[0];
                double y2    = static_cast<double>( j ) / Ng[1];
                double z2    = static_cast<double>( k ) / Ng[2];
                auto tmp     = d_geometry2->physical( AMP::Geometry::Point( x2, y2, z2 ) );
                x( i, j, k ) = tmp.x();
                y( i, j, k ) = tmp.y();
                z( i, j, k ) = tmp.z();
            }
        }
    }
    if ( PhysicalDim == 1 )
        return { x, {}, {} };
    else if ( PhysicalDim == 1 )
        return { x, y, {} };
    else
        return { x, y, z };
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool StructuredGeometryMesh::operator==( const Mesh &rhs ) const
{
    // Check base class variables
    if ( !BoxMesh::operator==( rhs ) )
        return false;
    // Check if we can cast to a MovableBoxMesh
    auto mesh = dynamic_cast<const StructuredGeometryMesh *>( &rhs );
    if ( !mesh )
        return false;
    // Perform basic comparison
    bool equal = *d_geometry2 == *mesh->d_geometry2;
    return equal;
}


} // namespace AMP::Mesh
