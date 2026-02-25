#include "AMP/mesh/structured/PureLogicalMesh.h"
#include "AMP/mesh/MeshParameters.h"


namespace AMP::Mesh {


/****************************************************************
 * Constructors                                                 *
 ****************************************************************/
PureLogicalMesh::PureLogicalMesh( std::shared_ptr<const MeshParameters> params ) : BoxMesh( params )
{
    // Check for valid inputs
    AMP_INSIST( params.get(), "Params must not be null" );
    auto db = params->getDatabase();
    AMP_INSIST( db.get(), "Database must exist" );
    if ( db->keyExists( "commSize" ) ) {
        const_cast<int &>( d_rank ) = db->getScalar<int>( "commRank" );
        const_cast<int &>( d_size ) = db->getScalar<int>( "commSize" );
        d_comm                      = AMP_COMM_NULL;
    } else {
        AMP_INSIST( !d_comm.isNull(), "Communicator must be set" );
    }
    // Fill basic mesh information
    auto size = db->getVector<int>( "Size" );
    auto per  = db->getWithDefault<std::vector<bool>>( "Periodic",
                                                      std::vector<bool>( size.size(), false ) );
    AMP_INSIST( size.size() >= 1u && size.size() <= 3u, "bad value for Size" );
    AMP_ASSERT( per.size() == size.size() );
    PhysicalDim = size.size();
    GeomDim     = static_cast<AMP::Mesh::GeomType>( size.size() );
    d_max_gcw   = db->getWithDefault<int>( "GCW", 2 );
    AMP_ASSERT( PhysicalDim == db->getWithDefault<int>( "dim", PhysicalDim ) );
    std::array<int, 3> size2 = { 1, 1, 1 };
    std::array<int, 6> ids   = { 0, 1, 2, 3, 4, 5 };
    for ( size_t d = 0; d < size.size(); d++ ) {
        size2[d] = size[d];
        if ( per[d] ) {
            ids[2 * d + 0] = -1;
            ids[2 * d + 1] = -1;
        }
    }
    // Initialize the logical mesh
    BoxMesh::initialize(
        size2, ids, db->getWithDefault<std::vector<int>>( "LoadBalanceMinSize", {} ) );
    BoxMesh::finalize( db->getString( "MeshName" ), getDisplacement( db ) );
}
PureLogicalMesh::PureLogicalMesh( const PureLogicalMesh &mesh ) = default;


/********************************************************
 * Create domain info                                    *
 ********************************************************/
void PureLogicalMesh::createBoundingBox()
{
    // Fill the bounding box
    d_box       = std::vector<double>( 2 * PhysicalDim );
    d_box_local = std::vector<double>( 2 * PhysicalDim );
    auto local  = getLocalBlock( d_comm.getRank() );
    for ( int d = 0; d < PhysicalDim; d++ ) {
        d_box_local[2 * d + 0] = local[2 * d + 0];
        d_box_local[2 * d + 1] = local[2 * d + 1];
        d_box[2 * d + 0]       = 0;
        d_box[2 * d + 1]       = d_globalSize[d];
    }
}


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string PureLogicalMesh::meshClass() const { return "PureLogicalMesh"; }


/****************************************************************
 * Basic functions                                               *
 ****************************************************************/
Mesh::Movable PureLogicalMesh::isMeshMovable() const { return Mesh::Movable::Fixed; }
uint64_t PureLogicalMesh::positionHash() const { return 0; }
void PureLogicalMesh::displaceMesh( const std::vector<double> & )
{
    AMP_ERROR( "displaceMesh is not supported for PureLogicalMesh" );
}
void PureLogicalMesh::displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> )
{
    AMP_ERROR( "displaceMesh is not supported for PureLogicalMesh" );
}
AMP::Geometry::Point PureLogicalMesh::physicalToLogical( const AMP::Geometry::Point &x ) const
{
    return x;
}
std::unique_ptr<Mesh> PureLogicalMesh::clone() const
{
    return std::make_unique<PureLogicalMesh>( *this );
}


/****************************************************************
 * Return the coordinates                                        *
 ****************************************************************/
void PureLogicalMesh::coord( const MeshElementIndex &index, double *pos ) const
{
    AMP_ASSERT( index.type() == AMP::Mesh::GeomType::Vertex );
    for ( int d = 0; d < PhysicalDim; d++ )
        pos[d] = index.index( d );
}
std::array<AMP::Array<double>, 3> PureLogicalMesh::localCoord() const
{
    auto local = getLocalBlock( d_comm.getRank() );
    ArraySize size( { local[1] - local[0] + 2, local[3] - local[2] + 2, local[5] - local[4] + 2 },
                    static_cast<int>( GeomDim ) );
    AMP::Array<double> x( size ), y( size ), z( size );
    for ( size_t k = 0; k < size[2]; k++ ) {
        for ( size_t j = 0; j < size[1]; j++ ) {
            for ( size_t i = 0; i < size[0]; i++ ) {
                x( i, j, k ) = i + size[0];
                y( i, j, k ) = j + size[2];
                z( i, j, k ) = k + size[4];
            }
        }
    }
    if ( PhysicalDim == 1 )
        return { x, {}, {} };
    else if ( PhysicalDim == 2 )
        return { x, y, {} };
    else
        return { x, y, z };
}
std::array<AMP::Array<double>, 3> PureLogicalMesh::globalCoord() const
{
    ArraySize size( { d_globalSize[0] + 1, d_globalSize[1] + 1, d_globalSize[2] + 1 },
                    static_cast<int>( GeomDim ) );
    AMP::Array<double> x( size ), y( size ), z( size );
    for ( size_t k = 0; k < size[2]; k++ ) {
        for ( size_t j = 0; j < size[1]; j++ ) {
            for ( size_t i = 0; i < size[0]; i++ ) {
                x( i, j, k ) = i;
                y( i, j, k ) = j;
                z( i, j, k ) = k;
            }
        }
    }
    if ( PhysicalDim == 1 )
        return { x, {}, {} };
    else if ( PhysicalDim == 2 )
        return { x, y, {} };
    else
        return { x, y, z };
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool PureLogicalMesh::operator==( const Mesh &rhs ) const
{
    // Check base class variables
    if ( !BoxMesh::operator==( rhs ) )
        return false;
    // Check if we can cast to a PureLogicalMesh
    auto mesh = dynamic_cast<const PureLogicalMesh *>( &rhs );
    if ( !mesh )
        return false;
    return true;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void PureLogicalMesh::writeRestart( int64_t fid ) const { BoxMesh::writeRestart( fid ); }
PureLogicalMesh::PureLogicalMesh( int64_t fid, AMP::IO::RestartManager *manager )
    : BoxMesh( fid, manager )
{
}


} // namespace AMP::Mesh
