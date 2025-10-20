#include "AMP/mesh/structured/MovableBoxMesh.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/mesh/structured/structuredMeshIterator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Mesh {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
MovableBoxMesh::MovableBoxMesh( const AMP::Mesh::BoxMesh &mesh ) : BoxMesh( mesh ), d_pos_hash( 0 )
{
    if ( dynamic_cast<const MovableBoxMesh *>( &mesh ) ) {
        // We are copying another MovableBoxMesh
        auto rhs = dynamic_cast<const MovableBoxMesh &>( mesh );
        d_first  = rhs.d_first;
        d_last   = rhs.d_last;
        d_coord  = rhs.d_coord;
    } else {
        // Get the ghost box
        auto box   = getLocalBlock( d_rank );
        auto range = getIteratorRange( box, GeomType::Vertex, d_max_gcw );
        AMP_ASSERT( range.size() == 1 );
        std::tie( d_first, d_last ) = range[0];
        // Get the coordinates for all local/ghost nodes
        AMP::ArraySize size( PhysicalDim,
                             d_last[0] - d_first[0] + 1,
                             d_last[1] - d_first[1] + 1,
                             d_last[2] - d_first[2] + 1 );
        d_coord.resize( size );
        for ( int k = d_first[2]; k <= d_last[2]; k++ ) {
            for ( int j = d_first[1]; j <= d_last[1]; j++ ) {
                for ( int i = d_first[0]; i <= d_last[0]; i++ ) {
                    auto x2 = &d_coord( 0, i - d_first[0], j - d_first[1], k - d_first[2] );
                    MeshElementIndex index( GeomType::Vertex, 0, i, j, k );
                    mesh.fixPeriodic( index );
                    mesh.coord( index, x2 );
                }
            }
        }
    }
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MovableBoxMesh::writeRestart( int64_t fid ) const
{
    BoxMesh::writeRestart( fid );
    IO::writeHDF5( fid, "pos_hash", d_pos_hash );
    IO::writeHDF5( fid, "first", d_first );
    IO::writeHDF5( fid, "last", d_last );
    IO::writeHDF5( fid, "coord", d_coord );
}
MovableBoxMesh::MovableBoxMesh( int64_t fid, AMP::IO::RestartManager *manager )
    : BoxMesh( fid, manager )
{
    IO::readHDF5( fid, "pos_hash", d_pos_hash );
    IO::readHDF5( fid, "first", d_first );
    IO::readHDF5( fid, "last", d_last );
    IO::readHDF5( fid, "coord", d_coord );
    BoxMesh::finalize( d_name, {} );
}


/****************************************************************
 * Functions to displace the mesh                                *
 ****************************************************************/
Mesh::Movable MovableBoxMesh::isMeshMovable() const { return Mesh::Movable::Deform; }
uint64_t MovableBoxMesh::positionHash() const { return d_pos_hash; }
void MovableBoxMesh::displaceMesh( const std::vector<double> &x )
{
    AMP_ASSERT( x.size() == PhysicalDim );
    size_t N = d_coord.length() / PhysicalDim;
    for ( size_t i = 0; i < N; i++ ) {
        for ( int d = 0; d < PhysicalDim; d++ )
            d_coord( d, i ) += x[d];
    }
    for ( int d = 0; d < PhysicalDim; d++ ) {
        d_box[2 * d + 0] += x[d];
        d_box[2 * d + 1] += x[d];
        d_box_local[2 * d + 0] += x[d];
        d_box_local[2 * d + 1] += x[d];
    }
    if ( d_geometry )
        d_geometry->displace( x.data() );
    d_pos_hash++;
}
void MovableBoxMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
    // Clear the geometry if it exists to ensure consistency
    d_geometry.reset();
    // Create the position vector with the necessary ghost nodes
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        shared_from_this(),
        getIterator( AMP::Mesh::GeomType::Vertex, d_max_gcw ),
        getIterator( AMP::Mesh::GeomType::Vertex, 0 ),
        PhysicalDim );
    auto nodalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "tmp_pos" );
    auto displacement  = AMP::LinearAlgebra::createVector( DOFs, nodalVariable, false );
    std::vector<size_t> dofs1( PhysicalDim );
    std::vector<size_t> dofs2( PhysicalDim );
    auto cur  = getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end  = cur.end();
    auto DOFx = x->getDOFManager();
    std::vector<double> data( PhysicalDim );
    while ( cur != end ) {
        AMP::Mesh::MeshElementID id = cur->globalID();
        DOFx->getDOFs( id, dofs1 );
        DOFs->getDOFs( id, dofs2 );
        x->getValuesByGlobalID( PhysicalDim, &dofs1[0], &data[0] );
        displacement->setValuesByGlobalID( PhysicalDim, &dofs2[0], &data[0] );
        ++cur;
    }
    displacement->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    // Move all nodes (including the ghost nodes)
    std::vector<size_t> dofs( PhysicalDim );
    for ( int k = d_first[2]; k <= d_last[2]; k++ ) {
        for ( int j = d_first[1]; j <= d_last[1]; j++ ) {
            for ( int i = d_first[0]; i <= d_last[0]; i++ ) {
                MeshElementIndex index( GeomType::Vertex, 0, i, j, k );
                fixPeriodic( index );
                MeshElementID id = structuredMeshElement( index, this ).globalID();
                DOFs->getDOFs( id, dofs );
                AMP_ASSERT( dofs.size() == PhysicalDim );
                double disp[3];
                displacement->getValuesByGlobalID( (int) PhysicalDim, &dofs[0], disp );
                auto x2 = &d_coord( 0, i - d_first[0], j - d_first[1], k - d_first[2] );
                for ( int d = 0; d < PhysicalDim; d++ )
                    x2[d] += disp[d];
            }
        }
    }
    // Compute the new bounding box of the mesh
    d_box_local = std::vector<double>( 2 * PhysicalDim );
    for ( int d = 0; d < PhysicalDim; d++ ) {
        d_box_local[2 * d + 0] = 1e100;
        d_box_local[2 * d + 1] = -1e100;
    }
    auto box   = getLocalBlock( d_rank );
    auto range = getIteratorRange( box, GeomType::Vertex, 0 );
    AMP_ASSERT( range.size() == 1 );
    auto [first, last] = range[0];
    for ( int k = first[2]; k <= last[2]; k++ ) {
        for ( int j = first[1]; j <= last[1]; j++ ) {
            for ( int i = first[0]; i <= last[0]; i++ ) {
                auto x2 = &d_coord( 0, i - d_first[0], j - d_first[1], k - d_first[2] );
                for ( int d = 0; d < PhysicalDim; d++ ) {
                    d_box_local[2 * d + 0] = std::min( d_box_local[2 * d + 0], x2[d] );
                    d_box_local[2 * d + 1] = std::max( d_box_local[2 * d + 1], x2[d] );
                }
            }
        }
    }
    d_box = Mesh::reduceBox( d_box_local, d_comm );
    d_pos_hash++;
}


/********************************************************
 * Return the class name                                 *
 ********************************************************/
std::string MovableBoxMesh::meshClass() const { return "MovableBoxMesh"; }


/****************************************************************
 * Copy the mesh                                                 *
 ****************************************************************/
std::unique_ptr<Mesh> MovableBoxMesh::clone() const
{
    return std::make_unique<MovableBoxMesh>( *this );
}


/****************************************************************
 * Return the coordinate                                         *
 ****************************************************************/
void MovableBoxMesh::coord( const MeshElementIndex &index0, double *pos ) const
{
    AMP_ASSERT( index0.type() == AMP::Mesh::GeomType::Vertex );
    auto index = index0;
    for ( int d = 0; d < PhysicalDim; d++ ) {
        if ( d_surfaceId[2 * d] == -1 ) {
            // Periodic boundary
            if ( index[d] < d_first[d] )
                index[d] += d_globalSize[d];
            else if ( index[d] > d_last[d] )
                index[d] -= d_globalSize[d];
        }
    }
    AMP_DEBUG_ASSERT( index[0] >= d_first[0] && index[0] <= d_last[0] && index[1] >= d_first[1] &&
                      index[1] <= d_last[1] && index[2] >= d_first[2] && index[2] <= d_last[2] );
    auto x2 = &d_coord( 0, index[0] - d_first[0], index[1] - d_first[1], index[2] - d_first[2] );
    for ( int d = 0; d < PhysicalDim; d++ )
        pos[d] = x2[d];
}


/****************************************************************
 * Return the logical coordinates                                *
 ****************************************************************/
AMP::Geometry::Point MovableBoxMesh::physicalToLogical( const AMP::Geometry::Point & ) const
{
    AMP_ERROR( "physicalToLogical is not supported in MovableBoxMesh" );
    return AMP::Geometry::Point();
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool MovableBoxMesh::operator==( const Mesh &rhs ) const
{
    // Check base class variables
    if ( !BoxMesh::operator==( rhs ) )
        return false;
    // Check if we can cast to a MovableBoxMesh
    auto mesh = dynamic_cast<const MovableBoxMesh *>( &rhs );
    if ( !mesh )
        return false;
    // Perform final comparisons
    bool test = d_first == mesh->d_first && d_last == mesh->d_last;
    test &= d_coord == mesh->d_coord;
    return test;
}


} // namespace AMP::Mesh
