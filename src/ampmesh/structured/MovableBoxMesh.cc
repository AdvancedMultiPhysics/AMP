#include "ampmesh/structured/MovableBoxMesh.h"
#include "ampmesh/structured/BoxMesh.h"
#include "ampmesh/MultiIterator.h"
#include "ampmesh/structured/structuredMeshElement.h"
#include "ampmesh/structured/structuredMeshIterator.h"

#ifdef USE_AMP_VECTORS
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"
#endif
#ifdef USE_AMP_DISCRETIZATION
#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#endif

namespace AMP {
namespace Mesh {


/****************************************************************
* Constructors                                                  *
****************************************************************/
MovableBoxMesh::MovableBoxMesh( const AMP::Mesh::BoxMesh& mesh ):
    BoxMesh( mesh )
{
    // Get a list of all nodes on the current processor
    MeshIterator nodeIterator = mesh.getIterator( Vertex, d_max_gcw );
    d_index.reserve( nodeIterator.size() );
    for ( size_t i = 0; i < nodeIterator.size(); ++i, ++nodeIterator ) {
        auto element = dynamic_cast<structuredMeshElement*>( nodeIterator->getRawElement() );
        AMP_ASSERT( element != nullptr );
        d_index.emplace_back( element->getIndex() );
    }
    AMP::Utilities::quicksort( d_index );

    // Generate coordinates
    d_coord[0].resize(d_index.size(),0);
    d_coord[1].resize(d_index.size(),0);
    d_coord[2].resize(d_index.size(),0);
    for (size_t i=0; i<d_index.size(); i++) {
        double pos[3];
        mesh.coord( d_index[i], pos );
        for (int d=0; d<PhysicalDim; d++)
            d_coord[d][i] = pos[d];
    }
}



/****************************************************************
* Functions to displace the mesh                                *
****************************************************************/
int MovableBoxMesh::isMeshMovable( ) const
{
    return 2;
}
void MovableBoxMesh::displaceMesh( const std::vector<double> &x )
{
    AMP_ASSERT( x.size() == PhysicalDim );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        for ( size_t j = 0; j < d_coord[i].size(); j++ )
            d_coord[i][j] += x[i];
        d_box[2 * i + 0] += x[i];
        d_box[2 * i + 1] += x[i];
        d_box_local[2 * i + 0] += x[i];
        d_box_local[2 * i + 1] += x[i];
    }
    if ( d_geometry != nullptr )
        d_geometry->displaceMesh( x );
}
#ifdef USE_AMP_VECTORS
void MovableBoxMesh::displaceMesh( const AMP::LinearAlgebra::Vector::const_shared_ptr x )
{
#ifdef USE_AMP_DISCRETIZATION
    // Create the position vector with the necessary ghost nodes
    AMP::Discretization::DOFManager::shared_ptr DOFs =
        AMP::Discretization::simpleDOFManager::create( shared_from_this(),
                                                       getIterator( AMP::Mesh::Vertex, d_max_gcw ),
                                                       getIterator( AMP::Mesh::Vertex, 0 ),
                                                       PhysicalDim );
    AMP::LinearAlgebra::Variable::shared_ptr nodalVariable(
        new AMP::LinearAlgebra::Variable( "tmp_pos" ) );
    AMP::LinearAlgebra::Vector::shared_ptr displacement =
        AMP::LinearAlgebra::createVector( DOFs, nodalVariable, false );
    std::vector<size_t> dofs1( PhysicalDim );
    std::vector<size_t> dofs2( PhysicalDim );
    AMP::Mesh::MeshIterator cur                      = getIterator( AMP::Mesh::Vertex, 0 );
    AMP::Mesh::MeshIterator end                      = cur.end();
    AMP::Discretization::DOFManager::shared_ptr DOFx = x->getDOFManager();
    std::vector<double> data( PhysicalDim );
    while ( cur != end ) {
        AMP::Mesh::MeshElementID id = cur->globalID();
        DOFx->getDOFs( id, dofs1 );
        DOFs->getDOFs( id, dofs2 );
        x->getValuesByGlobalID( PhysicalDim, &dofs1[0], &data[0] );
        displacement->setValuesByGlobalID( PhysicalDim, &dofs2[0], &data[0] );
        ++cur;
    }
    displacement->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
    // Move all nodes (including the ghost nodes)
    std::vector<size_t> dofs( PhysicalDim );
    std::vector<double> disp( PhysicalDim );
    for ( size_t i = 0; i < d_coord[0].size(); i++ ) {
        MeshElementID id = structuredMeshElement( d_index[i], this ).globalID();
        DOFs->getDOFs( id, dofs );
        AMP_ASSERT( dofs.size() == PhysicalDim );
        displacement->getValuesByGlobalID( (int) PhysicalDim, &dofs[0], &disp[0] );
        for ( int j = 0; j < PhysicalDim; j++ )
            d_coord[j][i] += disp[j];
    }
    // Compute the new bounding box of the mesh
    d_box_local = std::vector<double>( 2 * PhysicalDim );
    for ( int d = 0; d < PhysicalDim; d++ ) {
        d_box_local[2 * d + 0] = 1e100;
        d_box_local[2 * d + 1] = -1e100;
        for ( size_t i = 0; i < d_coord[d].size(); i++ ) {
            if ( d_coord[d][i] < d_box_local[2 * d + 0] )
                d_box_local[2 * d + 0] = d_coord[d][i];
            if ( d_coord[d][i] > d_box_local[2 * d + 1] )
                d_box_local[2 * d + 1] = d_coord[d][i];
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
#else
    AMP_ERROR( "displaceMesh requires DISCRETIZATION" );
#endif
}
#endif


/****************************************************************
* Copy the mesh                                                 *
****************************************************************/
AMP::shared_ptr<Mesh> MovableBoxMesh::copy() const
{
    return AMP::shared_ptr<MovableBoxMesh>( new MovableBoxMesh(*this) );
}


/****************************************************************
* Return the coordinate                                         *
****************************************************************/
void MovableBoxMesh::coord( const MeshElementIndex &index, double *pos ) const
{
    AMP_ASSERT( index.type() == AMP::Mesh::Vertex );
    size_t i = AMP::Utilities::findfirst( d_index, index );
    AMP_ASSERT( d_index[i] == index );
    for ( int d  = 0; d < PhysicalDim; d++ )
        pos[d] = d_coord[d][i];
}


/****************************************************************
* Return the logical coordinates                                *
****************************************************************/
std::array<double,3> MovableBoxMesh::physicalToLogical( const double* ) const
{
    AMP_ERROR("physicalToLogical is not supported in MovableBoxMesh");
    return std::array<double,3>();
}



} // Mesh namespace
} // AMP namespace