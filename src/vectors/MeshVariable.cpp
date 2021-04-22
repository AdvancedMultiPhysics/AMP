#ifdef USE_AMP_MESH
#include "MeshVariable.h"

#include <utility>

namespace AMP {
namespace LinearAlgebra {


MeshVariable::MeshVariable( const std::string &name,
                            AMP::Mesh::Mesh::shared_ptr mesh,
                            bool useMeshComm )
    : SubsetVariable( name )
{
    AMP_ASSERT( mesh );
    d_mesh        = mesh;
    d_useMeshComm = useMeshComm;
}


AMP::Discretization::DOFManager::shared_ptr
MeshVariable::getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( d_mesh, d_useMeshComm );
}


MeshIteratorVariable::MeshIteratorVariable( const std::string &name,
                                            const AMP::Mesh::MeshIterator &iterator,
                                            const AMP_MPI &comm )
    : SubsetVariable( name ), d_comm( std::move( comm ) ), d_iterator( iterator )
{
}


AMP::Discretization::DOFManager::shared_ptr MeshIteratorVariable::getSubsetDOF(
    std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( d_iterator, d_comm );
}


} // namespace LinearAlgebra
} // namespace AMP

#endif
