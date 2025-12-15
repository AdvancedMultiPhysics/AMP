#include "AMP/vectors/MeshVariable.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/VectorSelector.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * MeshVariable                                                  *
 ****************************************************************/
MeshVariable::MeshVariable( const std::string &name,
                            std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                            bool useMeshComm )
    : SubsetVariable( name )
{
    AMP_ASSERT( mesh );
    d_mesh        = mesh;
    d_useMeshComm = useMeshComm;
}
std::shared_ptr<AMP::Discretization::DOFManager>
MeshVariable::getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( d_mesh, d_useMeshComm );
}
AMP::AMP_MPI MeshVariable::getComm( const AMP::AMP_MPI &comm ) const { return comm; }
std::shared_ptr<VectorSelector> MeshVariable::createVectorSelector() const
{
    return std::make_shared<VS_Mesh>( d_mesh, d_useMeshComm );
}
uint64_t MeshVariable::getID() const
{
    return d_mesh->getComm().bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/****************************************************************
 * MeshVariable                                                  *
 ****************************************************************/
MeshIteratorVariable::MeshIteratorVariable( const std::string &name,
                                            const AMP::Mesh::MeshIterator &iterator,
                                            const AMP_MPI &comm )
    : SubsetVariable( name ), d_comm( std::move( comm ) ), d_iterator( iterator )
{
}
std::shared_ptr<AMP::Discretization::DOFManager> MeshIteratorVariable::getSubsetDOF(
    std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( d_iterator, d_comm );
}
AMP::AMP_MPI MeshIteratorVariable::getComm( const AMP::AMP_MPI &comm ) const { return comm; }
std::shared_ptr<VectorSelector> MeshIteratorVariable::createVectorSelector() const
{
    return std::make_shared<VS_MeshIterator>( d_iterator, d_comm );
}
uint64_t MeshIteratorVariable::getID() const
{
    return d_comm.bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/****************************************************************
 * Restart                                                       *
 ****************************************************************/
MeshVariable::MeshVariable( int64_t fid ) : SubsetVariable( fid ) { AMP_ERROR( "Not finished" ); }
void MeshVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}
MeshIteratorVariable::MeshIteratorVariable( int64_t fid ) : SubsetVariable( fid )
{
    AMP_ERROR( "Not finished" );
}
void MeshIteratorVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}


} // namespace AMP::LinearAlgebra
