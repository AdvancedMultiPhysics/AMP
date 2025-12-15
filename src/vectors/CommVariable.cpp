#include "AMP/vectors/CommVariable.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/VectorSelector.h"


namespace AMP::LinearAlgebra {


CommVariable::CommVariable( const std::string &name, const AMP_MPI &comm ) : SubsetVariable( name )
{
    AMP_ASSERT( !comm.isNull() );
    d_comm = comm;
}


std::shared_ptr<AMP::Discretization::DOFManager>
CommVariable::getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( d_comm );
}


AMP::AMP_MPI CommVariable::getComm( const AMP::AMP_MPI &comm ) const
{
    return AMP::AMP_MPI::intersect( d_comm, comm );
}


std::shared_ptr<VectorSelector> CommVariable::createVectorSelector() const
{
    return std::make_shared<VS_Comm>( d_comm );
}


uint64_t CommVariable::getID() const
{
    return d_comm.bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/****************************************************************
 * Restart                                                       *
 ****************************************************************/
void CommVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}

CommVariable::CommVariable( int64_t fid ) : SubsetVariable( fid ) { AMP_ERROR( "Not finished" ); }


} // namespace AMP::LinearAlgebra
