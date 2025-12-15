#include "AMP/vectors/CommSelfVariable.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/subsetCommSelfDOFManager.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/data/SubsetCommSelfVectorData.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
CommSelfVariable::CommSelfVariable( const std::string &name ) : SubsetVariable( name ) {}


/****************************************************************
 * Basic functions                                               *
 ****************************************************************/
std::shared_ptr<AMP::Discretization::DOFManager>
CommSelfVariable::getSubsetDOF( std::shared_ptr<AMP::Discretization::DOFManager> parentDOF ) const
{
    return parentDOF->subset( AMP_COMM_SELF );
}
AMP::AMP_MPI CommSelfVariable::getComm( const AMP::AMP_MPI & ) const { return AMP_COMM_SELF; }
std::shared_ptr<VectorSelector> CommSelfVariable::createVectorSelector() const
{
    return std::make_shared<VS_Comm>( AMP_COMM_SELF );
}
uint64_t CommSelfVariable::getID() const { return reinterpret_cast<uint64_t>( this ); }


/****************************************************************
 * view                                                          *
 ****************************************************************/
Vector::shared_ptr CommSelfVariable::view( Vector::shared_ptr v ) const
{
    if ( v->getComm().getSize() == 1 )
        return v;
    if ( std::dynamic_pointer_cast<MultiVector>( v ) ) {
        // We are dealing with a multivector
        // It is more efficient to subset the individual pieces, then create a new mulitvector
        auto mv = std::dynamic_pointer_cast<MultiVector>( v );
        std::vector<Vector::shared_ptr> vec_list( mv->getNumberOfSubvectors() );
        for ( size_t i = 0; i < vec_list.size(); i++ )
            vec_list[i] = view( mv->getVector( i ) );
        return std::make_shared<MultiVector>( v->getName(), AMP_COMM_SELF, vec_list );
    }
    // Create the new subset vector
    auto parentDOF = v->getDOFManager();
    auto subsetDOF = std::make_shared<AMP::Discretization::subsetCommSelfDOFManager>( parentDOF );
    auto var       = std::const_pointer_cast<Variable>( shared_from_this() );
    auto ops       = v->getVectorOperations();
    auto data      = std::make_shared<SubsetCommSelfVectorData>( v->getVectorData() );
    auto retVal    = std::make_shared<Vector>( data, ops, var, subsetDOF );
    return retVal;
}


/****************************************************************
 * Restart                                                       *
 ****************************************************************/
void CommSelfVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}

CommSelfVariable::CommSelfVariable( int64_t fid ) : SubsetVariable( fid )
{
    AMP_ERROR( "Not finished" );
}


} // namespace AMP::LinearAlgebra
