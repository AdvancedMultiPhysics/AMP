#include "AMP/vectors/SubsetVariable.h"
#include "AMP/discretization/subsetCommSelfDOFManager.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/SubsetCommSelfVectorData.h"
#include "AMP/vectors/data/SubsetVectorData.h"
#include "AMP/vectors/data/VectorDataIterator.h"

#include "ProfilerApp.h"

#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * createVectorSelector                                          *
 ****************************************************************/
std::shared_ptr<VectorSelector> SubsetVariable::createVectorSelector() const
{
    AMP_ERROR( "createVectorSelector is not available for SubsetVariabl" );
    return nullptr;
}


/****************************************************************
 * view                                                          *
 ****************************************************************/
Vector::shared_ptr SubsetVariable::view( Vector::shared_ptr v ) const
{
    if ( std::dynamic_pointer_cast<MultiVector>( v ) ) {
        // We are dealing with a multivector
        // It is more efficient to subset the individual pieces, then create a new mulitvector
        auto mv = std::dynamic_pointer_cast<MultiVector>( v );
        std::vector<Vector::shared_ptr> vec_list;
        for ( size_t i = 0; i < mv->getNumberOfSubvectors(); i++ ) {
            auto sub_vec = SubsetVariable::view( mv->getVector( i ) );
            if ( sub_vec != nullptr )
                vec_list.push_back( sub_vec );
        }
        if ( vec_list.empty() ) {
            return {};
        }
        auto comm = getComm( v->getDOFManager()->getComm() );
        return std::make_shared<MultiVector>( v->getName(), comm, vec_list );
    }
    // Subset the DOFManager and create a new communication list
    auto parentDOF = v->getDOFManager();
    auto subsetDOF = getSubsetDOF( parentDOF );
    if ( !subsetDOF ) {
        return Vector::shared_ptr();
    } else if ( subsetDOF->numGlobalDOF() == 0 ) {
        return Vector::shared_ptr();
    } else if ( subsetDOF == parentDOF ) {
        return v;
    }
    auto var = std::const_pointer_cast<Variable>( shared_from_this() );
    if ( std::dynamic_pointer_cast<AMP::Discretization::subsetCommSelfDOFManager>( subsetDOF ) ) {
        // Create the new subset vector
        auto ops    = v->getVectorOperations();
        auto data   = std::make_shared<SubsetCommSelfVectorData>( v->getVectorData() );
        auto retVal = std::make_shared<Vector>( data, ops, var, subsetDOF );
        return retVal;
    } else {
        auto remote_DOFs = subsetDOF->getRemoteDOFs();
        bool ghosts      = subsetDOF->getComm().anyReduce( !remote_DOFs.empty() );
        std::shared_ptr<CommunicationList> commList;
        if ( !ghosts ) {
            commList = std::make_shared<CommunicationList>( subsetDOF->numLocalDOF(),
                                                            subsetDOF->getComm() );
        } else {
            // Construct the communication list
            auto params    = std::make_shared<AMP::LinearAlgebra::CommunicationListParameters>();
            params->d_comm = subsetDOF->getComm();
            params->d_localsize   = subsetDOF->numLocalDOF();
            params->d_remote_DOFs = remote_DOFs;
            commList = std::make_shared<AMP::LinearAlgebra::CommunicationList>( params );
        }
        // Create the new subset vector
        auto ops             = std::make_shared<VectorOperationsDefault<double>>();
        auto params          = std::make_shared<SubsetVectorParameters>();
        params->d_ViewVector = std::const_pointer_cast<Vector>( v );
        params->d_DOFManager = subsetDOF;
        params->d_CommList   = commList;
        auto data            = std::make_shared<SubsetVectorData>( params );
        auto retVal          = std::make_shared<Vector>( data, ops, var, subsetDOF );
        return retVal;
    }
}
Vector::const_shared_ptr SubsetVariable::view( Vector::const_shared_ptr v ) const
{
    return SubsetVariable::view( std::const_pointer_cast<Vector>( v ) );
}


/****************************************************************
 * Restart                                                       *
 ****************************************************************/
SubsetVariable::SubsetVariable( int64_t fid ) : Variable( fid ) { AMP_ERROR( "Not finished" ); }
void SubsetVariable::writeRestart( int64_t fid ) const
{
    Variable::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}


} // namespace AMP::LinearAlgebra
