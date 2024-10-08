#include "AMP/operators/CoupledOperator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"

#include <vector>


namespace AMP::Operator {


static inline AMP::LinearAlgebra::UpdateState
getState( AMP::LinearAlgebra::Vector::const_shared_ptr u )
{
    if ( u )
        return u->getUpdateStatus();
    return AMP::LinearAlgebra::UpdateState::UNCHANGED;
}
static inline void checkState( AMP::LinearAlgebra::UpdateState initial,
                               AMP::LinearAlgebra::Vector::const_shared_ptr u,
                               std::shared_ptr<const Operator> op )
{
    auto UNCHANGED = AMP::LinearAlgebra::UpdateState::UNCHANGED;
    if ( initial == UNCHANGED && u )
        AMP_INSIST( u->getUpdateStatus() == UNCHANGED,
                    op->type() + " left vector in an inconsistent state" );
}


CoupledOperator::CoupledOperator( std::shared_ptr<const OperatorParameters> params )
    : ColumnOperator( params )
{
    auto myparams = std::dynamic_pointer_cast<const CoupledOperatorParameters>( params );
    d_operators.push_back( myparams->d_NodeToGaussPointOperator );
    d_operators.push_back( myparams->d_CopyOperator );
    d_operators.push_back( myparams->d_MapOperator );
    d_operators.push_back( myparams->d_BVPOperator );
}


void CoupledOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );
    AMP_ASSERT( getState( u ) == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    // Fill the gauss-point vector if necessary
    if ( d_operators[0] ) {
        auto state = getState( d_frozenGaussPointVector );
        d_operators[0]->apply( u, d_frozenGaussPointVector );
        checkState( state, d_frozenGaussPointVector, d_operators[0] );
    }
    // Call copy vector
    auto state = getState( r );
    if ( d_operators[1] ) {
        if ( d_operators[0] ) {
            d_operators[1]->apply( d_frozenGaussPointVector, r );
        } else {
            d_operators[1]->apply( u, r );
        }
        checkState( state, r, d_operators[1] );
    }
    // Call the map
    if ( d_operators[2] ) {
        d_operators[2]->apply( u, r );
        checkState( state, r, d_operators[2] );
    }
    // Call the operator
    d_operators[3]->apply( u, r );
    checkState( state, r, d_operators[3] );
}

void CoupledOperator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                                AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                AMP::LinearAlgebra::Vector::shared_ptr r )
{
    this->apply( u, r );

    // the rhs can be NULL
    if ( f ) {
        //        AMP::LinearAlgebra::Vector::const_shared_ptr fInternal = subsetOutputVector( f );
        r->subtract( *f, *r );
    } else {
        r->scale( -1.0 );
    }

    r->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}
} // namespace AMP::Operator
