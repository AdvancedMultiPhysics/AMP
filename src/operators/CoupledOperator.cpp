#include "CoupledOperator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"

#include "ProfilerApp.h"

#include <vector>


namespace AMP {
namespace Operator {


CoupledOperator::CoupledOperator( const std::shared_ptr<OperatorParameters> &params )
    : ColumnOperator( params )
{
    std::shared_ptr<CoupledOperatorParameters> myparams =
        std::dynamic_pointer_cast<CoupledOperatorParameters>( params );
    d_Operators.push_back( myparams->d_NodeToGaussPointOperator );
    d_Operators.push_back( myparams->d_CopyOperator );
    d_Operators.push_back( myparams->d_MapOperator );
    d_Operators.push_back( myparams->d_BVPOperator );
}


void CoupledOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE_START( "apply" );
    // Fill the gauss-point vector if necessary
    if ( d_Operators[0] ) {
        d_Operators[0]->apply( u, d_frozenGaussPointVector );
    }
    // Call copy vector
    if ( d_Operators[1] ) {
        if ( d_Operators[0] ) {
            d_Operators[1]->apply( d_frozenGaussPointVector, r );
        } else {
            d_Operators[1]->apply( u, r );
        }
    }
    // Call the map
    if ( d_Operators[2] ) {
        d_Operators[2]->apply( u, r );
    }
    // Call the operator
    d_Operators[3]->apply( u, r );
    PROFILE_STOP( "apply" );
}

void CoupledOperator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                                AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                AMP::LinearAlgebra::Vector::shared_ptr r )
{
    this->apply( u, r );

    //    AMP::LinearAlgebra::Vector::shared_ptr rInternal = subsetOutputVector( r );
    //    AMP_INSIST( ( rInternal.get() != nullptr ), "rInternal is NULL" );

    // the rhs can be NULL
    if ( f.get() != nullptr ) {
        //        AMP::LinearAlgebra::Vector::const_shared_ptr fInternal = subsetOutputVector( f );
        r->subtract( f, r );
    } else {
        r->scale( -1.0 );
    }

    r->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
}
} // namespace Operator
} // namespace AMP