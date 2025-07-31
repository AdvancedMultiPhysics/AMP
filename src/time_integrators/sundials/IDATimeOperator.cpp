#include "AMP/time_integrators/sundials/IDATimeOperator.h"


namespace AMP::TimeIntegrator {

IDATimeOperator::IDATimeOperator( std::shared_ptr<AMP::Operator::OperatorParameters> in_params )
    : TimeOperator( in_params ), d_current_time( 0 )
{
    d_cloningHappened = false;
    // BP, commenting out because this class does not define
    // a reset and the reset called is the reset for the base
    // class which already has been called once
    //        reset(in_params);
}

IDATimeOperator::~IDATimeOperator() = default;


/*
void
IDATimeOperator::reset( std::shared_ptr<OperatorParameters> in_params)
{
    auto params = std::dynamic_pointer_cast<IDATimeOperatorParameters>(in_params);
    getFromInput(params->d_db);
}
*/

void IDATimeOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( u )
        AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    if ( d_cloningHappened == 0 ) {
        d_pScratchVector = r->clone();
        d_pScratchVector->zero();
        d_cloningHappened = true;
    }

    d_pMassOperator->apply( d_pIDATimeDerivative, d_pScratchVector );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "Output of M * yp in IDATimeOperator" << std::endl;
        AMP::pout << d_pScratchVector << std::endl;
    }

    if ( d_pAlgebraicVariable ) {
        auto algebraicComponent = d_pScratchVector->subsetVectorForVariable( d_pAlgebraicVariable );
        algebraicComponent->zero();
    }

    d_pRhsOperator->apply( u, r );
    r->axpby( 1.0, 1.0, *d_pScratchVector );

    bool dpSourceTermNull = ( d_pSourceTerm.get() == nullptr );

    if ( d_iDebugPrintInfoLevel > 5 ) {
        AMP::pout << "Output of M * yp-frhs(y,t) in IDATimeOperator" << std::endl;
        AMP::pout << r << std::endl;
    }

    if ( !dpSourceTermNull ) {
        r->axpby( -1.0, 1.0, *d_pSourceTerm );
    }

    if ( d_iDebugPrintInfoLevel > 6 ) {
        AMP::pout << "Output of M * yp-frhs(y,t)-g in IDATimeOperator" << std::endl;
        AMP::pout << r << std::endl;
    }
}

void IDATimeOperator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                                AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                AMP::LinearAlgebra::Vector::shared_ptr r )
{
    AMP_INSIST( u, "NULL Solution Vector" );
    AMP_INSIST( r, "NULL Residual Vector" );

    apply( u, r );

    auto rInternal = subsetOutputVector( r );
    AMP_INSIST( ( rInternal ), "rInternal is NULL" );

    // the rhs can be NULL
    if ( f ) {
        auto fInternal = subsetOutputVector( f );
        rInternal->subtract( *fInternal, *rInternal );
    } else {
        rInternal->scale( -1.0 );
    }

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

std::shared_ptr<AMP::Operator::OperatorParameters>
IDATimeOperator::getParameters( const std::string &type,
                                AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                std::shared_ptr<AMP::Operator::OperatorParameters> params )
{

    auto timeOperator_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
    auto Overwrite       = AMP::Database::Check::Overwrite;
    timeOperator_db->putScalar( "CurrentDt", d_dCurrentDt, {}, Overwrite );
    timeOperator_db->putScalar( "name", "TimeOperator", {}, Overwrite );
    timeOperator_db->putScalar( "bLinearMassOperator", d_bLinearMassOperator, {}, Overwrite );
    timeOperator_db->putScalar( "bLinearRhsOperator", d_bLinearRhsOperator, {}, Overwrite );
    timeOperator_db->putScalar( "bAlgebraicComponent", d_bAlgebraicComponent, {}, Overwrite );
    timeOperator_db->putScalar( "ScalingFactor", 1.0 / d_dCurrentDt, {}, Overwrite );

    auto timeOperatorParameters =
        std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOperator_db );

    timeOperatorParameters->d_Mesh = d_Mesh;
    // if we have a linear rhs operator then just pass the pointer to the rhs operator instead of
    // the parameter object
    if ( d_bLinearRhsOperator ) {
        timeOperatorParameters->d_pRhsOperator = d_pRhsOperator;
    } else {
        timeOperatorParameters->d_pRhsOperatorParameters =
            d_pRhsOperator->getParameters( type, u, params );
    }

    if ( !d_bAlgebraicComponent ) {
        // if we have a linear mass operator then just pass the pointer to the mass operator instead
        // of the parameter
        // object
        if ( d_bLinearMassOperator ) {
            timeOperatorParameters->d_pMassOperator = d_pMassOperator;
        } else {
            timeOperatorParameters->d_pMassOperatorParameters =
                d_pMassOperator->getParameters( type, u, params );
        }
    }

    return timeOperatorParameters;
}
} // namespace AMP::TimeIntegrator
