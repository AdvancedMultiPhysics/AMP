#include "AMP/time_integrators/LinearTimeOperator.h"
#include "AMP/time_integrators/TimeOperatorParameters.h"
#include "AMP/utils/Database.h"


namespace AMP::TimeIntegrator {

LinearTimeOperator::LinearTimeOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> in_params )
    : LinearOperator( in_params )
{
    d_bModifyRhsOperatorMatrix = false;
    d_bAlgebraicComponent      = false;

    d_dScalingFactor = 0.0;
    d_dCurrentDt     = 0.0;

    auto params = std::dynamic_pointer_cast<const TimeOperatorParameters>( in_params );

    d_pRhsOperator  = std::dynamic_pointer_cast<LinearOperator>( params->d_pRhsOperator );
    d_pMassOperator = std::dynamic_pointer_cast<LinearOperator>( params->d_pMassOperator );

    d_current_time = 0.0;
    d_beta         = 1.0;
    reset( in_params );
}

LinearTimeOperator::~LinearTimeOperator() = default;

void LinearTimeOperator::getFromInput( std::shared_ptr<AMP::Database> db )
{

    AMP_INSIST( db->keyExists( "ScalingFactor" ), "key ScalingFactor missing in input" );

    d_dScalingFactor      = db->getScalar<double>( "ScalingFactor" );
    d_current_time        = db->getWithDefault<double>( "CurrentTime", 0.0 );
    d_bAlgebraicComponent = db->getWithDefault<bool>( "bAlgebraicComponent", false );
}

void LinearTimeOperator::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> in_params )
{
    auto params = std::dynamic_pointer_cast<const TimeOperatorParameters>( in_params );

    getFromInput( params->d_db );

    // the parameter object for the rhs operator will be NULL during construction, but should
    // not be NULL during a reset based on getJacobianParameters
    if ( params->d_pRhsOperatorParameters ) {
        d_pRhsOperator->reset( params->d_pRhsOperatorParameters );
    }

    // the parameter object for the mass operator will be NULL during construction, but should
    // not be NULL during a reset based on getJacobianParameters
    if ( params->d_pMassOperatorParameters ) {
        d_pMassOperator->reset( params->d_pMassOperatorParameters );
    }

    auto pRhsOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pRhsOperator );
    AMP_INSIST( pRhsOperator, "ERROR: RhsOperator is not of type LinearOperator" );

    auto pMatrix = pRhsOperator->getMatrix();

    if ( d_bModifyRhsOperatorMatrix ) {
        AMP_INSIST( pMatrix, "ERROR: NULL matrix pointer" );
        setMatrix( pMatrix );
    } else {
        // if it's not okay to modify the rhs matrix then copy it over
        if ( !d_matrix ) {
            AMP_INSIST( pMatrix, "ERROR: NULL matrix pointer" );
            d_matrix = pMatrix->clone();
            AMP_INSIST( d_matrix, "ERROR: NULL matrix pointer" );
        }

        d_matrix->zero();
        d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

        d_matrix->axpy( 1.0, *pMatrix );
    }

    if ( !d_bAlgebraicComponent ) {
        auto pMassOperator =
            std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pMassOperator );
        AMP_INSIST( pMassOperator, "ERROR: MassOperator is not of type LinearOperator" );

        std::shared_ptr<AMP::LinearAlgebra::Matrix> pMassMatrix = pMassOperator->getMatrix();

        AMP_INSIST( pMassMatrix, "ERROR: NULL matrix pointer" );
        // update the matrix to incorporate the contribution from the time derivative
        d_matrix->axpy( d_dScalingFactor, *pMassMatrix );
    }

    d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

std::shared_ptr<AMP::Operator::OperatorParameters>
LinearTimeOperator::getParameters( const std::string &type,
                                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                   std::shared_ptr<AMP::Operator::OperatorParameters> params )
{
    auto timeOperator_db = std::make_shared<AMP::Database>( "LinearTimeOperatorDatabase" );
    auto Overwrite       = AMP::Database::Check::Overwrite;
    timeOperator_db->putScalar( "CurrentDt", d_dCurrentDt, {}, Overwrite );
    timeOperator_db->putScalar( "name", "LinearTimeOperator", {}, Overwrite );
    timeOperator_db->putScalar( "ScalingFactor", 1.0 / d_dCurrentDt, {}, Overwrite );

    auto timeOperatorParameters =
        std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOperator_db );
    timeOperatorParameters->d_pRhsOperatorParameters =
        d_pRhsOperator->getParameters( type, u, params );
    timeOperatorParameters->d_pMassOperatorParameters =
        d_pMassOperator->getParameters( type, u, params );

    return timeOperatorParameters;
}
} // namespace AMP::TimeIntegrator
