#include "AMP/time_integrators/ColumnTimeOperator.h"
#include "AMP/operators/ColumnOperatorParameters.h"
#include "AMP/time_integrators/LinearTimeOperator.h"
#include "AMP/time_integrators/TimeOperatorParameters.h"
#include "AMP/utils/Database.h"

namespace AMP::TimeIntegrator {

ColumnTimeOperator::ColumnTimeOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> in_params )
    : ColumnOperator( in_params )
{
    auto params    = std::dynamic_pointer_cast<const TimeOperatorParameters>( in_params );
    auto column_db = params->d_db;
    d_Mesh         = params->d_Mesh;
    d_pRhsOperator = std::dynamic_pointer_cast<ColumnOperator>( params->d_pRhsOperator );
    AMP_INSIST( d_pRhsOperator,
                "Error: ColumnTimeOperator::ColumnTimeOperator() rhs "
                "operator must be a non-NULL column operator" );

    d_pMassOperator = std::dynamic_pointer_cast<ColumnOperator>( params->d_pMassOperator );
    AMP_INSIST( d_pRhsOperator,
                "Error: ColumnTimeOperator::ColumnTimeOperator() "
                "mass operator must be a non-NULL column operator" );

    d_bCreateLinearTimeOperators =
        column_db->getWithDefault<bool>( "CreateLinearTimeOperators", true );

    // for now we only allow one algebraic component
    d_iAlgebraicComponent = column_db->getWithDefault<int>( "algebraicComponent", -1 );

    d_dCurrentDt = column_db->getWithDefault<double>( "CurrentTime", 0.0 );

    if ( d_bCreateLinearTimeOperators ) {
        const int numberOfOperators = d_pRhsOperator->getNumberOfOperators();

        for ( int i = 0; i < numberOfOperators; i++ ) {
            auto timeOp_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
            // we assume for now that either all operators in the column operator are linear or all
            // are nonlinear
            auto Overwrite = AMP::Database::Check::Overwrite;
            auto currentDt = column_db->getWithDefault<double>( "CurrentDt", 1.0e-08 );
            auto currentT  = column_db->getWithDefault<double>( "CurrentTime", 0.0 );
            auto scale     = column_db->getWithDefault<double>( "ScalingFactor", 1.0e6 );
            bool massOp    = column_db->getWithDefault<bool>( "bLinearMassOperator", true );
            bool rhsOp     = column_db->getWithDefault<bool>( "bLinearRhsOperator", false );
            timeOp_db->putScalar( "CurrentDt", currentDt, {}, Overwrite );
            timeOp_db->putScalar( "CurrentTime", currentT, {}, Overwrite );
            timeOp_db->putScalar( "name", "TimeOperator", {}, Overwrite );
            timeOp_db->putScalar( "bLinearMassOperator", massOp, {}, Overwrite );
            timeOp_db->putScalar( "bLinearRhsOperator", rhsOp, {}, Overwrite );
            timeOp_db->putScalar( "ScalingFactor", scale, {}, Overwrite );
            auto timeOpParams =
                std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOp_db );
            timeOpParams->d_pRhsOperator = d_pRhsOperator->getOperator( i );

            // if there are algebraic components set the mass operator to NULL
            if ( i != d_iAlgebraicComponent ) {
                timeOpParams->d_pMassOperator = d_pMassOperator->getOperator( i );
            } else {
                timeOp_db->putScalar( "bAlgebraicComponent", true, {}, Overwrite );
            }

            auto op = std::make_shared<AMP::TimeIntegrator::LinearTimeOperator>( timeOpParams );

            d_operators.push_back( op );
        }
    } else {
        AMP::pout << "Error: ColumnTimeOperator::ColumnTimeOperator() currently implemented for "
                     "column rhs and mass "
                     "operators where all components are LinearOperators"
                  << std::endl;
    }

    reset( in_params );

    d_pSourceTerm = params->d_pSourceTerm;
}

ColumnTimeOperator::~ColumnTimeOperator() = default;

void ColumnTimeOperator::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> in_params )
{
    auto params         = std::dynamic_pointer_cast<const TimeOperatorParameters>( in_params );
    auto column_db      = params->d_db;
    auto pRhsParameters = std::dynamic_pointer_cast<AMP::Operator::ColumnOperatorParameters>(
        params->d_pRhsOperatorParameters );
    auto pMassParameters = std::dynamic_pointer_cast<AMP::Operator::ColumnOperatorParameters>(
        params->d_pMassOperatorParameters );

    AMP_INSIST( params, "Error: NULL TimeOperatorParameters object" );

    getFromInput( params->d_db );

    const int numberOfOperators = d_pRhsOperator->getNumberOfOperators();

    for ( int i = 0; i < numberOfOperators; i++ ) {
        auto timeOp_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
        // we assume for now that either all operators in the column operator are linear or all are
        // nonlinear
        auto Overwrite = AMP::Database::Check::Overwrite;
        auto currentDt = column_db->getWithDefault<double>( "CurrentDt", 1.0e-08 );
        auto currentT  = column_db->getWithDefault<double>( "CurrentTime", 0.0 );
        auto scale     = column_db->getWithDefault<double>( "ScalingFactor", 1.0e6 );
        bool massOp    = column_db->getWithDefault<bool>( "bLinearMassOperator", true );
        bool rhsOp     = column_db->getWithDefault<bool>( "bLinearRhsOperator", false );
        timeOp_db->putScalar( "CurrentDt", currentDt, {}, Overwrite );
        timeOp_db->putScalar( "CurrentTime", currentT, {}, Overwrite );
        timeOp_db->putScalar( "name", "TimeOperator", {}, Overwrite );
        timeOp_db->putScalar( "bLinearMassOperator", massOp, {}, Overwrite );
        timeOp_db->putScalar( "bLinearRhsOperator", rhsOp, {}, Overwrite );
        timeOp_db->putScalar( "ScalingFactor", scale, {}, Overwrite );
        auto timeOpParams =
            std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOp_db );
        if ( pRhsParameters ) {
            timeOpParams->d_pRhsOperatorParameters = ( pRhsParameters->d_OperatorParameters )[i];
        }

        // if there are algebraic components set the mass operator to NULL
        if ( i != d_iAlgebraicComponent ) {
            if ( pMassParameters ) {
                timeOpParams->d_pMassOperatorParameters =
                    ( pMassParameters->d_OperatorParameters )[i];
            }
        } else {
            timeOp_db->putScalar( "bAlgebraicComponent", true, {}, Overwrite );
        }

        d_operators[i]->reset( timeOpParams );
    }
}

void ColumnTimeOperator::getFromInput( std::shared_ptr<AMP::Database> db )
{
    AMP_INSIST( db->keyExists( "CurrentDt" ), "key CurrentDt missing in input" );

    d_dCurrentDt = db->getScalar<double>( "CurrentDt" );
}

void ColumnTimeOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr /* u */,
                                AMP::LinearAlgebra::Vector::shared_ptr /* f */ )
{
    AMP_ERROR( "Not Finished" );
}

void ColumnTimeOperator::append( std::shared_ptr<Operator> /* op */ )
{
    AMP::pout
        << "Error: ColumnTimeOperator::append(): this routine is disabled for ColumnTimeOperators"
        << std::endl;
}
} // namespace AMP::TimeIntegrator
