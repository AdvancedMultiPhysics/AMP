#include "SolverStrategy.h"
#include "AMP/utils/Utilities.h"


namespace AMP {
namespace Solver {


int SolverStrategy::d_iInstanceId = 0;


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
SolverStrategy::SolverStrategy()
{
    d_iNumberIterations    = -1;
    d_dResidualNorm        = -1;
    d_bUseZeroInitialGuess = true;
    d_iDebugPrintInfoLevel = 0;
    d_iMaxIterations       = 0;
    d_dMaxRhs              = 0;
    d_dMaxError            = 0;
    d_iObjectId            = 0;
}
SolverStrategy::SolverStrategy( std::shared_ptr<SolverStrategyParameters> parameters )
{
    AMP_INSIST( parameters.get() != nullptr, "NULL SolverStrategyParameters object" );

    d_iObjectId            = SolverStrategy::d_iInstanceId;
    d_iNumberIterations    = -1;
    d_dResidualNorm        = -1;
    d_dMaxRhs              = 1.0;
    d_bUseZeroInitialGuess = true;

    d_pOperator = parameters->d_pOperator;

    SolverStrategy::d_iInstanceId++;
    SolverStrategy::getFromInput( parameters->d_db );
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
SolverStrategy::~SolverStrategy() = default;


/****************************************************************
 * Initialize                                                    *
 ****************************************************************/
void SolverStrategy::getFromInput( std::shared_ptr<AMP::Database> db )
{
    AMP_INSIST( db.get() != nullptr, "InputDatabase object must be non-NULL" );
    d_iMaxIterations       = db->getWithDefault( "max_iterations", 1 );
    d_dMaxError            = db->getWithDefault<double>( "max_error", 1.0e-12 );
    d_iDebugPrintInfoLevel = db->getWithDefault( "print_info_level", 0 );
    d_bUseZeroInitialGuess = db->getWithDefault( "zero_initial_guess", true );
}
void SolverStrategy::initialize( std::shared_ptr<SolverStrategyParameters> const parameters )
{
    AMP_INSIST( parameters.get() != nullptr, "SolverStrategyParameters object cannot be NULL" );
}


/****************************************************************
 * Reset                                                         *
 ****************************************************************/
void SolverStrategy::resetOperator(
    const std::shared_ptr<AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator.get() != nullptr ) {
        d_pOperator->reset( params );
    }
}

void SolverStrategy::reset( std::shared_ptr<SolverStrategyParameters> ) {}


/****************************************************************
 * Set properties                                                *
 ****************************************************************/
void SolverStrategy::setConvergenceTolerance( const int max_iterations, const double max_error )
{
    AMP_INSIST( max_iterations >= 0, "max_iterations must be non-negative" );
    AMP_INSIST( max_error >= 0.0, "max_eror must be non-negative" );
    d_iMaxIterations = max_iterations;
    d_dMaxError      = max_error;
}

void SolverStrategy::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> ) {}

} // namespace Solver
} // namespace AMP