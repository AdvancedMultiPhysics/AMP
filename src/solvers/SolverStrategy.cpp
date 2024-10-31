#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/Utilities.h"

#include <cmath>
#include <numeric>

namespace AMP::Solver {


int SolverStrategy::d_iInstanceId = 0;


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
SolverStrategy::SolverStrategy()
    : d_dResidualNorm{ -1.0 }, d_iObjectId{ SolverStrategy::d_iInstanceId }
{
    SolverStrategy::d_iInstanceId++;
}

SolverStrategy::SolverStrategy( std::shared_ptr<const SolverStrategyParameters> parameters )
    : d_dResidualNorm{ -1.0 },
      d_iObjectId{ SolverStrategy::d_iInstanceId },
      d_db( parameters->d_db ),
      d_global_db( parameters->d_global_db )
{
    AMP_INSIST( parameters, "NULL SolverStrategyParameters object" );
    SolverStrategy::d_iInstanceId++;
    d_pOperator = parameters->d_pOperator;
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
    AMP_INSIST( db, "InputDatabase object must be non-NULL" );
    d_iMaxIterations       = db->getWithDefault<int>( "max_iterations", 100 );
    d_iDebugPrintInfoLevel = db->getWithDefault<int>( "print_info_level", 0 );
    d_bUseZeroInitialGuess = db->getWithDefault<bool>( "zero_initial_guess", true );
    d_dAbsoluteTolerance   = db->getWithDefault<double>( "absolute_tolerance", 1.0e-14 );
    d_dRelativeTolerance   = db->getWithDefault<double>( "relative_tolerance", 1.0e-09 );
    d_bComputeResidual     = db->getWithDefault<bool>( "compute_residual", false );
}

void SolverStrategy::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_INSIST( parameters, "SolverStrategyParameters object cannot be NULL" );
}


/****************************************************************
 * Reset                                                         *
 ****************************************************************/
void SolverStrategy::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator ) {
        d_pOperator->reset( params );
    }
}

void SolverStrategy::reset( std::shared_ptr<SolverStrategyParameters> ) {}

void SolverStrategy::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> ) {}

int SolverStrategy::getTotalNumberOfIterations( void )
{
    return std::accumulate( d_iterationHistory.begin(), d_iterationHistory.end(), 0 );
}

bool SolverStrategy::checkStoppingCriteria( AMP::Scalar res_norm )
{
    // default to diverged other, which is held during the solve
    d_ConvergenceStatus   = SolverStatus::DivergedOther;
    d_dResidualNorm       = res_norm;
    const auto res_norm_d = static_cast<double>( res_norm );
    AMP_ASSERT( res_norm == res_norm_d );

    // check stopping criteria and ensure more restrictive categories
    // are tested first (e.g. don't set diverged max iters if solver
    // hit rel tol on final step)
    if ( d_dResidualNorm < d_dAbsoluteTolerance ) {
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        return true;
    } else if ( d_dResidualNorm < d_dRelativeTolerance * d_dInitialResidual ) {
        d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
        return true;
    } else if ( d_iNumberIterations == d_iMaxIterations ) {
        d_ConvergenceStatus = SolverStatus::DivergedMaxIterations;
        return true;
    } else if ( std::isnan( res_norm_d ) ) {
        d_ConvergenceStatus = SolverStatus::DivergedOnNan;
        return true;
    }

    return false;
}

/****************************************************************
 * residual                                                         *
 ****************************************************************/
void SolverStrategy::residual( std::shared_ptr<const AMP::LinearAlgebra::Vector>,
                               std::shared_ptr<const AMP::LinearAlgebra::Vector>,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> )
{
    AMP_ERROR( "Not implemented" );
}


} // namespace AMP::Solver
