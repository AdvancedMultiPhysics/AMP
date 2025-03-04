#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/TFQMRSolver.h"
#include "ProfilerApp.h"


#include <array>
#include <cmath>
#include <limits>

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/

template<typename T>
TFQMRSolver<T>::TFQMRSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );

    // Initialize
    initialize( parameters );
}

/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
template<typename T>
void TFQMRSolver<T>::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );
    auto db = parameters->d_db;
    getFromInput( db );

    if ( parameters->d_pNestedSolver ) {
        d_pPreconditioner = parameters->d_pNestedSolver;
    } else {
        if ( d_bUsesPreconditioner ) {
            auto pcName  = db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
            auto outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;
            if ( outerDB ) {
                auto pcDB = outerDB->getDatabase( pcName );
                auto innerParameters =
                    std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
                innerParameters->d_global_db = parameters->d_global_db;
                innerParameters->d_pOperator = d_pOperator;
                d_pPreconditioner = AMP::Solver::SolverFactory::create( innerParameters );
                AMP_ASSERT( d_pPreconditioner );
            }
        }
    }
}

// Function to get values from input
template<typename T>
void TFQMRSolver<T>::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );

    // default is right preconditioning, options are right, left, both
    if ( d_bUsesPreconditioner ) {
        d_preconditioner_side = db->getWithDefault<std::string>( "preconditioner_side", "right" );
    }

    // TFQMR only has bounds on residual norm naturally
    // Default to manually computing residual at the end
    d_bComputeResidual = db->getWithDefault<bool>( "compute_residual", true );
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void TFQMRSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                            std::shared_ptr<AMP::LinearAlgebra::Vector> x )
{
    PROFILE( "TFQMRSolver<T>::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT( ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // residual vector
    auto res = f->clone();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        res->copyVector( f );
    } else {
        d_pOperator->residual( f, x, res );
    }

    // compute the current residual norm
    auto res_norm = static_cast<T>( res->L2Norm() );
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual = res_norm > std::numeric_limits<T>::epsilon() ? res_norm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TFQMRSolver<T>::apply: initial L2Norm of solution vector: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "TFQMRSolver<T>::apply: initial L2Norm of rhs vector: " << f->L2Norm()
                  << std::endl;
        AMP::pout << "TFQMRSolver<T>::apply: initial L2Norm of residual: " << res_norm << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( res_norm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "TFQMRSolver<T>::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in TFQMR
    T theta  = static_cast<T>( 0.0 );
    T eta    = static_cast<T>( 0.0 );
    T tau    = res_norm;
    auto rho = tau * tau;

    std::array<AMP::LinearAlgebra::Vector::shared_ptr, 2> u;
    u[0] = f->clone();
    u[1] = f->clone();
    u[0]->zero();
    u[1]->zero();

    std::array<AMP::LinearAlgebra::Vector::shared_ptr, 2> y;
    y[0] = f->clone();
    y[1] = f->clone();
    y[0]->zero();
    y[1]->zero();

    // z is allocated only if the preconditioner is used
    AMP::LinearAlgebra::Vector::shared_ptr z;
    if ( d_bUsesPreconditioner ) {
        z = f->clone();
        z->zero();
    }

    auto delta = f->clone();
    delta->zero();

    auto w = res->clone();
    w->copyVector( res );

    y[0]->copyVector( res );

    auto d = res->clone();
    d->zero();

    auto v = res->clone();

    if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
        d_pPreconditioner->apply( y[0], z );
    } else {
        z = y[0];
    }

    z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_pOperator->apply( z, v );

    u[0]->copyVector( v );

    // must start at iter == 1 to make m in the inner loop make sense
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        auto sigma = static_cast<T>( res->dot( *v ) );

        // replace by soft-equal
        if ( sigma == static_cast<T>( 0.0 ) ) {
            // set diverged reason
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "TFQMRSolver<T>::apply: Breakdown, sigma == 0" );
            break;
        }

        auto alpha = rho / sigma;

        bool converged = false;
        T res_bound{ 0.0 };
        for ( int j = 0; j <= 1; ++j ) {
            if ( j == 1 ) {
                y[1]->axpy( -alpha, *v, *y[0] );
                if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
                    d_pPreconditioner->apply( y[1], z );
                } else {
                    z = y[1];
                }

                z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                d_pOperator->apply( z, u[1] );
            }

            const int m = 2 * d_iNumberIterations - 1 + j;
            w->axpy( -alpha, *u[j], *w );
            d->axpy( ( theta * theta * eta / alpha ), *d, *y[j] );

            theta = static_cast<T>( w->L2Norm() ) / tau;
            const auto c =
                static_cast<T>( 1.0 ) / std::sqrt( static_cast<T>( 1.0 ) + theta * theta );
            tau = tau * theta * c;
            eta = c * c * alpha;

            // update the increment to the solution
            delta->axpy( eta, *d, *delta );

            // Use upper bound on residual norm to test convergence cheaply
            res_bound = tau * std::sqrt( static_cast<T>( m + 1.0 ) );

            if ( checkStoppingCriteria( res_bound ) ) {
                if ( d_iDebugPrintInfoLevel > 1 ) {
                    AMP::pout << "TFQMR: outer/inner iteration " << ( d_iNumberIterations ) << "/"
                              << j << ", residual " << res_bound << std::endl;
                }
                res_norm  = res_bound; // this is likely an over-estimate
                converged = true;
                break;
            }
        }

        if ( converged ) {
            // unwind the preconditioner if necessary
            if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
                d_pPreconditioner->apply( delta, z );
            } else {
                z = delta;
            }
            x->axpy( 1.0, *z, *x );
            break;
        }

        // replace by soft-equal
        if ( rho == static_cast<T>( 0.0 ) ) {
            // set diverged reason
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "TFQMRSolver<T>::apply: Breakdown, rho == 0" );
            break;
        }

        auto rho_n = static_cast<T>( res->dot( *w ) );
        auto beta  = rho_n / rho;
        rho        = rho_n;

        y[0]->axpy( beta, *y[1], *w );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            d_pPreconditioner->apply( y[0], z );
        } else {
            z = y[0];
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( z, u[0] );

        v->axpy( beta, *v, *u[1] );
        v->axpy( beta, *v, *u[0] );

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "TFQMR: outer iteration " << ( d_iNumberIterations ) << ", residual "
                      << res_bound << std::endl;
        }
    }

    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // should this always be true since TFQMR only gives a bound on the
    // residual?
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, x, res );
        res_norm = static_cast<T>( res->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( res_norm );
    }

    // Store final residual should it be queried elsewhere
    d_dResidualNorm = res_norm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "TFQMRSolver<T>::apply: final L2Norm of solution: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "TFQMRSolver<T>::apply: final L2Norm of residual: " << res_norm << std::endl;
        AMP::pout << "TFQMRSolver<T>::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "TFQMRSolver<T>::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

template<typename T>
void TFQMRSolver<T>::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator ) {
        d_pOperator->reset( params );
    }

    // should add a mechanism for the linear operator to provide updated parameters for the
    // preconditioner operator
    // though it's unclear where this might be necessary
    if ( d_pPreconditioner ) {
        d_pPreconditioner->resetOperator( params );
    }
}
} // namespace AMP::Solver
