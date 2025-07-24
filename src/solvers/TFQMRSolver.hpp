#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/TFQMRSolver.h"
#include "ProfilerApp.h"


#include <array>
#include <cmath>
#include <iomanip>
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

    registerOperator( d_pOperator );

    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
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
                d_pNestedSolver = AMP::Solver::SolverFactory::create( innerParameters );
                AMP_ASSERT( d_pNestedSolver );
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

template<typename T>
void TFQMRSolver<T>::allocateScratchVectors( std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{
    AMP_INSIST( u, "Input to BiCGSTABSolver::allocateScratchVectors must be non-null" );
    d_z     = u->clone();
    d_delta = u->clone();
    d_w     = u->clone();
    d_d     = u->clone();
    d_v     = u->clone();

    d_v->setNoGhosts();

    for ( size_t i = 0; i < 2; ++i ) {
        d_u[i] = u->clone();
        d_y[i] = u->clone();
        // ensure d_u[i], d_y[i] do no communication
        d_u[i]->setNoGhosts();
        d_y[i]->setNoGhosts();
    }
}

template<typename T>
void TFQMRSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    // not sure about excluding op == d_pOperator
    d_pOperator = op;

    if ( d_pOperator ) {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        if ( linearOp ) {
            d_r = linearOp->getRightVector();
            allocateScratchVectors( d_r );
        }
    }
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void TFQMRSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                            std::shared_ptr<AMP::LinearAlgebra::Vector> x )
{
    PROFILE( "TFQMRSolver<T>::apply" );

    if ( !d_r ) {
        d_r = x->clone();
        allocateScratchVectors( d_r );
    }

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 1;

    // Check input vector states
    AMP_ASSERT( ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        d_r->copyVector( f );
        x->zero();
    } else {
        d_pOperator->residual( f, x, d_r );
    }

    // compute the current residual norm
    d_dResidualNorm = d_r->L2Norm();
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual =
        d_dResidualNorm > std::numeric_limits<T>::epsilon() ? d_dResidualNorm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TFQMR: initial solution L2-norm: " << x->L2Norm() << std::endl;
        AMP::pout << "TFQMR: initial rhs L2-norm: " << f->L2Norm() << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "TFQMR: initial residual" << std::setw( 32 ) << d_dResidualNorm << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( d_dResidualNorm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "TFQMR: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in TFQMR
    T theta  = static_cast<T>( 0.0 );
    T eta    = static_cast<T>( 0.0 );
    T tau    = static_cast<T>( d_dResidualNorm );
    auto rho = tau * tau;

    d_u[0]->zero();
    d_u[1]->zero();

    d_y[0]->zero();
    d_y[1]->zero();

    d_z->zero();
    d_delta->zero();
    d_d->zero();

    d_w->copyVector( d_r );

    d_y[0]->copyVector( d_r );

    if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
        d_pNestedSolver->apply( d_y[0], d_z );
    } else {
        d_z->copyVector( d_y[0] );
    }

    d_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_pOperator->apply( d_z, d_v );

    d_u[0]->copyVector( d_v );

    // must start at iter == 1 to make m in the inner loop make sense
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        auto sigma = static_cast<T>( d_r->dot( *d_v ) );

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
                d_y[1]->axpy( -alpha, *d_v, *d_y[0] );
                if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
                    d_pNestedSolver->apply( d_y[1], d_z );
                } else {
                    d_z->copyVector( d_y[1] );
                }

                d_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                d_pOperator->apply( d_z, d_u[1] );
            }

            const int m = 2 * d_iNumberIterations - 2 + j;
            d_w->axpy( -alpha, *d_u[j], *d_w );
            d_d->axpy( ( theta * theta * eta / alpha ), *d_d, *d_y[j] );

            theta = static_cast<T>( d_w->L2Norm() ) / tau;
            const auto c =
                static_cast<T>( 1.0 ) / std::sqrt( static_cast<T>( 1.0 ) + theta * theta );
            tau = tau * theta * c;
            eta = c * c * alpha;

            // update the increment to the solution
            d_delta->axpy( eta, *d_d, *d_delta );
            if ( d_iDebugPrintInfoLevel > 2 ) {
                AMP::pout << "TFQMR: outer/inner iteration " << std::setw( 6 )
                          << d_iNumberIterations << "/" << j << ", solution update norm "
                          << d_delta->L2Norm() << std::endl;
            }

            // Use upper bound on residual norm to test convergence cheaply
            res_bound = tau * std::sqrt( static_cast<T>( m + 1.0 ) );

            if ( checkStoppingCriteria( res_bound ) ) {
                if ( d_iDebugPrintInfoLevel > 0 ) {
                    AMP::pout << "TFQMR: outer/inner iteration " << std::setw( 6 )
                              << d_iNumberIterations << "/" << j << ", residual " << res_bound
                              << std::endl;
                }
                d_dResidualNorm = res_bound; // this is likely an over-estimate
                converged       = true;
                break;
            }
        }

        if ( converged ) {
            break;
        }

        // replace by soft-equal
        if ( rho == static_cast<T>( 0.0 ) ) {
            // set diverged reason
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "TFQMRSolver<T>::apply: Breakdown, rho == 0" );
            break;
        }

        auto rho_n = static_cast<T>( d_r->dot( *d_w ) );
        auto beta  = rho_n / rho;
        rho        = rho_n;

        d_y[0]->axpy( beta, *d_y[1], *d_w );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            d_pNestedSolver->apply( d_y[0], d_z );
        } else {
            d_z->copyVector( d_y[0] );
        }

        d_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( d_z, d_u[0] );

        // replace by axpbycz when ready
        d_v->axpy( beta, *d_v, *d_u[1] );
        d_v->axpy( beta, *d_v, *d_u[0] );

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "TFQMR: outer iteration " << std::setw( 14 ) << d_iNumberIterations
                      << ", residual " << res_bound << std::endl;
        }
    }

    // unwind the preconditioner if necessary
    if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
        d_pNestedSolver->apply( d_delta, d_z );
    } else {
        d_z->copyVector( d_delta );
    }

    x->add( *d_z, *x );

    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // should this always be true since TFQMR only gives a bound on the
    // residual?
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, x, d_r );
        d_dResidualNorm = d_r->L2Norm();
        // final check updates flags if needed
        checkStoppingCriteria( d_dResidualNorm );
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "TFQMR: final residual: " << d_dResidualNorm
                  << " iterations: " << d_iNumberIterations << " convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "TFQMRSolver<T>::apply: final solution L2Norm: " << x->L2Norm() << std::endl;
    }
}
} // namespace AMP::Solver
