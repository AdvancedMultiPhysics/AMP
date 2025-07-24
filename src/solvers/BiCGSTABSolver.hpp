#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/BiCGSTABSolver.h"
#include "AMP/solvers/SolverFactory.h"

#include "ProfilerApp.h"

#include <cmath>
#include <iomanip>
#include <limits>

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
template<typename T>
BiCGSTABSolver<T>::BiCGSTABSolver() : d_restarts( 0 )
{
}

template<typename T>
BiCGSTABSolver<T>::BiCGSTABSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters ), d_restarts( 0 )
{
    AMP_ASSERT( parameters );
    initialize( parameters );
}


/****************************************************************
 *  Destructor                                                   *
 ****************************************************************/
template<typename T>
BiCGSTABSolver<T>::~BiCGSTABSolver() = default;

/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
template<typename T>
void BiCGSTABSolver<T>::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
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
void BiCGSTABSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
}

template<typename T>
void BiCGSTABSolver<T>::allocateScratchVectors(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{
    // allocates d_p, d_w, d_z (if necessary)
    AMP_INSIST( u, "Input to BiCGSTABSolver::allocateScratchVectors must be non-null" );
    d_r_tilde = u->clone();
    d_p       = u->clone();
    d_s       = u->clone();
    d_t       = u->clone();
    d_v       = u->clone();

    // ensure t, v do no communication
    d_t->setNoGhosts();
    d_v->setNoGhosts();

    if ( d_bUsesPreconditioner ) {
        d_p_hat = u->clone();
        d_s_hat = u->clone();
    }
}

template<typename T>
void BiCGSTABSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
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
void BiCGSTABSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    // NOTE:: Things that need to be rechecked
    // 1. Should p = res initially
    // 2. Should res be the preconditioned residual
    // 3. Should the algorithm use the residual and a zero initial solution
    //    and add the solution back at the end. Literature suggests so
    // 4. Will 3, be affected by the transition to using checkStoppingCriteria
    // 5. This implementation is both BiCGSTAB & Flexible BiCGSTAB with right preconditioning
    //    See J. Vogels paper
    PROFILE( "BiCGSTABSolver<T>::apply" );

    if ( !d_r ) {
        d_r = u->clone();
        allocateScratchVectors( d_r );
    }

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 1;

    // Check input vector states
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        d_r->copyVector( f );
        u->zero();
    } else {
        d_pOperator->residual( f, u, d_r );
    }

    // compute the current residual norm
    d_dResidualNorm   = d_r->L2Norm();
    auto r_tilde_norm = static_cast<T>( d_dResidualNorm );
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual =
        d_dResidualNorm > std::numeric_limits<T>::epsilon() ? d_dResidualNorm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BiCGSTAB: initial solution L2Norm: " << u->L2Norm() << std::endl;
        AMP::pout << "BiCGSTAB: initial rhs L2Norm: " << f->L2Norm() << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BiCGSTAB: initial residual " << std::setw( 19 ) << d_dResidualNorm
                  << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( d_dResidualNorm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "BiCGSTAB: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in BiCGSTAB
    [[maybe_unused]] T alpha = static_cast<T>( 1.0 );
    [[maybe_unused]] T beta  = static_cast<T>( 0.0 );
    [[maybe_unused]] T omega = static_cast<T>( 1.0 );
    [[maybe_unused]] std::vector<T> rho( 2, static_cast<T>( 1.0 ) );

    // r_tilde is a non-zero initial direction chosen to be r
    // traditional choice is the initial residual
    d_r_tilde->copyVector( d_r );

    d_p->zero();
    d_v->zero();

    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        rho[1] = static_cast<T>( d_r_tilde->dot( *d_r ) );

        auto angle = std::sqrt( std::fabs( rho[1] ) );
        auto eps   = std::numeric_limits<T>::epsilon();

        if ( angle < eps * r_tilde_norm ) {
            // the method breaks down as the vectors are orthogonal to r0
            // attempt to restart with a new r0
            u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            d_pOperator->residual( f, u, d_r );
            d_r_tilde->copyVector( d_r );
            d_p->copyVector( d_r );
            d_dResidualNorm = d_r->L2Norm();
            rho[1] = r_tilde_norm = static_cast<T>( d_dResidualNorm );
            d_restarts++;
            continue;
        }

        if ( d_iNumberIterations == 1 ) {
            // NOTE: there are differences in the literature in what the initial p is
            // Van der Vorst, Eigen, Petsc : p = 0
            // J. Vogel, J. Chen et. al on FBiCGSTAB: p = res
            d_p->copyVector( d_r );
        } else {

            beta = ( rho[1] / rho[0] ) * ( alpha / omega );
            d_p->axpy( -omega, *d_v, *d_p );
            d_p->axpy( beta, *d_p, *d_r );
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pNestedSolver->apply( d_p, d_p_hat );
        } else {
            d_p_hat = d_p;
        }

        d_p_hat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( d_p_hat, d_v );

        alpha = static_cast<T>( d_r_tilde->dot( *d_v ) );
        AMP_ASSERT( alpha != static_cast<T>( 0.0 ) );
        alpha = rho[1] / alpha;

        d_s->axpy( -alpha, *d_v, *d_r );

        const auto s_norm = d_s->L2Norm();

        // Check for early convergence
        // s is residual wrt. h, if good replace u with h
        if ( checkStoppingCriteria( s_norm, false ) ) {
            u->axpy( alpha, *d_p_hat, *u );
            d_dResidualNorm = static_cast<T>( s_norm );
            break;
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pNestedSolver->apply( d_s, d_s_hat );
        } else {
            d_s_hat = d_s;
        }


        d_s_hat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( d_s_hat, d_t );

        // the L2NormAndDot is not optimized in vectors and needs to be
        // if BiCGSTAB becomes a priority
        auto [norm_sq, dot] = d_t->L2NormAndDot( *d_s );
        auto t_sqnorm       = static_cast<T>( norm_sq );
        auto t_dot_s        = static_cast<T>( dot );

        // note the choice of omega below corresponds to what van der Vorst calls BiCGSTAB-P
        omega = ( t_sqnorm == static_cast<T>( 0.0 ) ) ? static_cast<T>( 0.0 ) : t_dot_s / t_sqnorm;

        // this should be replaced by vec->axpbycz  when it is ready
        u->axpy( alpha, *d_p_hat, *u );
        u->axpy( omega, *d_s_hat, *u );

        d_r->axpy( -omega, *d_t, *d_s );

        // compute the current residual norm
        d_dResidualNorm = static_cast<T>( d_r->L2Norm() );

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "BiCGSTAB: iteration " << std::setw( 8 ) << d_iNumberIterations
                      << ", residual " << d_dResidualNorm << std::endl;
        }

        // break if the residual is low enough
        if ( checkStoppingCriteria( d_dResidualNorm ) ) {
            break;
        }

        if ( omega == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "BiCGSTAB: breakdown encountered, omega == 0" );
            break;
        }

        rho[0] = rho[1];
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, d_r );
        d_dResidualNorm = static_cast<T>( d_r->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( d_dResidualNorm );
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BiCGSTAB: final residual: " << d_dResidualNorm
                  << ", iterations: " << d_iNumberIterations << ", convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BiCGSTAB: final solution L2Norm: " << u->L2Norm() << std::endl;
    }
}
} // namespace AMP::Solver
