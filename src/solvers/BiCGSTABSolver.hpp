#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/BiCGSTABSolver.h"
#include "AMP/solvers/SolverFactory.h"

#include "ProfilerApp.h"

#include <cmath>
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
void BiCGSTABSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
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

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // residual vector
    AMP::LinearAlgebra::Vector::shared_ptr res = f->clone();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        res->copyVector( f );
    } else {
        d_pOperator->residual( f, u, res );
    }

    // compute the current residual norm
    auto res_norm     = static_cast<T>( res->L2Norm() );
    auto r_tilde_norm = res_norm;
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual = res_norm > std::numeric_limits<T>::epsilon() ? res_norm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BiCGSTABSolver<T>::solve: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "BiCGSTABSolver<T>::solve: initial L2Norm of rhs vector: " << f->L2Norm()
                  << std::endl;
        AMP::pout << "BiCGSTABSolver<T>::solve: initial L2Norm of residual " << res_norm
                  << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( res_norm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "BiCGSTABSolver<T>::solve: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in BiCGSTAB
    [[maybe_unused]] T alpha = static_cast<T>( 1.0 );
    [[maybe_unused]] T beta  = static_cast<T>( 0.0 );
    [[maybe_unused]] T omega = static_cast<T>( 1.0 );
    [[maybe_unused]] std::vector<T> rho( 2, static_cast<T>( 1.0 ) );

    // r_tilde is a non-zero initial direction chosen to be r
    std::shared_ptr<AMP::LinearAlgebra::Vector> r_tilde;
    // traditional choice is the initial residual
    r_tilde = res->clone();
    r_tilde->copyVector( res );

    auto p = res->clone();
    auto v = res->clone();
    p->zero();
    v->zero();

    std::shared_ptr<AMP::LinearAlgebra::Vector> p_hat, s, s_hat, t;

    for ( d_iNumberIterations = 0; d_iNumberIterations < d_iMaxIterations; ++d_iNumberIterations ) {

        rho[1] = static_cast<T>( r_tilde->dot( *res ) );

        auto angle = std::sqrt( std::fabs( rho[1] ) );
        auto eps   = std::numeric_limits<T>::epsilon();

        if ( angle < eps * r_tilde_norm ) {
            // the method breaks down as the vectors are orthogonal to r0
            // attempt to restart with a new r0
            u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            d_pOperator->residual( f, u, res );
            r_tilde->copyVector( res );
            p->copyVector( res );
            res_norm = static_cast<T>( res->L2Norm() );
            rho[1] = r_tilde_norm = res_norm;
            d_restarts++;
            continue;
        }

        if ( d_iNumberIterations == 0 ) {
            // NOTE: there are differences in the literature in what the initial p is
            // Van der Vorst, Eigen, Petsc : p = 0
            // J. Vogel, J. Chen et. al on FBiCGSTAB: p = res
            p->copyVector( res );
        } else {

            beta = ( rho[1] / rho[0] ) * ( alpha / omega );
            p->axpy( -omega, *v, *p );
            p->axpy( beta, *p, *res );
        }

        if ( !p_hat ) {
            p_hat = u->clone();
            p_hat->zero();
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pPreconditioner->apply( p, p_hat );
        } else {
            p_hat->copyVector( p );
        }

        p_hat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( p_hat, v );

        alpha = static_cast<T>( r_tilde->dot( *v ) );
        AMP_ASSERT( alpha != static_cast<T>( 0.0 ) );
        alpha = rho[1] / alpha;

        if ( !s ) {
            s = res->clone();
        }
        s->axpy( -alpha, *v, *res );

        const auto s_norm = s->L2Norm();

        // Check for early convergence
        // s is residual wrt. h, if good replace u with h
        if ( checkStoppingCriteria( s_norm, false ) ) {
            u->axpy( alpha, *p_hat, *u );
            res_norm = static_cast<T>( s_norm );
            break;
        }

        if ( !s_hat ) {
            s_hat = u->clone();
            s_hat->zero();
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pPreconditioner->apply( s, s_hat );
        } else {
            s_hat->copyVector( s );
        }


        if ( !t ) {
            t = res->clone();
        }

        s_hat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( s_hat, t );

        auto t_sqnorm = static_cast<T>( t->dot( *t ) );
        auto t_dot_s  = static_cast<T>( t->dot( *s ) );
        // note the choice of omega below corresponds to what van der Vorst calls BiCGSTAB-P
        omega = ( t_sqnorm == static_cast<T>( 0.0 ) ) ? static_cast<T>( 0.0 ) : t_dot_s / t_sqnorm;

        u->axpy( alpha, *p_hat, *u );
        u->axpy( omega, *s_hat, *u );

        res->axpy( -omega, *t, *s );

        // compute the current residual norm
        res_norm = static_cast<T>( res->L2Norm() );

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BiCGSTAB: iteration " << ( d_iNumberIterations + 1 ) << ", residual "
                      << res_norm << std::endl;
        }

        // break if the residual is low enough
        if ( checkStoppingCriteria( res_norm ) ) {
            break;
        }

        if ( omega == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "BiCGSTABSolver<T>::solve: breakdown encountered, omega == 0" );
            break;
        }

        rho[0] = rho[1];
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, res );
        res_norm = static_cast<T>( res->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( res_norm );
    }

    // Store final residual should it be queried elsewhere
    d_dResidualNorm = res_norm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BiCGSTABSolver<T>::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "BiCGSTABSolver<T>::apply: final L2Norm of residual: " << res_norm
                  << std::endl;
        AMP::pout << "BiCGSTABSolver<T>::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "BiCGSTABSolver<T>::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

template<typename T>
void BiCGSTABSolver<T>::resetOperator(
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
