#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/CGSolver.h"
#include "AMP/solvers/KrylovSolverParameters.h"
#include "ProfilerApp.h"

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/

template<typename T>
CGSolver<T>::CGSolver( std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters )
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
void CGSolver<T>::initialize( std::shared_ptr<const AMP::Solver::SolverStrategyParameters> params )
{
    auto parameters =
        std::dynamic_pointer_cast<const AMP::Solver::KrylovSolverParameters>( params );
    AMP_ASSERT( parameters );

    d_pPreconditioner = parameters->d_pPreconditioner;

    getFromInput( parameters->d_db );

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

// Function to get values from input
template<typename T>
void CGSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_dDivergenceTolerance = db->getWithDefault<T>( "divergence_tolerance", 1.0e+03 );
    d_bUsesPreconditioner  = db->getWithDefault<bool>( "uses_preconditioner", false );
}

/****************************************************************
 *  Solve                                                        *
 * TODO: store convergence history, iterations, convergence reason
 ****************************************************************/
template<typename T>
void CGSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE_START( "solve" );

    // Check input vector states
    AMP_ASSERT(
        ( f->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::UNCHANGED ) ||
        ( f->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT(
        ( u->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::UNCHANGED ) ||
        ( u->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::LOCAL_CHANGED ) );

    // compute the norm of the rhs in order to compute
    // the termination criterion
    const auto f_norm = static_cast<T>( f->L2Norm() );

    // enhance with convergence reason, number of iterations etc
    if ( f_norm == 0.0 )
        return;

    const T terminate_tol = d_dRelativeTolerance * f_norm;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        std::cout << "CGSolver<T>::solve: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        std::cout << "CGSolver<T>::solve: initial L2Norm of rhs vector: " << f_norm << std::endl;
    }

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }

    // z will store r when a preconditioner is not present
    // and will store the result of a preconditioner solve
    // when a preconditioner is present
    std::shared_ptr<AMP::LinearAlgebra::Vector> z;

    // residual vector
    AMP::LinearAlgebra::Vector::shared_ptr r = f->cloneVector();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        r->copyVector( f );
    } else {
        d_pOperator->residual( f, u, r );
    }
    // compute the current residual norm
    auto current_res = static_cast<T>( r->L2Norm() );

    // return if the residual is already low enough
    if ( current_res < terminate_tol ) {
        // provide a convergence reason
        // provide history (iterations, conv history etc)
        return;
    }

    z = u->cloneVector();

    // apply the preconditioner if it exists
    if ( d_bUsesPreconditioner ) {
        d_pPreconditioner->apply( r, z );
    } else {
        z->copyVector( r );
    }

    std::vector<T> rho( 2, 0.0 );
    rho[1] = static_cast<T>( z->dot( *r ) );
    rho[0] = rho[1];

    auto p = z->cloneVector();
    auto w = r->cloneVector();
    p->copyVector( z );

    for ( auto iter = 0; iter < d_iMaxIterations; ++iter ) {

        T beta = 1.0;

        // w = Ap
        d_pOperator->apply( p, w );

        // alpha = p'Ap
        T alpha = static_cast<T>( w->dot( *p ) );

        // sanity check, the curvature should be positive
        if ( alpha <= 0.0 ) {
            // set diverged reason
            AMP_ERROR( "Negative curvature encountered in CG!!" );
        }

        alpha = rho[1] / alpha;

        u->axpy( alpha, *p, *u );
        r->axpy( -alpha, *w, *r );

        // compute the current residual norm
        current_res = static_cast<T>( r->L2Norm() );
        if ( d_iDebugPrintInfoLevel > 0 ) {
            std::cout << "CG: iteration " << ( iter + 1 ) << ", residual " << current_res
                      << std::endl;
        }
        // check if converged
        if ( current_res < terminate_tol ) {
            // set a convergence reason
            break;
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pPreconditioner->apply( r, z );
        } else {
            z->copyVector( r );
        }

        rho[0] = rho[1];
        rho[1] = static_cast<T>( r->dot( *z ) );

        beta = rho[1] / rho[0];
        p->axpy( beta, *p, *z );
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        std::cout << "L2Norm of solution: " << u->L2Norm() << std::endl;
    }

    PROFILE_STOP( "solve" );
}

/****************************************************************
 *  Function to set the register the operator                    *
 ****************************************************************/
template<typename T>
void CGSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    AMP_ASSERT( op );
    d_pOperator = op;
}

template<typename T>
void CGSolver<T>::resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
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