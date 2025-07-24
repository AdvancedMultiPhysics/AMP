#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/CGSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "ProfilerApp.h"

#include <iomanip>

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
void CGSolver<T>::initialize(
    std::shared_ptr<const AMP::Solver::SolverStrategyParameters> parameters )
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

    if ( d_sVariant != "pcg" ) {
        d_vDirs.resize( d_max_dimension );
        d_gamma.resize( d_max_dimension );
    }
}

// Function to get values from input
template<typename T>
void CGSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_dDivergenceTolerance = db->getWithDefault<T>( "divergence_tolerance", 1.0e+03 );
    d_bUsesPreconditioner  = db->getWithDefault<bool>( "uses_preconditioner", false );
    d_sVariant             = db->getWithDefault<std::string>( "variant", "pcg" );

    if ( d_sVariant == "fcg" ) {
        d_max_dimension = db->getWithDefault<int>( "max_dimension", 10 );
    } else if ( d_sVariant == "ipcg" ) {
        d_max_dimension = 1;
    }
}

template<typename T>
void CGSolver<T>::allocateScratchVectors( std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{
    // allocates d_p, d_w, d_z (if necessary)
    AMP_INSIST( u, "Input to CGSolver::allocateScratchVectors must be non-null" );
    d_p = u->clone();
    d_w = u->clone();

    // ensure w does no communication
    d_w->setNoGhosts();

    if ( d_bUsesPreconditioner )
        d_z = u->clone();
}

template<typename T>
void CGSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    // not sure about excluding op == d_pOperator
    d_pOperator = op;

    if ( d_pOperator ) {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        if ( linearOp ) {
            d_r = linearOp->getRightVector();
            if ( !d_bUsesPreconditioner )
                d_z = d_r;
            allocateScratchVectors( d_r );
        }
    }
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void CGSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "CGSolver<T>::apply" );

    if ( !d_r ) {
        d_r = u->clone();
        allocateScratchVectors( d_r );
    }

    // ensure d_r does no communication
    if ( d_sVariant == "pcg" )
        d_r->setNoGhosts();

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        PROFILE( "CGSolver<T>:: u=0, r = f (initial)" );
        u->zero();
        d_r->copyVector( f );
    } else {
        PROFILE( "CGSolver<T>:: r = f-Au (initial)" );
        d_pOperator->residual( f, u, d_r );
    }

    // Store initial residual
    {
        PROFILE( "CGSolver<T>:: r->L2Norm (initial)" );
        d_dResidualNorm = static_cast<T>( d_r->L2Norm() );
    }

    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual =
        d_dResidualNorm > std::numeric_limits<T>::epsilon() ? d_dResidualNorm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "CG: initial L2Norm of solution vector: " << u->L2Norm() << std::endl;
        AMP::pout << "CG: initial L2Norm of rhs vector: " << f->L2Norm() << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "CG: initial residual " << std::setw( 19 ) << d_dResidualNorm << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( d_dResidualNorm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "CG: initial residual below tolerance" << std::endl;
        }
        return;
    }


    // apply the preconditioner if it exists
    if ( d_bUsesPreconditioner ) {
        PROFILE( "CGSolver<T>:: z = M^{-1}r (initial)" );
        d_pNestedSolver->apply( d_r, d_z );
    } else {
        d_z = d_r;
    }

    if ( d_sVariant != "pcg" ) {
        d_vDirs[0] = u->clone();
    }

    if ( d_sVariant == "fcg" ) {
        d_vDirs[0]->copyVector( d_z );
    }

    if ( d_sVariant == "ipcg" ) {
        d_vDirs[0]->copyVector( d_r );
    }

    T rho_0, rho_1, alpha, gamma;

    if ( d_bUsesPreconditioner ) {
        PROFILE( "CGSolver<T>:: rho_1 = <z,r> (initial)" );
        rho_1 = static_cast<T>( d_z->dot( *d_r ) );
    } else {
        rho_1 = static_cast<T>( d_dResidualNorm );
        rho_1 *= rho_1;
    }

    {
        PROFILE( "CGSolver<T>:: p = z (initial) " );
        d_p->copyVector( d_z );
    }
    auto k = -1;

    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        ++k;
        {
            PROFILE( "CGSolver<T>:: p->makeConsistent" );
            d_p->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        }
        {
            PROFILE( "CGSolver<T>:: w = Ap " );
            // w = Ap
            d_pOperator->apply( d_p, d_w );
        }
        {
            PROFILE( "CGSolver<T>:: gamma = <w,p>" );
            // gamma = p'Ap
            gamma = static_cast<T>( d_w->dot( *d_p ) );
        }
        if ( d_sVariant == "fcg" )
            d_gamma[k % d_max_dimension] = gamma;

        // sanity check, the curvature should be positive
        if ( gamma == 0.0 ) {
            // the checkStoppingCriteria looks wrong!! check
            // at solution
            checkStoppingCriteria( d_dResidualNorm );
            break;
        } else if ( gamma < 0.0 ) {
            // set diverged reason
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "CG: negative curvature encountered" );
            break;
        }

        alpha = rho_1 / gamma;

        u->axpy( alpha, *d_p, *u );
        d_r->axpy( -alpha, *d_w, *d_r );

        {
            PROFILE( "CGSolver<T>:: r->L2Norm" );
            // compute the current residual norm
            d_dResidualNorm = static_cast<T>( d_r->L2Norm() );
        }

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "CG: iteration " << std::setw( 8 ) << d_iNumberIterations << ", residual "
                      << d_dResidualNorm << std::endl;
        }

        // check if converged
        if ( checkStoppingCriteria( d_dResidualNorm ) ) {
            break;
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            PROFILE( "CGSolver<T>:: z = M^{-1}r" );
            d_pNestedSolver->apply( d_r, d_z );
        }

        rho_0 = rho_1;

        if ( d_sVariant == "ipcg" ) {

            d_vDirs[0]->axpy( static_cast<T>( -1.0 ), *d_vDirs[0], *d_r );
            rho_1 = static_cast<T>( d_vDirs[0]->dot( *d_z ) );
            d_vDirs[0]->copyVector( d_r );

        } else if ( d_sVariant == "fcg" ) {

            d_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            d_pOperator->apply( d_z, d_w );
            std::vector<T> vbeta( d_max_dimension );

            // This should be combined into a single operation across multiple vectors
            for ( auto j = std::max( 0, k - d_max_dimension + 1 ); j <= k; ++j ) {
                auto idx   = j % d_max_dimension;
                auto dp    = static_cast<T>( d_vDirs[idx]->dot( *d_w ) );
                vbeta[idx] = dp / d_gamma[idx];
            }

            auto d = ( k + 1 ) % d_max_dimension;

            if ( !d_vDirs[d] )
                d_vDirs[d] = u->clone();

            d_vDirs[d]->copyVector( d_z );

            for ( auto j = std::max( 0, k - d_max_dimension + 1 ); j <= k; ++j ) {
                auto idx = j % d_max_dimension;
                d_vDirs[d]->axpy( -vbeta[idx], *d_vDirs[idx], *d_vDirs[d] );
            }

            d_p->copyVector( d_vDirs[d] );
            rho_1 = static_cast<T>( d_r->dot( *d_p ) );

        } else {
            if ( d_bUsesPreconditioner ) {
                PROFILE( "CGSolver<T>:: rho_1 = <r,z>" );
                rho_1 = static_cast<T>( d_r->dot( *d_z ) );
            } else {
                rho_1 = static_cast<T>( d_dResidualNorm );
                rho_1 *= rho_1;
            }
        }

        if ( d_sVariant != "fcg" ) {
            const T beta = rho_1 / rho_0;
            d_p->axpy( beta, *d_p, *d_z );
        }

        if ( d_sVariant == "ipcg" ) {
            rho_1 = static_cast<T>( d_r->dot( *d_z ) );
        }
    }

    {
        PROFILE( "CGSolver<T>:: u->makeConsistent" );
        u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
    {
        PROFILE( "CGSolver<T>:: r = f-Au (final)" );
        d_pOperator->residual( f, u, d_r );
    }
    {
        PROFILE( "CGSolver<T>:: r->L2Norm (final)" );
        d_dResidualNorm = static_cast<T>( d_r->L2Norm() );
    }
    // final check updates flags if needed
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "CG: final residual: " << d_dResidualNorm
                  << ", iterations: " << d_iNumberIterations << ", convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "CG: final L2Norm of solution: " << u->L2Norm() << std::endl;
    }
}

} // namespace AMP::Solver
