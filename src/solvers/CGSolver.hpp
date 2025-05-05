#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/CGSolver.h"
#include "AMP/solvers/SolverFactory.h"
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
void CGSolver<T>::initialize(
    std::shared_ptr<const AMP::Solver::SolverStrategyParameters> parameters )
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

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void CGSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f_in,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> u_in )
{
    PROFILE( "CGSolver<T>::apply" );

    AMP_ASSERT( d_pOperator );
    auto u = d_pOperator->subsetInputVector( u_in );
    auto f = d_pOperator->subsetOutputVector( f_in );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // z will store r when a preconditioner is not present
    // and will store the result of a preconditioner solve
    // when a preconditioner is present
    auto z = u->clone();
    auto p = z->clone();
    auto w = f->clone();

    // residual vector
    AMP::LinearAlgebra::Vector::shared_ptr r = f->clone();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        r->copyVector( f );
    } else {
        d_pOperator->residual( f, u, r );
    }

    // Store initial residual
    auto d_dResidualNorm = static_cast<T>( r->L2Norm() );
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual =
        d_dResidualNorm > std::numeric_limits<T>::epsilon() ? d_dResidualNorm : 1.0;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "CG: initial residual: " << d_dResidualNorm << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "CG: initial L2Norm of solution vector: " << u->L2Norm() << std::endl;
        AMP::pout << "CG: initial L2Norm of rhs vector: " << f->L2Norm() << std::endl;
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
        d_pPreconditioner->apply( r, z );
    } else {
        z->copyVector( r );
    }

    if ( d_sVariant != "pcg" ) {
        d_vDirs[0] = u->clone();
    }

    if ( d_sVariant == "fcg" ) {
        d_vDirs[0]->copyVector( z );
    }

    if ( d_sVariant == "ipcg" ) {
        d_vDirs[0]->copyVector( r );
    }

    auto rho_1 = static_cast<T>( z->dot( *r ) );
    auto rho_0 = rho_1;

    p->copyVector( z );

    auto k = -1;

    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        ++k;
        p->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        // w = Ap
        d_pOperator->apply( p, w );

        // gamma = p'Ap
        auto gamma = static_cast<T>( w->dot( *p ) );

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

        auto alpha = rho_1 / gamma;

        u->axpy( alpha, *p, *u );
        r->axpy( -alpha, *w, *r );

        // compute the current residual norm
        d_dResidualNorm = static_cast<T>( r->L2Norm() );
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "CG: iteration " << d_iNumberIterations << ", residual " << d_dResidualNorm
                      << std::endl;
        }

        // check if converged
        if ( checkStoppingCriteria( d_dResidualNorm ) ) {
            break;
        }

        // apply the preconditioner if it exists
        if ( d_bUsesPreconditioner ) {
            d_pPreconditioner->apply( r, z );
        } else {
            z->copyVector( r );
        }

        rho_0 = rho_1;

        if ( d_sVariant == "ipcg" ) {

            d_vDirs[0]->axpy( static_cast<T>( -1.0 ), *d_vDirs[0], *r );
            rho_1 = static_cast<T>( d_vDirs[0]->dot( *z ) );
            d_vDirs[0]->copyVector( r );

        } else if ( d_sVariant == "fcg" ) {

            z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            d_pOperator->apply( z, w );
            std::vector<T> vbeta( d_max_dimension );

            // This should be combined into a single operation across multiple vectors
            for ( auto j = std::max( 0, k - d_max_dimension + 1 ); j <= k; ++j ) {
                auto idx   = j % d_max_dimension;
                auto dp    = static_cast<T>( d_vDirs[idx]->dot( *w ) );
                vbeta[idx] = dp / d_gamma[idx];
            }

            auto d = ( k + 1 ) % d_max_dimension;

            if ( !d_vDirs[d] )
                d_vDirs[d] = u->clone();

            d_vDirs[d]->copyVector( z );

            for ( auto j = std::max( 0, k - d_max_dimension + 1 ); j <= k; ++j ) {
                auto idx = j % d_max_dimension;
                d_vDirs[d]->axpy( -vbeta[idx], *d_vDirs[idx], *d_vDirs[d] );
            }

            p->copyVector( d_vDirs[d] );
            rho_1 = static_cast<T>( r->dot( *p ) );

        } else {
            rho_1 = static_cast<T>( r->dot( *z ) );
        }

        if ( d_sVariant != "fcg" ) {
            const T beta = rho_1 / rho_0;
            p->axpy( beta, *p, *z );
        }

        if ( d_sVariant == "ipcg" ) {
            rho_1 = static_cast<T>( r->dot( *z ) );
        }
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    d_pOperator->residual( f, u, r );
    d_dResidualNorm = static_cast<T>( r->L2Norm() );
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
