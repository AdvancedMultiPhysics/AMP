#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/QMRCGSTABSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "ProfilerApp.h"

#include <array>
#include <cmath>
#include <limits>

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
template<typename T>
QMRCGSTABSolver<T>::QMRCGSTABSolver( std::shared_ptr<SolverStrategyParameters> parameters )
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
void QMRCGSTABSolver<T>::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;
    getFromInput( db );

    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
        d_pNestedSolver->setIsNestedSolver( true );
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
                d_pNestedSolver->setIsNestedSolver( true );
                AMP_ASSERT( d_pNestedSolver );
            }
        }
    }
}

// Function to get values from input
template<typename T>
void QMRCGSTABSolver<T>::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );

    // default is right preconditioning, options are right, left, both
    if ( d_bUsesPreconditioner ) {
        d_preconditioner_side = db->getWithDefault<std::string>( "preconditioner_side", "right" );
    }
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void QMRCGSTABSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                std::shared_ptr<AMP::LinearAlgebra::Vector> x )
{
    PROFILE( "QMRCGSTABSolver<T>::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // compute the norm of the rhs in order to compute
    // the termination criterion
    auto f_norm = static_cast<T>( f->L2Norm() );

    // Zero rhs implies zero solution, bail out early --check whether this should be removed!!
    if ( f_norm == static_cast<T>( 0.0 ) ) {
        x->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "TFQMRSolver<T>::apply: solution is zero" << std::endl;
        }
        return;
    }

    // residual vector
    auto r0 = f->clone();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        r0->copyVector( f );
        x->zero();
    } else {
        d_pOperator->residual( f, x, r0 );
    }

    // compute the current residual norm
    d_dResidualNorm    = static_cast<T>( r0->L2Norm() );
    d_dInitialResidual = d_dResidualNorm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "QMRCGSTAB: initial L2Norm of residual: " << d_dInitialResidual << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "QMRCGSTAB: initial L2Norm of solution vector: " << x->L2Norm() << std::endl;
        AMP::pout << "QMRCGSTAB: initial L2Norm of rhs vector: " << f_norm << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( d_dResidualNorm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "QMRCGSTAB: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in QMRCGSTAB
    auto tau  = static_cast<T>( d_dInitialResidual );
    T eta     = 1.0;
    T theta   = 1.0;
    auto rho1 = tau * tau;

    auto p  = f->clone();
    auto v  = f->clone();
    auto d  = f->clone();
    auto d2 = f->clone();
    auto r  = f->clone();
    auto s  = f->clone();
    auto t  = f->clone();
    auto z  = f->clone();
    auto x2 = f->clone();

    p->copyVector( r0 );
    v->zero();
    d->zero();
    d2->zero();
    r->zero();
    s->zero();
    t->zero();
    z->zero();
    x2->zero();

    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {

        if ( d_bUsesPreconditioner ) {
            AMP_INSIST( d_preconditioner_side == "right",
                        "QMRCGSTAB only uses right preconditioning" );
            d_pNestedSolver->apply( p, z );
        } else {
            z->copyVector( p );
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( z, v );

        int k     = d_iNumberIterations;
        auto rho2 = static_cast<T>( r0->dot( *v ) );

        // replace by soft-equal
        if ( rho2 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTAB: Breakdown, rho2 == 0, division by zero" );
            break;
        }

        // replace by soft-equal
        if ( rho1 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTAB: Breakdown, rho1 == 0, stagnated" );
            break;
        }

        auto alpha = rho1 / rho2;

        s->axpy( -alpha, *v, *r );

        // first quasi minimization and iterate update as per paper
        const auto theta2 = static_cast<T>( s->L2Norm() ) / tau;
        auto c = static_cast<T>( 1.0 ) / ( std::sqrt( static_cast<T>( 1.0 ) + theta2 * theta2 ) );
        const auto tau2 = tau * theta2 * c;
        const auto eta2 = c * c * alpha;

        d2->axpy( theta * theta * eta / alpha, *d, *z );

        x2->axpy( eta2, *d2, *x );

        if ( d_bUsesPreconditioner ) {
            d_pNestedSolver->apply( s, z );
        } else {
            z->copyVector( s );
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( z, t );

        auto uu       = static_cast<T>( s->dot( *t ) );
        const auto vv = static_cast<T>( t->dot( *t ) );

        if ( vv == static_cast<T>( 0.0 ) ) {
            // Note this section is based on Petsc implementation
            // Could be we have converged or have a breakdown
            auto uu = static_cast<T>( s->dot( *s ) );
            if ( uu != 0.0 ) {
                d_ConvergenceStatus = SolverStatus::DivergedOther;
                AMP_WARNING( "QMRCGSTAB: Breakdown, vv == 0, division by zero" );
                break;
            }
            x->axpy( alpha, *z, *x );
            d_dResidualNorm     = 0.0;
            d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
            break;
        }

        auto nv = static_cast<T>( v->L2Norm() );

        // correction from Petsc implementation
        if ( nv == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTAB: Breakdown, singular system" );
            break;
        }

        // correction from Petsc implementation
        if ( uu == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTAB: Breakdown, stagnation" );
            break;
        }

        const auto omega = uu / vv;

        r->axpy( -omega, *t, *s );

        // correction from Petsc implementation
        if ( tau2 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTAB: Breakdown, tau2 == 0" );
            break;
        }

        // second quasi minimization and iterate update as per paper
        d_dResidualNorm = r->L2Norm();
        theta           = static_cast<T>( d_dResidualNorm ) / tau2;
        c   = static_cast<T>( 1.0 ) / ( std::sqrt( static_cast<T>( 1.0 ) + theta * theta ) );
        tau = static_cast<T>( d_dResidualNorm );
        tau = tau * c;
        eta = c * c * omega;

        d->axpy( theta2 * theta2 * eta2 / omega, *d2, *z );

        x->axpy( eta, *d, *x2 );

        rho2 = static_cast<T>( r->dot( *r0 ) );

        // The commented section is from the original paper and should be re-enabled if possible
#if 0
        // Use upper bound on residual norm to test convergence cheaply
        const auto res_bound = std::fabs( tau ) * std::sqrt( static_cast<T>( k + 1.0 ) );

        if ( checkStoppingCriteria( res_bound ) ) {
            if ( d_iDebugPrintInfoLevel > 1 ) {
                AMP::pout << "QMRCGSTAB: iteration " << k << ", residual " << res_bound
                          << std::endl;
            }
            res_norm = res_bound; // this is likely an over-estimate
            break;
        }
#endif

        const auto beta = ( alpha * rho2 ) / ( omega * rho1 );
        p->axpy( beta, *p, *r );
        p->axpy( -omega * beta, *v, *p );
        rho1 = rho2;

        x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->residual( f, x, z );
        // for now explicitly compute residual
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "QMRCGSTAB: iteration " << k << ", residual " << d_dResidualNorm
                      << std::endl;
        }
        if ( checkStoppingCriteria( d_dResidualNorm ) ) {
            break;
        }
    }

    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // should this always be true since QMRCGSTAB only gives a bound on the
    // residual?
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, x, r0 );
        d_dResidualNorm = static_cast<T>( r0->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( d_dResidualNorm );
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "QMRCGSTAB: L2Norm of residual: " << d_dResidualNorm << std::endl;
        AMP::pout << "QMRCGSTAB: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "QMRCGSTAB: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "QMRCGSTAB: final L2Norm of solution: " << x->L2Norm() << std::endl;
    }
}
} // namespace AMP::Solver
