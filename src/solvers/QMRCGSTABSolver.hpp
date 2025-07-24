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

    // Zero rhs implies zero solution, bail out early
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
    auto res_norm      = static_cast<T>( r0->L2Norm() );
    d_dInitialResidual = res_norm;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "QMRCGSTABSolver<T>::apply: initial L2Norm of solution vector: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::apply: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::apply: initial L2Norm of residual: " << res_norm
                  << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( res_norm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "QMRCGSTABSolver<T>::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // parameters in QMRCGSTAB
    T tau     = res_norm;
    T eta     = 0.0;
    T theta   = 0.0;
    auto rho1 = tau * tau;

    auto p = f->clone();
    p->copyVector( r0 );

    auto v = f->clone();
    v->zero();

    auto d = f->clone();
    d->zero();

    auto d2 = f->clone();
    d2->zero();

    auto r = f->clone();
    r->zero();

    auto s = f->clone();
    s->zero();

    auto t = f->clone();
    t->zero();

    auto z = f->clone();
    z->zero();

    auto x2 = f->clone();
    x2->zero();

    if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
        d_pNestedSolver->apply( p, z );
    } else {
        z = p;
    }

    d_pOperator->apply( z, v );

    bool converged = false;
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        int k     = d_iNumberIterations;
        auto rho2 = static_cast<T>( r0->dot( *v ) );

        // replace by soft-equal
        if ( rho2 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTABSolver<T>::apply: Breakdown, rho2 == 0" );
            break;
        }

        // replace by soft-equal
        if ( rho1 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTABSolver<T>::apply: Breakdown, rho1 == 0" );
            break;
        }

        auto alpha = rho1 / rho2;

        s->axpy( -alpha, *v, *r );

        // first quasi minimization and iterate update as per paper
        const auto theta2 = static_cast<T>( s->L2Norm() ) / tau;
        T c = static_cast<T>( 1.0 ) / ( std::sqrt( static_cast<T>( 1.0 ) + theta2 * theta2 ) );
        const auto tau2 = tau * theta2 * c;
        const auto eta2 = c * c * alpha;

        d2->axpy( theta * theta * eta / alpha, *d, *p );

        x2->axpy( eta2, *d2, *x );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            d_pNestedSolver->apply( s, z );
        } else {
            z = s;
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( z, t );

        const auto uu = static_cast<T>( s->dot( *t ) );
        const auto vv = static_cast<T>( t->dot( *t ) );

        if ( vv == static_cast<T>( 0.0 ) ) {
            AMP_ERROR( "Matrix is singular" );
        }

        const auto omega = uu / vv;

        if ( omega == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTABSolver<T>::apply: Breakdown, omega == 0" );
            break;
        }

        r->axpy( omega, *t, *s );

        // second quasi minimization and iterate update as per paper
        theta = static_cast<T>( s->L2Norm() ) / tau2;
        c     = static_cast<T>( 1.0 ) / ( std::sqrt( static_cast<T>( 1.0 ) + theta * theta ) );
        tau   = tau2 * theta * c;
        eta   = c * c * omega;

        d->axpy( theta2 * theta2 * eta2 / omega, *d2, *s );

        x->axpy( eta, *d, *x2 );

        // Use upper bound on residual norm to test convergence cheaply
        const auto res_bound = std::fabs( tau ) * std::sqrt( static_cast<T>( k + 1.0 ) );

        if ( checkStoppingCriteria( res_bound ) ) {
            if ( d_iDebugPrintInfoLevel > 1 ) {
                AMP::pout << "QMRCGSTAB: iteration " << k << ", residual " << res_bound
                          << std::endl;
            }
            res_norm  = res_bound; // this is likely an over-estimate
            converged = true;
            break;
        }

        rho2 = static_cast<T>( r->dot( *r0 ) );
        // replace by soft-equal
        if ( rho2 == static_cast<T>( 0.0 ) ) {
            d_ConvergenceStatus = SolverStatus::DivergedOther;
            AMP_WARNING( "QMRCGSTABSolver<T>::apply: Breakdown, rho2 == 0" );
            break;
        }

        const auto beta = ( alpha * rho2 ) / ( omega * rho1 );
        p->axpy( -omega, *v, *p );
        p->axpy( beta, *p, *r );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            d_pNestedSolver->apply( p, z );
        } else {
            z = p;
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( z, v );
        rho1 = rho2;

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "QMRCGSTAB: iteration " << k << ", residual " << tau << std::endl;
        }
    }

    if ( converged ) {
        // unwind the preconditioner if necessary
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            z->copyVector( x );
            d_pNestedSolver->apply( z, x );
        }
    }

    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // should this always be true since QMRCGSTAB only gives a bound on the
    // residual?
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, x, r0 );
        res_norm = static_cast<T>( r0->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( res_norm );
    }

    // Store final residual should it be queried elsewhere
    d_dResidualNorm = res_norm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "QMRCGSTABSolver<T>::apply: final L2Norm of solution: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::apply: final L2Norm of residual: " << res_norm
                  << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}
} // namespace AMP::Solver
