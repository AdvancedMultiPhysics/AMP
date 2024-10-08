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
        d_pPreconditioner = parameters->d_pNestedSolver;
    } else {
        if ( d_bUsesPreconditioner ) {
            auto pcName  = db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
            auto outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;
            if ( outerDB ) {
                auto pcDB       = outerDB->getDatabase( pcName );
                auto parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
                parameters->d_pOperator = d_pOperator;
                d_pPreconditioner       = AMP::Solver::SolverFactory::create( parameters );
                AMP_ASSERT( d_pPreconditioner );
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
 * TODO: store convergence history, iterations, convergence reason
 ****************************************************************/
template<typename T>
void QMRCGSTABSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                std::shared_ptr<AMP::LinearAlgebra::Vector> x )
{
    PROFILE( "solve" );

    // Check input vector states
    AMP_ASSERT( ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT( ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( x->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // compute the norm of the rhs in order to compute
    // the termination criterion
    auto f_norm = static_cast<T>( f->L2Norm() );

    // if the rhs is zero we try to converge to the relative convergence
    if ( f_norm == static_cast<T>( 0.0 ) ) {
        f_norm = static_cast<T>( 1.0 );
    }

    const T terminate_tol = std::max( static_cast<T>( d_dRelativeTolerance * f_norm ),
                                      static_cast<T>( d_dAbsoluteTolerance ) );

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "QMRCGSTABSolver<T>::solve: initial L2Norm of solution vector: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "QMRCGSTABSolver<T>::solve: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
    }

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }

    // residual vector
    auto r0 = f->clone();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        r0->copyVector( f );
    } else {
        d_pOperator->residual( f, x, r0 );
    }

    // compute the current residual norm
    auto res_norm = static_cast<T>( r0->L2Norm() );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "QMRCGSTAB: initial residual " << res_norm << std::endl;
    }

    // return if the residual is already low enough
    if ( res_norm < terminate_tol ) {
        // provide a convergence reason
        // provide history (iterations, conv history etc)
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "QMRCGSTABSolver<T>::solve: initial residual norm " << res_norm
                      << " is below convergence tolerance: " << terminate_tol << std::endl;
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

        d_pPreconditioner->apply( p, z );

    } else {

        z = p;
    }

    d_pOperator->apply( z, v );

    int k = 0;

    bool converged = false;

    while ( k < d_iMaxIterations ) {

        ++k;
        auto rho2 = static_cast<T>( r0->dot( *v ) );

        // replace by soft-equal
        if ( rho2 == static_cast<T>( 0.0 ) ) {
            // the method breaks down as the vectors are orthogonal to r
            AMP_ERROR( "QMRCGSTAB breakdown, <r0,v> == 0 " );
        }

        // replace by soft-equal
        if ( rho1 == static_cast<T>( 0.0 ) ) {
            // the method breaks down as it stagnates
            AMP_ERROR( "QMRCGSTAB breakdown, rho1==0 " );
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

            d_pPreconditioner->apply( s, z );

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
            // the method breaks down as it stagnates
            AMP_ERROR( "QMRCGSTAB breakdown, omega==0.0 " );
        }

        r->axpy( omega, *t, *s );

        // second quasi minimization and iterate update as per paper
        theta = static_cast<T>( s->L2Norm() ) / tau2;
        c     = static_cast<T>( 1.0 ) / ( std::sqrt( static_cast<T>( 1.0 ) + theta * theta ) );
        tau   = tau2 * theta * c;
        eta   = c * c * omega;

        d->axpy( theta2 * theta2 * eta2 / omega, *d2, *s );

        x->axpy( eta, *d, *x2 );


        if ( std::fabs( tau ) * ( std::sqrt( (T) ( k + 1 ) ) ) <= terminate_tol ) {

            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << "QMRCGSTAB: iteration " << ( k + 1 ) << ", residual " << tau
                          << std::endl;
            }

            converged = true;
            break;
        }

        rho2 = static_cast<T>( r->dot( *r0 ) );
        // replace by soft-equal
        if ( rho2 == static_cast<T>( 0.0 ) ) {
            // the method breaks down as rho2==0
            AMP_ERROR( "QMRCGSTAB breakdown, rho2 == 0 " );
        }

        const auto beta = ( alpha * rho2 ) / ( omega * rho1 );
        p->axpy( -omega, *v, *p );
        p->axpy( beta, *p, *r );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            d_pPreconditioner->apply( p, z );
        } else {
            z = p;
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( z, v );
        rho1 = rho2;

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "QMRCGSTAB: iteration " << ( k + 1 ) << ", residual " << tau << std::endl;
        }
    }

    if ( converged ) {
        // unwind the preconditioner if necessary
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
            z->copyVector( x );
            d_pPreconditioner->apply( z, x );
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "L2Norm of solution: " << x->L2Norm() << std::endl;
    }
}

template<typename T>
void QMRCGSTABSolver<T>::resetOperator(
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
