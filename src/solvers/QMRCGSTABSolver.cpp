#include "AMP/solvers/QMRCGSTABSolver.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/KrylovSolverParameters.h"
#include "ProfilerApp.h"


#include <array>
#include <cmath>
#include <limits>

namespace AMP {
namespace Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
QMRCGSTABSolver::QMRCGSTABSolver() {}

QMRCGSTABSolver::QMRCGSTABSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );

    // Initialize
    initialize( parameters );
}


/****************************************************************
 *  Destructor                                                   *
 ****************************************************************/
QMRCGSTABSolver::~QMRCGSTABSolver() {}

/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
void QMRCGSTABSolver::initialize( std::shared_ptr<const SolverStrategyParameters> params )
{
    auto parameters = std::dynamic_pointer_cast<const KrylovSolverParameters>( params );
    AMP_ASSERT( parameters );

    d_pPreconditioner = parameters->d_pPreconditioner;

    getFromInput( parameters->d_db );

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

// Function to get values from input
void QMRCGSTABSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    d_iMaxIterations = db->getWithDefault<double>( "max_iterations", 1000 );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "use_preconditioner", false );

    // default is right preconditioning, options are right, left, both
    if ( d_bUsesPreconditioner ) {
        d_preconditioner_side = db->getWithDefault<std::string>( "preconditioner_side", "right" );
    }
}

/****************************************************************
 *  Solve                                                        *
 * TODO: store convergence history, iterations, convergence reason
 ****************************************************************/
void QMRCGSTABSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> x )
{
    PROFILE_START( "solve" );

    // Check input vector states
    AMP_ASSERT(
        ( f->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::UNCHANGED ) ||
        ( f->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT(
        ( x->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::UNCHANGED ) ||
        ( x->getUpdateStatus() == AMP::LinearAlgebra::VectorData::UpdateState::LOCAL_CHANGED ) );

    // compute the norm of the rhs in order to compute
    // the termination criterion
    double f_norm = static_cast<double>( f->L2Norm() );

    // if the rhs is zero we try to converge to the relative convergence
    if ( f_norm == 0.0 ) {
        f_norm = 1.0;
    }

    const double terminate_tol = d_dRelativeTolerance * f_norm;

    if ( d_iDebugPrintInfoLevel > 2 ) {
        std::cout << "QMRCGSTABSolver::solve: initial L2Norm of solution vector: " << x->L2Norm()
                  << std::endl;
        std::cout << "QMRCGSTABSolver::solve: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
    }

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }

    // residual vector
    auto r0 = f->cloneVector();

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        r0->copyVector( f );
    } else {
        d_pOperator->residual( f, x, r0 );
    }

    // compute the current residual norm
    double res_norm = static_cast<double>( r0->L2Norm() );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        std::cout << "QMRCGSTAB: initial residual " << res_norm << std::endl;
    }

    // return if the residual is already low enough
    if ( res_norm < terminate_tol ) {
        // provide a convergence reason
        // provide history (iterations, conv history etc)
        if ( d_iDebugPrintInfoLevel > 0 ) {
            std::cout << "QMRCGSTABSolver::solve: initial residual norm " << res_norm
                      << " is below convergence tolerance: " << terminate_tol << std::endl;
        }
        return;
    }

    // parameters in QMRCGSTAB
    double tau   = res_norm;
    double eta   = 0.0;
    double theta = 0.0;
    double rho1  = tau * tau;

    auto p = f->cloneVector();
    p->copyVector( r0 );

    auto v = f->cloneVector();
    v->zero();

    auto d = f->cloneVector();
    d->zero();

    auto d2 = f->cloneVector();
    d2->zero();

    auto r = f->cloneVector();
    r->zero();

    auto s = f->cloneVector();
    s->zero();

    auto t = f->cloneVector();
    t->zero();

    auto z = f->cloneVector();
    z->zero();

    auto x2 = f->cloneVector();
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
        auto rho2 = static_cast<double>( r0->dot( *v ) );

        // replace by soft-equal
        if ( rho2 == 0.0 ) {
            // the method breaks down as the vectors are orthogonal to r
            AMP_ERROR( "QMRCGSTAB breakdown, <r0,v> == 0 " );
        }

        // replace by soft-equal
        if ( rho1 == 0.0 ) {
            // the method breaks down as it stagnates
            AMP_ERROR( "QMRCGSTAB breakdown, rho1==0 " );
        }

        auto alpha = rho1 / rho2;

        s->axpy( -alpha, *v, *r );

        // first quasi minimization and iterate update as per paper
        const auto theta2 = static_cast<double>( s->L2Norm() ) / tau;
        auto c            = 1.0 / ( std::sqrt( 1.0 + theta2 * theta2 ) );
        const auto tau2   = tau * theta2 * c;
        const auto eta2   = c * c * alpha;

        d2->axpy( theta * theta * eta / alpha, *d, *p );

        x2->axpy( eta2, *d2, *x );

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {

            d_pPreconditioner->apply( s, z );

        } else {
            z = s;
        }

        d_pOperator->apply( z, t );

        const auto uu = static_cast<double>( s->dot( *t ) );
        const auto vv = static_cast<double>( t->dot( *t ) );

        if ( vv == 0.0 ) {
            AMP_ERROR( "Matrix is singular" );
        }

        const auto omega = uu / vv;

        if ( omega == 0.0 ) {
            // the method breaks down as it stagnates
            AMP_ERROR( "QMRCGSTAB breakdown, omega==0.0 " );
        }

        r->axpy( omega, *t, *s );

        // second quasi minimization and iterate update as per paper
        theta = static_cast<double>( s->L2Norm() ) / tau2;
        c     = 1.0 / ( std::sqrt( 1.0 + theta * theta ) );
        tau   = tau2 * theta * c;
        eta   = c * c * omega;

        d->axpy( theta2 * theta2 * eta2 / omega, *d2, *s );

        x->axpy( eta, *d, *x2 );


        if ( std::fabs( tau ) * ( std::sqrt( (double) ( k + 1 ) ) ) <= terminate_tol ) {

            if ( d_iDebugPrintInfoLevel > 0 ) {
                std::cout << "QMRCGSTAB: iteration " << ( k + 1 ) << ", residual " << tau
                          << std::endl;
            }

            converged = true;
            break;
        }

        rho2 = static_cast<double>( r->dot( *r0 ) );
        // replace by soft-equal
        if ( rho2 == 0.0 ) {
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

        d_pOperator->apply( z, v );
        rho1 = rho2;

        if ( d_iDebugPrintInfoLevel > 0 ) {
            std::cout << "QMRCGSTAB: iteration " << ( k + 1 ) << ", residual " << tau << std::endl;
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
        std::cout << "L2Norm of solution: " << x->L2Norm() << std::endl;
    }

    PROFILE_STOP( "solve" );
}

/****************************************************************
 *  Function to set the register the operator                    *
 ****************************************************************/
void QMRCGSTABSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    AMP_ASSERT( op );

    d_pOperator = op;

    std::shared_ptr<AMP::Operator::LinearOperator> linearOperator =
        std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_ASSERT( linearOperator );
}
void QMRCGSTABSolver::resetOperator(
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
} // namespace Solver
} // namespace AMP
