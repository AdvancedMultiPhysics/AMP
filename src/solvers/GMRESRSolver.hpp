#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/GMRESRSolver.h"
#include "AMP/solvers/GMRESSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


#include <cmath>
#include <limits>

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
template<typename T>
GMRESRSolver<T>::GMRESRSolver() : d_restarts( 0 )
{
}

template<typename T>
GMRESRSolver<T>::GMRESRSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters ), d_restarts( 0 )
{
    AMP_ASSERT( parameters );

    // Initialize
    initialize( parameters );
}

/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
template<typename T>
void GMRESRSolver<T>::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );
    auto db = parameters->d_db;

    getFromInput( db );

    // maximum dimension to allocate storage for
    const int max_dim = std::min( d_iMaxKrylovDimension, d_iMaxIterations );

    d_c.resize( max_dim );
    d_u.resize( max_dim );

    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
    } else {

        auto pcName  = db->getWithDefault<std::string>( "nested_solver", "Preconditioner" );
        auto outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;

        if ( outerDB && outerDB->keyExists( pcName ) ) {
            auto pcDB            = outerDB->getDatabase( pcName );
            auto innerParameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
            innerParameters->d_global_db = parameters->d_global_db;
            innerParameters->d_pOperator = d_pOperator;
            d_pNestedSolver              = AMP::Solver::SolverFactory::create( innerParameters );
            AMP_ASSERT( d_pNestedSolver );

        } else {
            if ( d_variant != "gcr" ) {
                AMP_ERROR( "The variant must be gcr or a nested solver must be provided" );
            }
        }
    }

    // Check if nested solver is GMRES and assert GMRES won't be restarted since it would
    // breack GMRESR
    if ( d_variant != "gcr" ) {
        if ( d_pNestedSolver->type() == "GMRESSolver" ) {
            auto gmres = std::dynamic_pointer_cast<AMP::Solver::GMRESSolver<T>>( d_pNestedSolver );
            AMP_INSIST( !gmres->restarted(), "Restarted GMRES can not be used within GMRESR" );
        }
    }
}

// Function to get values from input
template<typename T>
void GMRESRSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    // maximum dimension of space for truncation
    d_iMaxKrylovDimension = db->getWithDefault<int>( "max_dimension", 10 );
    d_variant             = db->getWithDefault<std::string>( "variant", "gmresr" );
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void GMRESRSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "GMRESRSolver<T>::apply" );

    if ( d_variant != "gcr" )
        AMP_INSIST( d_pNestedSolver, "Error: A nested solver must always be set for GMRESR" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // residual vector
    auto res = f->clone();
    auto c   = f->clone();
    auto z   = u->clone();

    z->zero();

    // the reference seems to suggest preconditioning explicitly the initial residual
    // but we will need to figure this out

    // compute the initial residual
    if ( d_bUseZeroInitialGuess ) {
        res->copyVector( f );
        u->zero();
    } else {
        d_pOperator->residual( f, u, res );
    }

    // compute residual norm
    auto r_norm = static_cast<T>( res->L2Norm() );

    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual = r_norm > std::numeric_limits<T>::epsilon() ? r_norm : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "GMRESRSolver<T>::apply: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESRSolver<T>::apply: initial L2Norm of rhs vector: " << f->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESRSolver<T>::apply: initial L2Norm of residual: " << r_norm << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( r_norm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "GMRESRSolver<T>::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    int k = -1;
    for ( d_iNumberIterations = 0; d_iNumberIterations < d_iMaxIterations; ++d_iNumberIterations ) {

        ++k;
        int j;
        if ( d_variant == "gmresr" ) {
            d_pNestedSolver->setZeroInitialGuess( true );
            d_pNestedSolver->apply( res, z );
        } else if ( d_variant == "gcr" ) {
            z->copyVector( res );
        } else {
            AMP_ERROR( "GMRESRSolver<T>::apply - unknown algorithmic variant " + d_variant +
                       " specified" );
        }

        z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        d_pOperator->apply( z, c );
        for ( int i = std::max( 0, k - d_iMaxKrylovDimension ); i <= k - 1; ++i ) {
            j = i % d_iMaxKrylovDimension;

            const auto alpha = d_c[j]->dot( *c );
            c->axpy( -alpha, *d_c[j], *c );
            z->axpy( -alpha, *d_u[j], *z );
        }

        auto c_norm = static_cast<T>( c->L2Norm() ); // add in a check for zero norm

        if ( AMP::Utilities::approx_equal<T>( c_norm, static_cast<T>( 0.0 ) ) &&
             !AMP::Utilities::approx_equal<T>( r_norm, static_cast<T>( 0.0 ) ) ) {
            AMP_ERROR( "GMRESR::Error breakdown condition encountered" );
        }

        j = k % d_iMaxKrylovDimension;

        if ( !d_c[j] )
            d_c[j] = c->clone();

        if ( !d_u[j] )
            d_u[j] = u->clone();

        d_c[j]->scale( static_cast<T>( 1.0 / c_norm ), *c );
        d_u[j]->scale( static_cast<T>( 1.0 / c_norm ), *z );

        auto cj_dot_rj = static_cast<T>( d_c[j]->dot( *res ) );
        u->axpy( cj_dot_rj, *d_u[j], *u );
        res->axpy( -cj_dot_rj, *d_c[j], *res );

        r_norm = static_cast<T>( res->L2Norm() );

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "GMRESR: iteration " << ( d_iNumberIterations + 1 ) << ", residual "
                      << r_norm << std::endl;
        }

        if ( checkStoppingCriteria( r_norm ) ) {
            break;
        }
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, res );
        r_norm = static_cast<T>( res->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( r_norm );
    }

    // Store final residual should it be queried elsewhere
    d_dResidualNorm = r_norm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "GMRESRSolver<T>::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESRSolver<T>::apply: final L2Norm of residual: " << r_norm << std::endl;
        AMP::pout << "GMRESRSolver<T>::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "GMRESRSolver<T>::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}
} // namespace AMP::Solver
