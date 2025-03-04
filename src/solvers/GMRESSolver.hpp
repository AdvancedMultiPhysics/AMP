#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/GMRESSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "ProfilerApp.h"


#include <cmath>
#include <limits>

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
template<typename T>
GMRESSolver<T>::GMRESSolver() : d_restarts( 0 )
{
}

template<typename T>
GMRESSolver<T>::GMRESSolver( std::shared_ptr<SolverStrategyParameters> parameters )
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
void GMRESSolver<T>::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );
    auto db = parameters->d_db;

    getFromInput( db );

    // maximum dimension to allocate storage for
    const int max_dim = std::min( d_iMaxKrylovDimension, d_iMaxIterations );
    d_dHessenberg.resize( max_dim + 1, max_dim + 1 );
    d_dHessenberg.fill( 0.0 );

    d_dcos.resize( max_dim + 1, 0.0 );
    d_dsin.resize( max_dim + 1, 0.0 );
    d_dw.resize( max_dim + 1, 0.0 );
    d_dy.resize( max_dim, 0.0 );

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
void GMRESSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    // the max iterations could be larger than the max Krylov dimension
    // in the case of restarted GMRES so we allow specification separately
    d_iMaxKrylovDimension      = db->getWithDefault<int>( "max_dimension", 100 );
    d_sOrthogonalizationMethod = db->getWithDefault<std::string>( "ortho_method", "MGS" );

    // default is right preconditioning, options are right, left, both
    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
    if ( d_bUsesPreconditioner ) {
        d_preconditioner_side = db->getWithDefault<std::string>( "preconditioner_side", "right" );
    }

    d_bRestart = db->getWithDefault<bool>( "gmres_restart", false );

    d_bFlexibleGMRES = db->getWithDefault<bool>( "flexible_gmres", false );

    if ( !d_bUsesPreconditioner )
        d_bFlexibleGMRES = false;

    if ( d_bUsesPreconditioner && d_preconditioner_side == "left" && d_bFlexibleGMRES )
        AMP_ERROR( "Flexible GMRES needs right preconditioning" );
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void GMRESSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                            std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "GMRESSolver<T>::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    // residual vector
    AMP::LinearAlgebra::Vector::shared_ptr res = f->clone();

    // z is only used if there is preconditioning
    AMP::LinearAlgebra::Vector::shared_ptr z;
    // z1 is only used if there is right preconditioning
    AMP::LinearAlgebra::Vector::shared_ptr z1;

    if ( d_bUsesPreconditioner ) {
        z = f->clone();
        if ( d_preconditioner_side == "right" ) {
            z1 = f->clone();
        }
    }

    // compute the initial residual
    computeInitialResidual( d_bUseZeroInitialGuess, f, u, z, res );

    // compute residual norm
    auto beta = static_cast<T>( res->L2Norm() );
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual = beta > std::numeric_limits<T>::epsilon() ? beta : 1.0;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "GMRESSolver<T>::apply: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESSolver<T>::apply: initial L2Norm of rhs vector: " << f->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESSolver<T>::apply: initial L2Norm of residual: " << beta << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( beta ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "GMRESSolver<T>::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // normalize the first basis vector
    res->scale( static_cast<T>( 1.0 ) / beta );

    // push the residual as the first basis vector
    d_vBasis.push_back( res );

    // 'w*e_1' is the rhs for the least squares minimization problem
    d_dw[0] = beta;

    auto v_norm = beta;

    int k = 0;
    for ( d_iNumberIterations = 0; d_iNumberIterations < d_iMaxIterations; ++d_iNumberIterations ) {

        AMP::LinearAlgebra::Vector::shared_ptr v;
        AMP::LinearAlgebra::Vector::shared_ptr zb;

        if ( k + 1 < static_cast<int>( d_vBasis.size() ) ) {
            // reuse basis vectors for restarts in order to avoid a clone
            v = d_vBasis[k + 1];
            if ( d_bFlexibleGMRES )
                // z_Basis is increased in size one behind wrt d_vBasis
                zb = d_zBasis[k];
        } else {
            // clone off of the rhs to create a new basis vector
            v = f->clone();
            d_vBasis.push_back( v );
            if ( d_bFlexibleGMRES )
                zb = f->clone();
        }

        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "left" ) ) {
            d_pOperator->apply( d_vBasis[k], z );
            z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            // construct the Krylov vector
            d_pPreconditioner->apply( z, v );
        } else {
            if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {
                // the makeConsistent calls below are there because the commented condition
                // on status appears not to be working. They are required or we have to change
                // policy on what the status of a vector is coming out of a solver.
                if ( !d_bFlexibleGMRES ) {
                    d_pPreconditioner->apply( d_vBasis[k], z );
                    //                    if ( z->getUpdateStatus() !=
                    //                         AMP::LinearAlgebra::UpdateState::UNCHANGED
                    //                         )
                    z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                    d_pOperator->apply( z, v );
                } else {
                    d_pPreconditioner->apply( d_vBasis[k], zb );
                    d_zBasis.push_back( zb );
                    zb->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                    d_pOperator->apply( zb, v );
                }
            } else {
                z = d_vBasis[k];
                z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                d_pOperator->apply( z, v );
            }
        }

        // orthogonalize to previous vectors and
        // add new column to Hessenberg matrix
        orthogonalize( k + 1, v );

        v_norm = d_dHessenberg( k + 1, k );
        // replace the conditional by a soft equality
        // check for happy breakdown
        if ( v_norm != static_cast<T>( 0.0 ) ) {
            v->scale( static_cast<T>( 1.0 ) / v_norm );
            v->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        }

        // apply all previous Givens rotations to
        // the k-th column of the Hessenberg matrix
        for ( int i = 0; i < k; ++i ) {
            applyGivensRotation( i, k );
        }

        if ( v_norm != static_cast<T>( 0.0 ) ) {
            // compute and store the Givens rotation that zeroes out
            // the subdiagonal for the current column
            computeGivensRotation( k );
            // zero out the subdiagonal
            applyGivensRotation( k, k );
            // explicitly set subdiag to zero to prevent round-off
            d_dHessenberg( k + 1, k ) = static_cast<T>( 0.0 );

            // explicitly apply the newly computed
            // Givens rotations to the rhs vector
            auto x = d_dw[k];
            auto c = d_dcos[k];
            auto s = d_dsin[k];

            d_dw[k]     = c * x;
            d_dw[k + 1] = -s * x;
        }

        // this is the norm of the residual thanks to the Givens rotations
        v_norm = std::fabs( d_dw[k + 1] );

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "GMRES: iteration " << ( d_iNumberIterations + 1 ) << ", residual "
                      << v_norm << std::endl;
        }

        if ( checkStoppingCriteria( v_norm ) ) {
            break;
        }

        ++k;
        if ( k == d_iMaxKrylovDimension ) {
            if ( d_bRestart ) {
                // with restarts, you start over with the last solution as new initial guess to
                // compute the residal: r^new = Ax^old - b
                if ( d_iDebugPrintInfoLevel > 2 ) {
                    AMP::pout << "GMRES: restarting" << std::endl;
                }
                // note: backwardSolve and addCorrection only go up to k - 1 because k has already
                // been increased with ++k (so they really go up to k)
                // compute y, the solution to the least squares minimization problem
                backwardSolve( k - 1 );
                // update the current approximation with the correction
                addCorrection( k - 1, z, z1, u );
                computeInitialResidual( false, f, u, z, d_vBasis[0] );
                v_norm = static_cast<T>( d_vBasis[0]->L2Norm() );
                d_vBasis[0]->scale( static_cast<T>( 1.0 ) / v_norm );
                d_dw[0] = v_norm;
                ++d_restarts;
                k = 0;
            } else {
                // set diverged reason
                d_ConvergenceStatus = SolverStatus::DivergedOther;
                AMP_WARNING( "GMRESSolver<T>::apply: Maximum Krylov dimension hit with restarts "
                             "disabled" );
                break;
            }
        }
    }

    if ( d_iDebugPrintInfoLevel > 3 ) {
        d_dHessenberg.print( AMP::pout, "Hessenberg Matrix H" );
    }

    if ( k > 0 ) {
        // compute y, the solution to the least squares minimization problem
        backwardSolve( k - 1 );

        // update the current approximation with the correction
        addCorrection( k - 1, z, z1, u );
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, res );
        v_norm = static_cast<T>( res->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( v_norm );
    }

    // Store final residual should it be queried elsewhere
    d_dResidualNorm = v_norm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "GMRESSolver<T>::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "GMRESSolver<T>::apply: final L2Norm of residual: " << v_norm << std::endl;
        AMP::pout << "GMRESSolver<T>::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "GMRESSolver<T>::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

template<typename T>
void GMRESSolver<T>::orthogonalize( const int k, std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    if ( d_sOrthogonalizationMethod == "CGS" ) {

        AMP_ERROR( "Classical Gram-Schmidt not implemented as yet" );
    } else if ( d_sOrthogonalizationMethod == "MGS" ) {

        for ( int j = 0; j < k; ++j ) {

            const auto h_jk = static_cast<T>( v->dot( *d_vBasis[j] ) );
            v->axpy( -h_jk, *d_vBasis[j], *v );
            d_dHessenberg( j, k - 1 ) = h_jk;
        }
    } else {

        AMP_ERROR( "Unknown orthogonalization method in GMRES" );
    }

    //    v->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // h_{k+1, k}
    const auto v_norm         = static_cast<T>( v->L2Norm() );
    d_dHessenberg( k, k - 1 ) = v_norm; // adjusting for zero starting index
}

template<typename T>
void GMRESSolver<T>::applyGivensRotation( const int i, const int k )
{
    // updates column k of the Hessenberg matrix by applying the i-th Givens rotations

    auto x = d_dHessenberg( i, k );
    auto y = d_dHessenberg( i + 1, k );
    auto c = d_dcos[i];
    auto s = d_dsin[i];

    d_dHessenberg( i, k )     = c * x + s * y;
    d_dHessenberg( i + 1, k ) = -s * x + c * y;
}

template<typename T>
void GMRESSolver<T>::computeGivensRotation( const int k )
{
    // computes the Givens rotation required to zero out
    // the subdiagonal on column k of the Hessenberg matrix

    // The implementation here follows Algorithm 1 in
    // "On Computing Givens rotations reliably and efficiently"
    // by D. Bindel, J. Demmel, W. Kahan, O. Marques
    // UT-CS-00-449, October 2000.

    auto f = d_dHessenberg( k, k );
    auto g = d_dHessenberg( k + 1, k );

    decltype( f ) c, s;
    if ( g == static_cast<T>( 0.0 ) ) {
        c = static_cast<T>( 1.0 );
        s = static_cast<T>( 0.0 );
    } else if ( f == static_cast<T>( 0.0 ) ) {
        c = static_cast<T>( 0.0 );
        s = ( g < static_cast<T>( 0.0 ) ) ? static_cast<T>( -1.0 ) : static_cast<T>( 1.0 );
    } else {
        decltype( f ) r;
        r = std::sqrt( f * f + g * g );
        r = static_cast<T>( 1.0 ) / r;
        c = std::fabs( f ) * r;
        s = std::copysign( g * r, f );
    }

    d_dcos[k] = c;
    d_dsin[k] = s;
}
template<typename T>
void GMRESSolver<T>::backwardSolve( const int nr )
{
    // lower corner
    d_dy[nr] = d_dw[nr] / d_dHessenberg( nr, nr );

    // backwards solve
    for ( int k = nr - 1; k >= 0; --k ) {

        d_dy[k] = d_dw[k];

        for ( int i = k + 1; i <= nr; ++i ) {
            d_dy[k] -= d_dHessenberg( k, i ) * d_dy[i];
        }

        d_dy[k] = d_dy[k] / d_dHessenberg( k, k );
    }
}

template<typename T>
void GMRESSolver<T>::resetOperator(
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

template<typename T>
void GMRESSolver<T>::computeInitialResidual( bool use_zero_guess,
                                             std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> u,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> tmp,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> res )
{
    if ( use_zero_guess ) {
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "left" ) ) {
            tmp->copyVector( f );
            d_pPreconditioner->apply( tmp, res );
        } else {
            res->copyVector( f );
        }
        u->zero();
    } else {
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "left" ) ) {
            d_pOperator->residual( f, u, tmp );
            d_pPreconditioner->apply( tmp, res );
        } else {
            // this computes res = f - L(u)
            d_pOperator->residual( f, u, res );
        }
    }
}

template<typename T>
void GMRESSolver<T>::addCorrection( const int nr,
                                    std::shared_ptr<AMP::LinearAlgebra::Vector> z,
                                    std::shared_ptr<AMP::LinearAlgebra::Vector> z1,
                                    std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{

    if ( d_bUsesPreconditioner && ( d_preconditioner_side == "right" ) ) {

        if ( !d_bFlexibleGMRES ) {
            z->setToScalar( static_cast<T>( 0.0 ) );

            for ( int i = 0; i <= nr; ++i ) {
                // this is V_m * y_m written as \sum_i i-th col(V_m) * i-th entry of y_m
                z->axpy( d_dy[i], *d_vBasis[i], *z );
            }

            // this solves M z1 = V_m * y_m (so z1= inv(M) * V_m * y_m)
            d_pPreconditioner->apply( z, z1 );
        } else {
            z1->setToScalar( static_cast<T>( 0.0 ) );
            for ( int i = 0; i <= nr; ++i ) {
                // this is Z_m * y_m written as \sum_i i-th col(Z_m) * i-th entry of y_m
                z1->axpy( d_dy[i], *d_zBasis[i], *z1 );
            }
        }
        u->axpy( 1.0, *z1, *u );

    } else {
        for ( int i = 0; i <= nr; ++i ) {
            u->axpy( d_dy[i], *d_vBasis[i], *u );
        }
    }
}
} // namespace AMP::Solver
