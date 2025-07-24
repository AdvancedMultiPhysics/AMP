#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/GMRESSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "ProfilerApp.h"


#include <cmath>
#include <iomanip>
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

    registerOperator( d_pOperator );

    // maximum dimension to allocate storage for
    const int max_dim = std::min( d_iMaxKrylovDimension, d_iMaxIterations );
    d_dHessenberg.resize( max_dim + 1, max_dim + 1 );
    d_dHessenberg.fill( 0.0 );

    d_dcos.resize( max_dim + 1, 0.0 );
    d_dsin.resize( max_dim + 1, 0.0 );
    d_dw.resize( max_dim + 1, 0.0 );
    d_dy.resize( max_dim, 0.0 );

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
void GMRESSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    // the max iterations could be larger than the max Krylov dimension
    // in the case of restarted GMRES so we allow specification separately
    d_iMaxKrylovDimension      = db->getWithDefault<int>( "max_dimension", 100 );
    d_sOrthogonalizationMethod = db->getWithDefault<std::string>( "ortho_method", "MGS" );
    // uncomment once we optimize vec ops
    //    d_sOrthogonalizationMethod = db->getWithDefault<std::string>( "ortho_method", "CGS2" );

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

    d_iBasisAllocSize = db->getWithDefault<int>( "basis_allocation_size", 4 );
}

template<typename T>
void GMRESSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    // not sure about excluding op == d_pOperator
    d_pOperator = op;

    if ( d_pOperator ) {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        if ( linearOp ) {
            if ( d_bUsesPreconditioner ) {
                d_z = linearOp->getRightVector();
                if ( d_preconditioner_side == "right" ) {
                    d_z1 = d_z->clone();
                }
            }

            allocateBasis();
        }
    }
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
    d_iNumberIterations = 1;

    // Check input vector states
    AMP_ASSERT( ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );

    if ( !d_z ) {
        d_z = u->clone();

        if ( ( !d_z1 ) && ( d_preconditioner_side == "right" ) ) {
            d_z1 = d_z->clone();
        }
    }

    // allocate basis
    if ( d_vBasis.empty() )
        allocateBasis( u );

    // residual vector
    AMP::LinearAlgebra::Vector::shared_ptr res = d_vBasis[0];

    // compute the initial residual
    computeInitialResidual( d_bUseZeroInitialGuess, f, u, d_z, res );

    // compute residual norm
    auto beta       = static_cast<T>( res->L2Norm() );
    d_dResidualNorm = beta;
    // Override zero initial residual to force relative tolerance convergence
    // here to potentially handle singular systems
    d_dInitialResidual = beta > std::numeric_limits<T>::epsilon() ? beta : 1.0;


    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "GMRES: initial solution L2Norm: " << u->L2Norm() << std::endl;
        AMP::pout << "GMRES: initial rhs L2Norm: " << f->L2Norm() << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "GMRES: initial residual " << std::setw( 19 ) << d_dInitialResidual
                  << std::endl;
    }

    // return if the residual is already low enough
    if ( checkStoppingCriteria( d_dInitialResidual ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "GMRES: initial residual below tolerance" << std::endl;
        }
        return;
    }

    // normalize the first basis vector
    res->scale( static_cast<T>( 1.0 ) / beta );

    // 'w*e_1' is the rhs for the least squares minimization problem
    d_dw[0] = beta;

    auto v_norm = beta;

    int k = 0;
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        PROFILE( "GMRESSolver<T>::apply: main loop" );

        AMP::LinearAlgebra::Vector::shared_ptr v;
        AMP::LinearAlgebra::Vector::shared_ptr zb;

        // reuse basis vectors for restarts in order to avoid a clone
        if ( static_cast<int>( d_vBasis.size() ) <= k + 1 )
            allocateBasis( u );

        v = d_vBasis[k + 1];
        if ( d_bFlexibleGMRES )
            zb = d_zBasis[k];

        // construct the Krylov vector
        if ( d_bUsesPreconditioner ) {
            if ( d_preconditioner_side == "left" ) {
                d_vBasis[k]->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
                d_pOperator->apply( d_vBasis[k], d_z );
                d_pNestedSolver->apply( d_z, v );
            } else if ( d_preconditioner_side == "right" ) {
                if ( !d_bFlexibleGMRES ) {
                    d_pNestedSolver->apply( d_vBasis[k], d_z );
                    d_pOperator->apply( d_z, v );
                } else {
                    d_pNestedSolver->apply( d_vBasis[k], zb );
                    d_pOperator->apply( zb, v );
                }
            } else {
                AMP_ERROR( "Left and right preconditioning not enabled" );
            }
        } else {
            PROFILE( "GMRESSolver<T>::apply: v = Az" );
            d_z = d_vBasis[k];
            d_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            d_pOperator->apply( d_z, v );
        }

        // orthogonalize to previous vectors and
        // add new column to Hessenberg matrix
        orthogonalize( k + 1, v );

        v_norm = d_dHessenberg( k + 1, k );
        // replace the conditional by a soft equality
        // check for happy breakdown
        if ( v_norm != static_cast<T>( 0.0 ) ) {
            v->scale( static_cast<T>( 1.0 ) / v_norm );
        }

        // apply all previous Givens rotations to
        // the k-th column of the Hessenberg matrix
        for ( int i = 0; i < k; ++i ) {
            PROFILE( "GMRESSolver<T>::apply: Givens rotations on previous cols" );
            applyGivensRotation( i, k );
        }

        if ( v_norm != static_cast<T>( 0.0 ) ) {
            PROFILE( "GMRESSolver<T>::apply: Givens on current col" );
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
        d_dResidualNorm = v_norm = std::fabs( d_dw[k + 1] );

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "GMRES: iteration " << std::setw( 8 ) << d_iNumberIterations
                      << ", residual " << v_norm << std::endl;
        }

        ++k;

        if ( checkStoppingCriteria( d_dResidualNorm ) ) {
            break;
        }

        if ( k == d_iMaxKrylovDimension ) {
            if ( d_bRestart ) {
                // with restarts, you start over with the last solution as new initial guess to
                // compute the residal: r^new = Ax^old - b
                if ( d_iDebugPrintInfoLevel > 2 ) {
                    AMP::pout << "GMRES: restarting" << std::endl;
                }
                // note: backwardSolve and addCorrection only go up to k - 1 because k has
                // already been increased with ++k (so they really go up to k) compute y, the
                // solution to the least squares minimization problem
                backwardSolve( k - 1 );
                // update the current approximation with the correction
                addCorrection( k - 1, d_z, d_z1, u );
                computeInitialResidual( false, f, u, d_z, d_vBasis[0] );
                d_dResidualNorm = v_norm = static_cast<T>( d_vBasis[0]->L2Norm() );
                d_vBasis[0]->scale( static_cast<T>( 1.0 ) / v_norm );
                d_dw[0] = v_norm;
                ++d_restarts;
                k = 0;
            } else {
                // set diverged reason
                d_ConvergenceStatus = SolverStatus::DivergedOther;
                AMP_WARNING( "GMRES: Maximum Krylov dimension hit with restarts "
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
        addCorrection( k - 1, d_z, d_z1, u );
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, res );
        d_dResidualNorm = v_norm = static_cast<T>( res->L2Norm() );
        // final check updates flags if needed
        checkStoppingCriteria( d_dResidualNorm );
    }

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "GMRES: final residual: " << d_dResidualNorm
                  << ", iterations: " << d_iNumberIterations << ", convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "GMRES: final solution L2Norm: " << u->L2Norm() << std::endl;
    }
}

template<typename T>
std::vector<T> GMRESSolver<T>::basisInnerProducts( const int k,
                                                   std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    std::vector<T> hcol( k );

    auto vecOps = v->getVectorOperations();
    AMP_ASSERT( vecOps );
    {
        PROFILE( "GMRESSolver<T>::basisInnerProducts::localDots" );

        for ( int j = 0; j < k; ++j ) {
            hcol[j] = static_cast<T>(
                vecOps->localDot( *( v->getVectorData() ), *( d_vBasis[j]->getVectorData() ) ) );
        }
    }
    {
        PROFILE( "GMRESSolver<T>::basisInnerProducts::sumReduce" );
        const auto &comm = v->getComm();
        comm.sumReduce( hcol.data(), k );
    }
    return hcol;
}

template<typename T>
void GMRESSolver<T>::cgs( const int k, std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    auto hcol = basisInnerProducts( k, v );

    for ( int j = 0; j < k; ++j ) {
        v->axpy( -hcol[j], *d_vBasis[j], *v );
        d_dHessenberg( j, k - 1 ) = hcol[j];
    }
}

template<typename T>
void GMRESSolver<T>::cgs2( const int k, std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    auto hcol = basisInnerProducts( k, v );

    for ( int j = 0; j < k; ++j ) {
        PROFILE( "GMRESSolver<T>::cgs2: axpy1" );
        v->axpy( -hcol[j], *d_vBasis[j], *v );
    }

    auto scol = basisInnerProducts( k, v );

    for ( int j = 0; j < k; ++j ) {
        PROFILE( "GMRESSolver<T>::cgs2: axpy2" );
        v->axpy( -scol[j], *d_vBasis[j], *v );
    }

    for ( int j = 0; j < k; ++j ) {
        d_dHessenberg( j, k - 1 ) = hcol[j] + scol[j];
    }
}

template<typename T>
void GMRESSolver<T>::orthogonalize( const int k, std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    PROFILE( "GMRESSolver<T>::orthogonalize" );
    if ( d_sOrthogonalizationMethod == "CGS" ) {
        this->cgs( k, v );
    } else if ( d_sOrthogonalizationMethod == "CGS2" ) {
        this->cgs2( k, v );
    } else if ( d_sOrthogonalizationMethod == "MGS" ) {

        for ( int j = 0; j < k; ++j ) {
            const auto h_jk = static_cast<T>( v->dot( *d_vBasis[j] ) );
            v->axpy( -h_jk, *d_vBasis[j], *v );
            d_dHessenberg( j, k - 1 ) = h_jk;
        }
    } else {

        AMP_ERROR( "Unknown orthogonalization method in GMRES" );
    }

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
void GMRESSolver<T>::computeInitialResidual( bool use_zero_guess,
                                             std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> u,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> tmp,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> res )
{
    PROFILE( "GMRESSolver<T>::compute initial residual" );
    if ( use_zero_guess ) {
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "left" ) ) {
            d_pNestedSolver->apply( f, res );
        } else {
            res->copyVector( f );
        }
        u->zero();
    } else {
        u->makeConsistent();
        if ( d_bUsesPreconditioner && ( d_preconditioner_side == "left" ) ) {
            d_pOperator->residual( f, u, tmp );
            d_pNestedSolver->apply( tmp, res );
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
            d_pNestedSolver->apply( z, z1 );
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


template<typename T>
void GMRESSolver<T>::allocateBasis( std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "GMRESSolver<T>::allocateBasis" );
    AMP_ASSERT( d_pOperator );

    std::shared_ptr<AMP::LinearAlgebra::Vector> v;

    if ( u ) {
        v = u->clone();
    } else {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        if ( linearOp )
            v = linearOp->getRightVector();
    }

    if ( v ) {
        d_vBasis.push_back( v );
        if ( d_bFlexibleGMRES )
            d_zBasis.push_back( v->clone() );

        for ( auto i = 1; i < d_iBasisAllocSize; i++ ) {
            d_vBasis.push_back( v->clone() );
            if ( d_bFlexibleGMRES )
                d_zBasis.push_back( v->clone() );
        }
    }
}

} // namespace AMP::Solver
