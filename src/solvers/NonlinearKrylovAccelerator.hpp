#include "AMP/IO/PIO.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/NonlinearKrylovAccelerator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"

#include "ProfilerApp.h"

#include <iomanip>

#define EOL -1

namespace AMP::Solver {

template<typename T>
NonlinearKrylovAccelerator<T>::NonlinearKrylovAccelerator(
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> params )
    : AMP::Solver::SolverStrategy( params )
{
    getFromInput( d_db );
    initialize( params );

    // initialize the preconditioner
    if ( d_uses_preconditioner ) {
        // 3 cases need to be addressed:
        // 1. A preconditioner is being passed in
        // 2. A preconditioner solver name is being specified
        // 3. A preconditioner solver name is not specified but the user will set on their own
        auto parameters =
            std::dynamic_pointer_cast<AMP::Solver::SolverStrategyParameters>( params );
        if ( parameters ) {
            d_preconditioner = parameters->d_pNestedSolver;
        }

        if ( d_preconditioner ) {
            if ( d_pOperator && ( !d_preconditioner->getOperator() ) ) {
                auto pcOperator = createPreconditionerOperator( d_pOperator );
                d_preconditioner->registerOperator( pcOperator );
            }
        } else {
            // construct the preconditioner
            if ( d_db->keyExists( "pc_solver_name" ) ) {
                AMP_ASSERT( params->d_global_db );
                auto pc_solver_name = d_db->getString( "pc_solver_name" );
                auto global_db      = params->d_global_db;
                AMP_ASSERT( global_db->keyExists( pc_solver_name ) );
                auto pc_solver_db = global_db->getDatabase( pc_solver_name );
                auto pcSolverParameters =
                    std::make_shared<AMP::Solver::SolverStrategyParameters>( pc_solver_db );
                pcSolverParameters->d_global_db = params->d_global_db;
                if ( d_pOperator ) {
                    auto pcOperator                 = createPreconditionerOperator( d_pOperator );
                    pcSolverParameters->d_pOperator = pcOperator;
                }

                d_preconditioner = AMP::Solver::SolverFactory::create( pcSolverParameters );
            }
        }
    }

    int n = d_mvec + 1;

    d_h    = new T *[n];
    d_h[0] = new T[n * n];

    for ( int j = 1; j < n; j++ ) {
        d_h[j] = d_h[j - 1] + n;
    }

    d_next.resize( n );
    d_prev.resize( n );

    restart();
}

template<typename T>
NonlinearKrylovAccelerator<T>::~NonlinearKrylovAccelerator( void )
{
    if ( d_h != nullptr ) {
        delete[] d_h[0];
        delete[] d_h;
    }
}

template<typename T>
void NonlinearKrylovAccelerator<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    if ( db->keyExists( "max_vectors" ) ) {
        d_mvec = db->getScalar<int>( "max_vectors" );
    } else {
        AMP_ERROR( "NonlinearKrylovAccelerator -- Key data `max_vectors' missing in input." );
    }

    if ( db->keyExists( "angle_tolerance" ) ) {
        d_vtol = db->getScalar<T>( "angle_tolerance" );
    } else {
        AMP_ERROR( "NonlinearKrylovAccelerator -- Key data `angle_tolerance' missing in input." );
    }

    d_maximum_function_evals = db->getWithDefault<int>( "maximum_function_evals", 50 );

    if ( db->keyExists( "uses_preconditioner" ) ) {
        d_uses_preconditioner = db->getScalar<bool>( "uses_preconditioner" );

        if ( d_uses_preconditioner ) {
            if ( db->keyExists( "freeze_pc" ) ) {
                d_freeze_pc = db->getScalar<bool>( "freeze_pc" );
            }
        }
    }

    d_use_qr = db->getWithDefault<bool>( "use_qr", false );

    d_print_residuals  = db->getWithDefault<bool>( "print_residuals", false );
    d_use_damping      = db->getWithDefault<bool>( "use_damping", false );
    d_adaptive_damping = db->getWithDefault<bool>( "adaptive_damping", false );
    if ( d_adaptive_damping )
        d_use_damping = true;

    if ( d_use_damping ) {
        d_eta = db->getWithDefault<T>( "damping_factor", static_cast<T>( 1.0 ) );
    }

    AMP_ASSERT( d_mvec > 0 );
    AMP_ASSERT( d_vtol > static_cast<T>( 0.0 ) );
}

template<typename T>
void NonlinearKrylovAccelerator<T>::initialize(
    std::shared_ptr<const AMP::Solver::SolverStrategyParameters> params )
{
    // temporary if-then-else till a cleanup
    if ( params->d_vectors.size() > 0 ) {
        d_solution_vector = params->d_vectors[0]->clone();
        d_solution_vector->copyVector( params->d_vectors[0] );
    } else {
        d_solution_vector = params->d_pInitialGuess;
        AMP_INSIST( d_solution_vector, "Initial guess vector cannot be null" );
    }

    AMP_ASSERT( d_solution_vector );

    d_solution_vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    int n = d_mvec + 1;

    if ( !d_solver_initialized ) {

        d_v.resize( n );

        for ( int j = 0; j < n; j++ ) {
            d_v[j] = d_solution_vector->clone();
        }

        d_w.resize( n );

        for ( int j = 0; j < n; j++ ) {
            d_w[j] = d_solution_vector->clone();
        }

        d_residual_vector   = d_solution_vector->clone();
        d_correction_vector = d_solution_vector->clone();

        d_solver_initialized = true;
    }

    d_residual_vector->setToScalar( static_cast<T>( 0.0 ) );
    d_correction_vector->setToScalar( static_cast<T>( 0.0 ) );
}

template<typename T>
void NonlinearKrylovAccelerator<T>::reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> )
{
    // the preconditioner reset needs to be thought about more..
    if ( d_uses_preconditioner ) {
        AMP_ASSERT( d_preconditioner );
        d_preconditioner->reset( {} );
    }
    restart();
}

template<typename T>
void NonlinearKrylovAccelerator<T>::correction( std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    T s;
    std::shared_ptr<AMP::LinearAlgebra::Vector> w;

    d_current_correction++;

    /*
     *  UPDATE THE ACCELERATION SUBSPACE
     */

    if ( d_pending ) {

        /* next function difference w_1 */
        w = d_w[d_first];

        w->axpy( static_cast<T>( -1.0 ), *f, *w );

        s = static_cast<T>( w->L2Norm() );

        /* If the function difference is 0, we can't update the subspace with
           this data; so we toss it out and continue.  In this case it is likely
           that the outer iterative solution procedure has gone badly awry
           (unless the function value is itself 0), and we merely want to do
           something reasonable here and hope that situation is detected on the
           outside. */
        if ( s == static_cast<T>( 0.0 ) ) {
            AMP_WARNING( "current vector not valid!!, relax() being called " );
            relax();
        }
    }

    // The above if statement can go into relax and d_pending can become false,
    // hence the second if
    if ( d_pending ) {

        auto v = d_v[d_first];

        /* Normalize w_1 and apply same factor to v_1. */
        T const sinv = (T) 1.0 / s; // debug float to ScalarType issues later
        w->scale( sinv, *w );
        v->scale( sinv, *v );

        if ( !d_use_qr ) {

            /* Update H. */
            for ( int k = d_next[d_first]; k != EOL; k = d_next[k] ) {
                d_h[d_first][k] = static_cast<T>( w->dot( *d_w[k] ) );
            }

            /*
             *  CHOLESKI FACTORIZATION OF H = W^t W
             *  original matrix kept in the upper triangle (implicit unit diagonal)
             *  lower triangle holds the factorization
             */
            factorizeNormalMatrix();

        } else {

            AMP_ERROR( "QR factorization not implemented" );
        }

        // set the boolean to indicate we have a subspace
        d_subspace = true;
        d_pending  = false;
    }

    //  ACCELERATED CORRECTION

    // locate storage location for the new vector by finding the
    // first free location in the list
    AMP_ASSERT( d_free != EOL );
    int new_loc = d_free;
    // update the free pointer to point to the next free location
    d_free = d_next[d_free];

    // store f in the currently free location of w, so that we can
    // update it to the difference on the next iteration
    d_w[new_loc]->copyVector( f );

    if ( d_subspace ) {

        // create a row vector to store the solution components for
        // the correction vector
        std::vector<T> cv( d_mvec + 1, static_cast<T>( 0.0 ) );

        if ( !d_use_qr ) {
            cv = forwardbackwardSolve( f );
        } else {
            AMP_ERROR( "QR factorization not implemented" );
        }
        // at this point the solution to the minimization
        // problem has been computed and stored in cv(:)
        // we now compute the accelerated correction

        for ( int k = d_first; k != EOL; k = d_next[k] ) {
            f->axpy( cv[k], *d_v[k], *f );
            f->axpy( -cv[k], *d_w[k], *f );
        }

        if ( d_use_damping ) {
            auto eta = d_eta;
            if ( d_adaptive_damping ) {
                eta = static_cast<T>( 1.0 ) -
                      std::pow( static_cast<T>( 0.9 ),
                                static_cast<T>( std::min( d_current_correction, d_mvec ) ) );
            }

            // scale the residual vector
            f->scale( eta, *f );
        }
    }

#if 0
   // if no subspace exists this is the first correction
   // and it should include the damping
   if(d_use_damping && (d_current_correction==1))
      {
         auto eta = d_eta;
         if(d_adaptive_damping)
            {
	      eta = static_cast<T>(1.0)-std::pow(static_cast<T>(0.9), static_cast<T>(std::min(d_current_correction, d_mvec)));
            }
         
         // scale the residual vector
         f->scale(eta, f);
      }
#endif

    // save the correction, accelerated or otherwise for the next
    // call in the v matrix
    d_v[new_loc]->copyVector( f );

    // prepend the vector to the list so that it is at the front of
    // the list and is dropped last among the existing subspace
    // vectors
    d_prev[new_loc] = EOL;
    d_next[new_loc] = d_first;

    if ( d_first == EOL ) {
        d_last = new_loc;
    } else {
        d_prev[d_first] = new_loc;
    }

    d_first = new_loc;

    /* The original f and accelerated correction are cached for the next call. */
    d_pending = true;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                           std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "NonlinearKrylovAccelerator<T>::apply" );
    d_ConvergenceStatus = AMP::Solver::SolverStrategy::SolverStatus::DivergedOther;
    AMP_ASSERT( u.get() != nullptr );
    AMP_ASSERT( d_pOperator != nullptr );

    if ( d_uses_preconditioner ) {
        AMP_ASSERT( d_preconditioner );
        AMP_ASSERT( d_preconditioner->getOperator() );
    }

    d_iNumberIterations = 0;

    d_solution_vector->copyVector( u );
    d_solution_vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    AMP_ASSERT( d_pOperator.get() != nullptr );

    // compute residual
    d_pOperator->residual( f, d_solution_vector, d_residual_vector );
    d_function_apply_count++;

    d_residual_vector->scale( static_cast<T>( -1.0 ), *d_residual_vector );
    d_residual_vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    auto residual_norm = d_residual_vector->L2Norm();

    if ( d_print_residuals || ( d_iDebugPrintInfoLevel > 0 ) ) {
        AMP::printp( "Nonlinear Krylov iteration: %zu, residual: %0.12e\n",
                     d_iNumberIterations,
                     static_cast<double>( residual_norm ) );
    }

    const auto initial_residual_norm = residual_norm;

    std::shared_ptr<AMP::Operator::Operator> pc_operator;
    if ( d_uses_preconditioner ) {
        auto pc_parameters = d_pOperator->getParameters( "Jacobian", d_solution_vector );
        AMP_ASSERT( pc_parameters.get() != nullptr );
        pc_operator = d_preconditioner->getOperator();
        AMP_ASSERT( pc_operator.get() != nullptr );

        // if using a frozen preconditioner set it up first
        if ( d_freeze_pc ) {
            pc_operator->reset( pc_parameters );
        }
    }

    bool converged = ( residual_norm < d_dAbsoluteTolerance );
    if ( converged )
        d_ConvergenceStatus = AMP::Solver::SolverStrategy::SolverStatus::ConvergedOnAbsTol;

    while ( ( d_iNumberIterations < d_iMaxIterations ) && ( !converged ) ) {
        if ( d_uses_preconditioner ) {
            if ( !d_freeze_pc ) {
                auto pc_parameters = d_pOperator->getParameters( "Jacobian", d_solution_vector );
                AMP_ASSERT( pc_parameters );

                pc_operator->reset( pc_parameters );
                d_preconditioner->reset( {} ); // PC's like MG might regenerate matrices/coeffs
            }

            AMP_ASSERT( d_preconditioner->getOperator() );

            auto solnScaling = this->getSolutionScaling();
            auto funcScaling = this->getFunctionScaling();
            if ( solnScaling )
                d_correction_vector->multiply( *d_correction_vector, *solnScaling );
            if ( funcScaling )
                d_residual_vector->multiply( *d_residual_vector, *funcScaling );

            // apply the preconditioner
            d_preconditioner->apply( d_residual_vector, d_correction_vector );

            if ( solnScaling )
                d_correction_vector->divide( *d_correction_vector, *solnScaling );
            if ( funcScaling )
                d_residual_vector->divide( *d_residual_vector, *funcScaling );

            d_preconditioner_apply_count++;

        } else {
            // identity preconditioning
            d_correction_vector->copyVector( d_residual_vector );
        }

        // compute AIN correction
        this->correction( d_correction_vector );

        // correct current solution
        d_solution_vector->axpy( static_cast<T>( -1.0 ), *d_correction_vector, *d_solution_vector );
        d_solution_vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        // compute the residual
        d_pOperator->residual( f, d_solution_vector, d_residual_vector );
        d_function_apply_count++;

        d_residual_vector->scale( static_cast<T>( -1.0 ), *d_residual_vector );
        d_residual_vector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

        // auto prev_residual_norm = residual_norm;
        residual_norm = d_residual_vector->L2Norm();

        d_iNumberIterations++;

        if ( d_print_residuals || ( d_iDebugPrintInfoLevel > 0 ) ) {
            AMP::printp( "Nonlinear Krylov iteration: %zu, residual: %0.12e\n",
                         d_iNumberIterations,
                         static_cast<double>( residual_norm ) );
        }

        converged = ( residual_norm < d_dAbsoluteTolerance ) ||
                    ( residual_norm < d_dRelativeTolerance * initial_residual_norm );
        if ( converged ) {
            d_ConvergenceStatus = ( residual_norm < d_dAbsoluteTolerance ) ?
                                      AMP::Solver::SolverStrategy::SolverStatus::ConvergedOnAbsTol :
                                      AMP::Solver::SolverStrategy::SolverStatus::ConvergedOnRelTol;
        }
    }

    d_iterationHistory.push_back( d_iNumberIterations );

    if ( !converged && ( d_iDebugPrintInfoLevel > 0 ) ) {
        AMP_WARNING( "NKA::solve did not converge to tolerance" );
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "NonlinearKrylovAccelerator convergence status "
                  << static_cast<int>( d_ConvergenceStatus ) << std::endl;
    }

    u->copyVector( d_solution_vector );
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

template<typename T>
void NonlinearKrylovAccelerator<T>::restart( void )
{
    d_current_correction = 0;

    /* No vectors are stored. */
    d_first    = EOL;
    d_last     = EOL;
    d_subspace = false;
    d_pending  = false;

    /* Initialize the free storage linked list. */
    d_free = 0;

    for ( int k = 0; k < d_mvec; ++k ) {
        d_next[k] = k + 1;
    }

    d_next[d_mvec] = EOL;

    // reset the number of levels for all the vectors
    if ( d_solution_vector ) {

        d_solution_vector->getVectorData()->reset();
        d_residual_vector->getVectorData()->reset();
        d_correction_vector->getVectorData()->reset();

        for ( int k = 0; k < d_mvec + 1; ++k ) {
            if ( d_v[k] ) {
                d_v[k]->getVectorData()->reset();
            }

            if ( d_w[k] ) {
                d_w[k]->getVectorData()->reset();
            }
        }
    }

    d_iNumberIterations = 0;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::relax( void )
{
    if ( d_pending ) {
        /* Drop the initial slot where the pending vectors are stored. */
        AMP_ASSERT( d_first >= 0 );
        int new_loc = d_first;
        d_first     = d_next[d_first];
        if ( d_first == EOL ) {
            d_last = EOL;
        } else {
            d_prev[d_first] = EOL;
        }

        /* Update the free storage list. */
        d_next[new_loc] = d_free;
        d_free          = new_loc;
        d_pending       = false;
    }
}

template<typename T>
void NonlinearKrylovAccelerator<T>::factorizeNormalMatrix( void )
{
    // Solve the least squares problem using a Cholesky
    // factorization, dropping any vectors that
    // render the system nearly rank deficient
    // we'll first follow Carlson's implementation

    // start the factorization at the entry indexed by first

    // Trivial initial factorization stage
    int nvec              = 1;
    d_h[d_first][d_first] = static_cast<T>( 1.0 );

    for ( int k = d_next[d_first]; k != EOL; k = d_next[k] ) {
        ++nvec;

        // maintain atmost mvec vectors, if we've reached the
        // limit throw out the last vector in the subspace and
        // update the subspace list and free list to reflect
        // this
        if ( nvec > d_mvec ) {
            AMP_ASSERT( d_last == k );
            d_next[d_last] = d_free;
            d_free         = k;
            d_last         = d_prev[k];
            d_next[d_last] = EOL;
            break;
        }

        // Single stage of Choleski factorization

        auto *hk = d_h[k]; // row k of H
        T hkk    = static_cast<T>( 1.0 );
        for ( int j = d_first; j != k; j = d_next[j] ) {
            auto *hj = d_h[j]; // row j of H
            T hkj    = hj[k];
            for ( int i = d_first; i != j; i = d_next[i] ) {
                hkj -= hk[i] * hj[i];
            }
            hkj /= hj[j];
            hk[j] = hkj;
            hkk -= hkj * hkj;
        }

        if ( hkk > d_vtol * d_vtol ) {
            hk[k] = std::sqrt( hkk );
        } else {
            // The current w nearly lies in the span of the previous vectors so drop
            // it
            AMP_ASSERT( d_prev[k] != EOL );
            d_next[d_prev[k]] = d_next[k];
            if ( d_next[k] == EOL )
                d_last = d_prev[k];
            else
                d_prev[d_next[k]] = d_prev[k];
            // update the free storage list
            d_next[k] = d_free;
            d_free    = k;
            // back-up and move on to the next vector
            k = d_prev[k];
            nvec--;
        }
    }
}

template<typename T>
std::vector<T>
NonlinearKrylovAccelerator<T>::forwardbackwardSolve( std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    std::vector<T> cv( d_mvec + 1, static_cast<T>( 0.0 ) );

    /* Project f onto the span of the w vectors: */
    /* forward substitution */
    for ( int j = d_first; j != EOL; j = d_next[j] ) {
        auto cj = static_cast<T>( f->dot( *d_w[j] ) );

        for ( int i = d_first; i != j; i = d_next[i] ) {
            cj -= d_h[j][i] * cv[i];
        }

        cv[j] = cj / d_h[j][j];
    }

    /* backward substitution */
    for ( int j = d_last; j != EOL; j = d_prev[j] ) {
        auto cj = cv[j];
        for ( int i = d_last; i != j; i = d_prev[i] ) {
            cj -= d_h[i][j] * cv[i];
        }
        cv[j] = cj / d_h[j][j];
    }

    return cv;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::setNestedSolver(
    std::shared_ptr<AMP::Solver::SolverStrategy> pc )
{
    d_preconditioner = pc;
}

/*
*************************************************************************
*                                                                       *
*  Access functions for parameters that control solver behavior.        *
*                                                                       *
*************************************************************************
*/
template<typename T>
int NonlinearKrylovAccelerator<T>::getMaxNonlinearIterations() const
{
    return ( d_iMaxIterations );
}

template<typename T>
void NonlinearKrylovAccelerator<T>::setMaxNonlinearIterations( int max_nli )
{
    d_iMaxIterations = max_nli;
}

template<typename T>
int NonlinearKrylovAccelerator<T>::getMaxFunctionEvaluations() const
{
    return d_maximum_function_evals;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::setMaxFunctionEvaluations( int max_feval )
{
    d_maximum_function_evals = max_feval;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::printStatistics( std::ostream &os )
{
    int total_nnl_iters = 0;

    for ( int n : d_iterationHistory )
        total_nnl_iters += n;

    os << "Total number of NKA nonlinear iterations       : " << total_nnl_iters << std::endl;
    os << "Total number of NKA preconditioner iterations  : " << d_preconditioner_apply_count
       << std::endl;
    os << "Total number of function evaluations           : " << d_function_apply_count
       << std::endl;
    os << "Average number of nonlinear iterations         : "
       << ( (T) total_nnl_iters ) / ( (T) d_iterationHistory.size() ) << std::endl;
}

template<typename T>
void NonlinearKrylovAccelerator<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    AMP_ASSERT( op );
    d_pOperator = op;
    if ( d_uses_preconditioner ) {
        AMP_ASSERT( d_preconditioner );
        auto pc_operator = createPreconditionerOperator( op );
        d_preconditioner->registerOperator( pc_operator );
    }
}

template<typename T>
std::shared_ptr<AMP::Operator::Operator>
NonlinearKrylovAccelerator<T>::createPreconditionerOperator(
    std::shared_ptr<AMP::Operator::Operator> op )
{
    AMP_ASSERT( op );
    auto pc_params = op->getParameters( "Jacobian", d_solution_vector );
    return AMP::Operator::OperatorFactory::create( pc_params );
}
} // namespace AMP::Solver
