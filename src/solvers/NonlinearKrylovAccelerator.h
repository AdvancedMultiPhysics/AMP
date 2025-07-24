#ifndef included_NonlinearKrylovAccelerator
#define included_NonlinearKrylovAccelerator

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP {
class Database;
namespace Solver {
class SolverStrategyParameters;
}
} // namespace AMP

namespace AMP::Solver {

template<typename T = double>
class NonlinearKrylovAccelerator : public AMP::Solver::SolverStrategy
{

public:
    explicit NonlinearKrylovAccelerator(
        std::shared_ptr<AMP::Solver::SolverStrategyParameters> params );

    ~NonlinearKrylovAccelerator();

    static std::unique_ptr<AMP::Solver::SolverStrategy>
    createSolver( std::shared_ptr<AMP::Solver::SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<NonlinearKrylovAccelerator<T>>( solverStrategyParameters );
    }

    std::string type() const override { return "NKASolver"; }

    void
    initialize( std::shared_ptr<const AMP::Solver::SolverStrategyParameters> parameters ) override;

    /**
     * Resets the solver internally with new parameters if necessary
     * @param parameters
     *        SolverStrategyParameters object that is NULL by default
     */
    void reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters ) override;

    void correction( std::shared_ptr<AMP::LinearAlgebra::Vector> f );

    void restart( void );

    void relax( void );

    /*!
     *  Get maximum iterations for nonlinear solver.
     */
    int getMaxNonlinearIterations() const;

    /*!
     *  Set maximum iterations for nonlinear solver.
     */
    void setMaxNonlinearIterations( int max_nli );

    /*!
     *  Get maximum function evaluations by nonlinear solver.
     */
    int getMaxFunctionEvaluations() const;

    /*!
     *  Set maximum function evaluations in nonlinear solver.
     */
    void setMaxFunctionEvaluations( int max_feval );

    /**
     * Solve the  system \f$A(u) = f\f$.  The solution \f$u\f$ and the
     * right hand side \f$f\f$ are specified via vectors on the patch
     * hierarchy.  This function returns zero if the solver converged within
     * the specified tolerances and nonzero otherwise.  Member functions
     * getIterations() and getRelativeNorm() return the iterations and
     * relative norm, respectively, from the solver.
     * \param f
     *            vector for right hand side, can be nullptr
     * \param u
     *            vector for solution variables
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    void printStatistics( std::ostream &os ) override;

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override final;

private:
    NonlinearKrylovAccelerator()                                     = delete;
    NonlinearKrylovAccelerator( const NonlinearKrylovAccelerator & ) = delete;
    NonlinearKrylovAccelerator( NonlinearKrylovAccelerator && )      = delete;
    NonlinearKrylovAccelerator &operator=( const NonlinearKrylovAccelerator & ) = delete;


    void getFromInput( std::shared_ptr<AMP::Database> db );

    void initialize( std::shared_ptr<AMP::LinearAlgebra::Vector> solution_vector );

    std::shared_ptr<AMP::Operator::Operator>
    createPreconditionerOperator( std::shared_ptr<AMP::Operator::Operator> op );

    void factorizeNormalMatrix( void );

    //! forward backward solve for Cholesky
    std::vector<T> forwardbackwardSolve( std::shared_ptr<AMP::LinearAlgebra::Vector> f );

    bool d_subspace = false; //! boolean: a nonempty subspace
    bool d_pending  = false; //! contains pending vectors -- boolean
    bool d_use_qr   = false; //! use qr factorization to solve the least squares problem

    int d_mvec = 0;   //! maximum number of subspace vectors
    T d_vtol   = 0.0; //! vector drop tolerance

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_solution_vector;   //! correction vectors
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_residual_vector;   //! correction vectors
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_correction_vector; //! correction vectors

    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_v; //! correction vectors
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_w; //! function difference vectors
    T **d_h; //! matrix of w vector inner products

    /* Linked-list organization of the vector storage. */
    int d_first = 0;         //! index of first subspace vector
    int d_last  = 0;         //! index of last subspace vector
    int d_free  = 0;         //! index of the initial vector in free storage linked list
    std::vector<int> d_next; //! next index link field
    std::vector<int> d_prev; //! previous index link field in doubly-linked subspace v

    int d_maximum_function_evals = 0;
    int d_current_correction     = 0; //! current number of corrections, updated each time
                                      //! correction is called
    int d_preconditioner_apply_count =
        0;                          //! keeps track of the number of preconditioner applications
    int d_function_apply_count = 0; //! keeps track of the number of function applications

    T d_eta = 1.0; //! damping factor

    bool d_uses_preconditioner = false; //! whether the solver uses a preconditioner or not
    bool d_freeze_pc       = true;  //! flag determining whether the preconditioner is frozen or not
    bool d_print_residuals = false; //! whether to print the residuals
    bool d_solver_initialized = false;
    bool d_use_damping        = false; //! whether to use damping
    bool d_adaptive_damping   = false; //! use adaptive damping
};
} // namespace AMP::Solver

#endif
