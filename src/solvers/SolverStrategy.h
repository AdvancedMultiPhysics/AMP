#ifndef included_AMP_SolverStrategy
#define included_AMP_SolverStrategy

#include "AMP/operators/Operator.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/vectors/Vector.h"

#include <memory>


// Declare some classes
namespace AMP::IO {
class Writer;
} // namespace AMP::IO


namespace AMP::Solver {

/**
 * Class SolverStrategy is a base class for methods to solve
 * equations of the form \f$A(u) = f\f$. $A$ may be a nonlinear
 * or linear operator.
 */

class SolverStrategy
{
public:
    typedef std::shared_ptr<AMP::Solver::SolverStrategy> shared_ptr;

    /**
     * Default constructor
     */
    SolverStrategy();

    /**
     *  Main constructor for the base class.
     *  @param[in] parameters   The parameters object contains a database object which must contain
     * the
     *                          following fields:
     *                          1. type: integer, name: max_iterations (required)
     *                             acceptable values (non-negative integer values)
     *                          2. type: double, name: max_error, (required)
     *                             acceptable values (non-negative real values)
     *                          3. type: integer, name: print_info_level, default value: 0,
     *                             acceptable values (non negative integer values, the higher
     *                             the value the more verbose the debugging information provided)
     *                          4. type: bool, name: zero_initial_guess, default value: false,
     *                             acceptable values (TRUE, FALSE)
     */
    explicit SolverStrategy( std::shared_ptr<const SolverStrategyParameters> parameters );

    /**
     * Default destructor. Currently does not do anything.
     */
    virtual ~SolverStrategy();

    //! Return the name of the solver
    virtual std::string type() const = 0;

    enum class SolverStatus {
        ConvergedOnAbsTol,
        ConvergedOnRelTol,
        ConvergedUserCondition,
        MaxIterations,
        DivergedLineSearch,
        DivergedStepSize,
        DivergedFunctionCount,
        DivergedOnNan,
        DivergedNestedSolver,
        DivergedOther
    };

    static std::string statusToString( SolverStatus status )
    {
        switch ( status ) {
        case SolverStatus::ConvergedOnAbsTol:
            return "ConvergedOnAbsTol";
        case SolverStatus::ConvergedOnRelTol:
            return "ConvergedOnRelTol";
        case SolverStatus::ConvergedUserCondition:
            return "ConvergedUserCondition";
        case SolverStatus::MaxIterations:
            return "MaxIterations";
        case SolverStatus::DivergedLineSearch:
            return "DivergedLineSearch";
        case SolverStatus::DivergedStepSize:
            return "DivergedStepSize";
        case SolverStatus::DivergedFunctionCount:
            return "DivergedFunctionCount";
        case SolverStatus::DivergedOnNan:
            return "DivergedOnNan";
        case SolverStatus::DivergedNestedSolver:
            return "DivergedNestedSolver";
        default:
            return "DivergedOther";
        }
    }

    /**
     * Solve the system \f$A(u) = f\f$.  This is a pure virtual function that the derived classes
     * need to provide an implementation of.
     * @param[in]  f    shared pointer to right hand side vector
     * @param[out] u    shared pointer to approximate computed solution
     */
    virtual void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> u ) = 0;

    /**
     * Initialize the solution vector and potentially create internal vectors needed for solution
     * @param[in] parameters    The parameters object contains a database object.
     *                          Currently there are no required fields for the database object.
     */
    virtual void initialize( std::shared_ptr<const SolverStrategyParameters> parameters );

    /**
     * Provide the initial guess for the solver. This is a pure virtual function that the derived
     * classes
     * need to provide an implementation of.
     * @param[in] initialGuess: shared pointer to the initial guess vector.
     */
    virtual void setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess );

    /**
     * Specify level of diagnostic information printed during iterations.
     * @param[in] print_level    integer level value with permissible values 0 and higher. Setting
     *                           to zero should provide minimial debugging information with higher
     *                           values resulting in increasingly verbose information being printed
     * out.
     */
    virtual void setDebugPrintInfoLevel( int print_level ) { d_iDebugPrintInfoLevel = print_level; }

    /**
     * Get level of diagnostic information printed during iterations.
     */
    int getDebugPrintInfoLevel( void ) { return d_iDebugPrintInfoLevel; }

    /**
     * Return the number of iterations taken by the solver to converge.
     */
    virtual int getIterations( void ) const { return ( d_iNumberIterations ); }

    /**
     * Tells the solver to use an initial guess of zero and not try to
     * copy an initial guess into the solution vector
     * @param[in] use_zero_guess    boolean to specify whether zero initial guess should be used or
     * not.
     */
    virtual void setZeroInitialGuess( bool use_zero_guess )
    {
        d_bUseZeroInitialGuess = use_zero_guess;
    }

    /**
     * Set a nested solver, eg, Krylov for Newton, preconditioner for Krylov etc. Null op in base
     * class
     */
    virtual void setNestedSolver( std::shared_ptr<SolverStrategy> solver )
    {
        d_pNestedSolver = solver;
    }

    /**
     * Return a nested solver (eg preconditioner) if it exists. By default return a nullptr
     */

    virtual std::shared_ptr<SolverStrategy> getNestedSolver( void ) { return d_pNestedSolver; }

    /**
     * Register the operator that the solver will use during solves
     * @param [in] op shared pointer to operator \f$A()\f$ for equation \f$A(u) = f\f$
     */
    virtual void registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
    {
        d_pOperator = op;
    }

    /**
     * \brief  Registers a writer with the solver
     * \details  This function will register a writer with the solver.  The solver
     *  may then register any vector components it "owns" with the writer.
     * \param writer   The writer to register
     */
    virtual void registerWriter( std::shared_ptr<AMP::IO::Writer> writer ) { d_writer = writer; }

    /**
     * Resets the operator registered with the solver with new parameters if necessary
     * @param parameters
     *        OperatorParameters object that is NULL by default
     */
    virtual void
    resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> parameters );

    /**
     * Resets the solver internally with new parameters if necessary
     * @param parameters
     *        SolverStrategyParameters object that is NULL by default
     */
    virtual void reset( std::shared_ptr<SolverStrategyParameters> parameters );

    /**
     * Return a shared pointer to the operator registered with the solver.
     */
    virtual std::shared_ptr<AMP::Operator::Operator> getOperator( void ) { return d_pOperator; }

    AMP::Scalar getAbsoluteTolerance() const { return ( d_dAbsoluteTolerance ); }

    virtual void setAbsoluteTolerance( AMP::Scalar abs_tol ) { d_dAbsoluteTolerance = abs_tol; }

    AMP::Scalar getRelativeTolerance() const { return ( d_dRelativeTolerance ); }

    virtual void setRelativeTolerance( AMP::Scalar rel_tol ) { d_dRelativeTolerance = rel_tol; }

    virtual void setMaxIterations( const int max_iterations ) { d_iMaxIterations = max_iterations; }

    int getMaxIterations( void ) const { return d_iMaxIterations; }

    virtual void printStatistics( std::ostream &os = AMP::pout )
    {
        os << "Not implemented for this solver!" << std::endl;
    }

    bool getConverged( void ) const
    {
        return d_ConvergenceStatus <= SolverStatus::ConvergedUserCondition ? 1 : 0;
    }

    SolverStatus getConvergenceStatus( void ) const { return d_ConvergenceStatus; }

    std::string getConvergenceStatusString( void ) const
    {
        return statusToString( d_ConvergenceStatus );
    }

    virtual void print( std::ostream & ) {}

    virtual AMP::Scalar getResidualNorm( void ) const { return d_dResidualNorm; }

    virtual AMP::Scalar getInitialResidual( void ) const { return d_dInitialResidual; }

    virtual const std::vector<int> &getIterationHistory( void ) { return d_iterationHistory; }

    int getTotalNumberOfIterations( void );

    virtual void residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                           std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                           std::shared_ptr<AMP::LinearAlgebra::Vector> r );

    virtual void printConvergenceStatus( SolverStrategy::SolverStatus,
                                         std::ostream & = AMP::pout ) const
    {
    }

    //! for multiphysics problems it may be necessary to scale the solution
    // and nonlinear function for correct solution.
    // The first vector is for solution scaling, the second for function
    void setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> f )
    {
        d_pSolutionScaling = s;
        d_pFunctionScaling = f;
    }

    std::shared_ptr<AMP::LinearAlgebra::Vector> getSolutionScaling() { return d_pSolutionScaling; }
    std::shared_ptr<AMP::LinearAlgebra::Vector> getFunctionScaling() { return d_pFunctionScaling; }

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );
    virtual bool checkStoppingCriteria( AMP::Scalar res_norm, bool check_iters = true );

    SolverStatus d_ConvergenceStatus = SolverStatus::DivergedOther;

    std::string d_sName;

    int d_iNumberIterations = 0; // iterations in solver

    int d_iMaxIterations = 0;

    AMP::Scalar d_dResidualNorm;
    AMP::Scalar d_dInitialResidual;

    AMP::Scalar d_dAbsoluteTolerance = 1.0e-14;
    AMP::Scalar d_dRelativeTolerance = 1.0e-09;

    int d_iDebugPrintInfoLevel = 0;

    bool d_bUseZeroInitialGuess = true;

    bool d_bComputeResidual = false;

    int d_iObjectId;

    static int d_iInstanceId; // used to differentiate between different instances of the class

    //! keeps track of iteration statistics over solver lifetime
    std::vector<int> d_iterationHistory;

    std::shared_ptr<AMP::Database> d_db = nullptr;

    /** Pointer to global database
     *  This is temporary fix and eventually either d_global_db or d_db should go away
     *  This is introduced to allow for solver factories to access databases in the global
     *  database for the construction of nested solvers
     */
    std::shared_ptr<AMP::Database> d_global_db = nullptr;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSolutionScaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pFunctionScaling;

    std::shared_ptr<AMP::Operator::Operator> d_pOperator = nullptr;

    //! nested solver used by this solver
    std::shared_ptr<AMP::Solver::SolverStrategy> d_pNestedSolver = nullptr;

    std::shared_ptr<AMP::IO::Writer> d_writer = nullptr;


private:
};

} // namespace AMP::Solver

#endif
