#ifndef included_AMP_SolverStrategyParameters
#define included_AMP_SolverStrategyParameters

#include "AMP/operators/Operator.h"
#include "AMP/utils/ParameterBase.h"
#include <memory>


namespace AMP::Solver {

class SolverStrategy;

/**\class SolverStrategyParameters
 *
 * SolverStrategyParameters encapsulates parameters used to initialize
 * SolverStrategy objects
 */

class SolverStrategyParameters : public ParameterBase
{
public:
    /**
     * Empty constructor.
     */
    SolverStrategyParameters();

    /**
     * Construct and initialize a parameter list according to input
     * data.  Guess what the required and optional keywords are.
     */
    explicit SolverStrategyParameters( std::shared_ptr<AMP::Database> db );

    /**
     * Destructor.
     */
    virtual ~SolverStrategyParameters();

    AMP_MPI d_comm;

    std::shared_ptr<AMP::Operator::Operator> d_pOperator = nullptr;

    /**
     * Pointer to nested solver, e.g. Krylov for Newton, or preconditioner, can be null
     */
    std::shared_ptr<AMP::Solver::SolverStrategy> d_pNestedSolver = nullptr;

    //! initial guess for solver -- probably can go away in favour of d_vectors
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pInitialGuess;

    /**
     * List of vectors to be used during solver initialization
     */
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_vectors;

    /** Pointer to global database
     *  This is temporary fix and eventually either d_global_db or d_db should go away
     *  This is introduced to allow for solver factories to access databases in the global
     *  database for the construction of nested solvers
     */
    std::shared_ptr<AMP::Database> d_global_db;

protected:
private:
};
} // namespace AMP::Solver

#endif
