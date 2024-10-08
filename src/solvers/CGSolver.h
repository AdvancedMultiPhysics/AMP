#ifndef included_AMP_CGSolver
#define included_AMP_CGSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::Solver {

/**
 * The CGSolver class implements the Conjugate Gradient method
 */

template<typename T = double>
class CGSolver : public SolverStrategy
{
public:
    /**
     * default constructor
     */
    CGSolver() = default;

    /**
     * main constructor
     @param [in] params The parameters object
     contains a database objects containing the following fields:

     1. type: double, name : relative_tolerance, default value of $1.0e-9$, relative tolerance for
    CG solver
    acceptable values (non-negative real values)

     2. type: bool, name : uses_preconditioner, default value false
        acceptable values (false, true),
        side effect: if false sets string pc_type to "none"
     */
    explicit CGSolver( std::shared_ptr<SolverStrategyParameters> params );

    /**
     * static create routine that is used by SolverFactory
     @param [in] params The parameters object
     contains a database objects with the fields listed for the constructor above
     */
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<CGSolver<T>>( params );
    }

    /**
     * Default destructor
     */
    virtual ~CGSolver() = default;

    std::string type() const override { return "CGSolver"; }

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the CGSolver. Should not be necessary for the user to call in general.
     * @param params
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> params ) override;

    /**
     * sets a shared pointer to a preconditioner object. The preconditioner is derived from
     * a SolverStrategy class
     * @param pc shared pointer to preconditioner
     */
    inline void setNestedSolver( std::shared_ptr<AMP::Solver::SolverStrategy> pc ) override
    {
        d_pPreconditioner = pc;
    }

    inline std::shared_ptr<AMP::Solver::SolverStrategy> getNestedSolver() override
    {
        return d_pPreconditioner;
    }

    /**
     * Resets the registered operator internally with new parameters if necessary
     * @param params    OperatorParameters object that is NULL by default
     */
    void resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );

private:
    T d_dDivergenceTolerance = 1e3;

    bool d_bUsesPreconditioner = false;

    std::shared_ptr<AMP::Solver::SolverStrategy> d_pPreconditioner;
};
} // namespace AMP::Solver

#endif
