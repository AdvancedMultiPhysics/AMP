#ifndef included_AMP_TFQMRSolver
#define included_AMP_TFQMRSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::Solver {

/**
 * The TFQMRSolver class implements the TFQMR method for non-symmetric linear systems
 * introduced by R. W. Freund
 *    Freund, R. W., "A Transpose-Free Quasi-Minimal Residual Algorithm for Non-Hermitian
 *    Linear Systems", SIAM J. Sci. and Stat. Comput. 14 (2): 490-482, (1993).
 * The implementation here is mostly based on the MATLAB code by C. T. Kelley
 * http://www4.ncsu.edu/~ctk/roots/tfqmr.m
 * If a preconditioner is provided right preconditioning is done
 */

template<typename T = double>
class TFQMRSolver : public SolverStrategy
{
public:
    /**
     * default constructor
     */
    TFQMRSolver() = default;

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

     3. type: string, name : pc_side, default value "RIGHT",
     acceptable values ("RIGHT", "LEFT", "SYMMETRIC" )
         active only when uses_preconditioner set to true
     */
    explicit TFQMRSolver( std::shared_ptr<SolverStrategyParameters> params );

    /**
     * static create routine that is used by SolverFactory
     @param [in] params The parameters object
     contains a database objects with the fields listed for the constructor above
     */
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<TFQMRSolver<T>>( params );
    }

    /**
     * Default destructor
     */
    virtual ~TFQMRSolver() = default;

    std::string type() const override { return "TFQMRSolver"; }

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the TFQMRSolver. Should not be necessary for the user to call in general.
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
    void getFromInput( std::shared_ptr<const AMP::Database> db );

private:
    bool d_bUsesPreconditioner = false;

    std::string d_preconditioner_side;

    std::shared_ptr<AMP::Solver::SolverStrategy> d_pPreconditioner;
};
} // namespace AMP::Solver

#endif
