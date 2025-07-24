#ifndef included_AMP_QMRCGSTABSolver
#define included_AMP_QMRCGSTABSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::Solver {

/**
 * The QMRCGSTABSolver class implements the QMRCGSTAB method for non-symmetric linear systems
 * introduced by Chan et. al.
 *
 * The implementation here is mostly based on the MATLAB code at
 * https://link.springer.com/content/pdf/bbm%3A978-3-8348-8100-7%2F1.pdf
 * Currently no preconditioning
 */

template<typename T = double>
class QMRCGSTABSolver : public SolverStrategy
{
public:
    /**
     * default constructor
     */
    QMRCGSTABSolver() = default;

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
    explicit QMRCGSTABSolver( std::shared_ptr<SolverStrategyParameters> params );

    /**
     * static create routine that is used by SolverFactory
     @param [in] params The parameters object
     contains a database objects with the fields listed for the constructor above
     */
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<QMRCGSTABSolver<T>>( params );
    }

    /**
     * Default destructor
     */
    virtual ~QMRCGSTABSolver() = default;

    std::string type() const override { return "QMRCGSTABSolver"; }

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the QMRCGSTABSolver. Should not be necessary for the user to call in general.
     * @param params
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> params ) override;

protected:
    void getFromInput( std::shared_ptr<const AMP::Database> db );

private:
    bool d_bUsesPreconditioner = false;

    std::string d_preconditioner_side;
};
} // namespace AMP::Solver

#endif
