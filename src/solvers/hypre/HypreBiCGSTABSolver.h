#ifndef included_AMP_HypreBiCGSTABSolver
#define included_AMP_HypreBiCGSTABSolver


#include "AMP/matrices/Matrix.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/hypre/HypreSolver.h"

namespace AMP::LinearAlgebra {
class HypreMatrixAdaptor;
}

namespace AMP::Solver {


using HypreBiCGSTABSolverParameters = SolverStrategyParameters;


/**
 * The HypreBiCGSTABSolver is a wrapper to the HYPRE BiCGSTAB solver.
 * The wrapper at present simply provides an adapter to enable AMP
 * users to use the black box Hypre BiCGSTAB solver.
 */

class HypreBiCGSTABSolver final : public HypreSolver
{

public:
    /**
     * Default constructor
     */
    HypreBiCGSTABSolver();

    /**
     * Main constructor.
     @param [in] parameters The parameters object contains a database object which must contain the
     following fields in addition to the fields expected by the base class SolverStrategy class:

    */
    explicit HypreBiCGSTABSolver( std::shared_ptr<HypreBiCGSTABSolverParameters> parameters );

    /**
     * Default destructor
     */
    virtual ~HypreBiCGSTABSolver();

    std::string type() const override { return "HypreBiCGSTABSolver"; }

    //! static create routine that is used by SolverFactory
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<HypreBiCGSTABSolver>( solverStrategyParameters );
    }

    /**
     * Solve the system \f$Au = f\f$.
     @param [in] f : shared pointer to right hand side vector
     @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the solution vector and potentially create internal vectors needed for solution
     @param [in] parameters The parameters object
     contains a database object. Refer to the documentation for the constructor to see what fields
     are required.
     This routine assumes that a non-NULL operator of type LinearOperator has been registered with
     the solver.
     The LinearOperator currently is assumed to contain a pointer to an EpetraMatrix object.
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> parameters ) override;

    void getFromInput( std::shared_ptr<const AMP::Database> db );

    void reset( std::shared_ptr<SolverStrategyParameters> params ) override;

private:
    void setupHypreSolver( std::shared_ptr<const SolverStrategyParameters> parameters );

    bool d_bUsesPreconditioner = false;
    bool d_bDiagScalePC        = false; //! use diagonal scaled preconditioner
};
} // namespace AMP::Solver

#endif
