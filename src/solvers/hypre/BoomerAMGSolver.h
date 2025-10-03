#ifndef included_AMP_BoomerAMGSolver
#define included_AMP_BoomerAMGSolver


#include "AMP/matrices/Matrix.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/hypre/HypreSolver.h"

namespace AMP::LinearAlgebra {
class HypreMatrixAdaptor;
}

namespace AMP::Solver {


using BoomerAMGSolverParameters = SolverStrategyParameters;


/**
 * The BoomerAMGSolver is a wrapper to the HYPRE BoomerAMG solver.
 * BoomerAMG provides implementations of various algebraic multigrid methods.
 * The wrapper at present simply provides an adapter to enable AMP
 * users to use the black box BoomerAMG preconditioner.
 */

class BoomerAMGSolver final : public HypreSolver
{

public:
    /**
     * Default constructor
     */
    BoomerAMGSolver();

    /**
     * Main constructor.
     @param [in] parameters The parameters object contains a database object which must contain the
     following fields in addition to the fields expected by the base class SolverStrategy class:

    */
    explicit BoomerAMGSolver( std::shared_ptr<BoomerAMGSolverParameters> parameters );

    /**
     * Default destructor
     */
    virtual ~BoomerAMGSolver();

    std::string type() const override { return "BoomerAMGSolver"; }

    //! static create routine that is used by SolverFactory
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<BoomerAMGSolver>( solverStrategyParameters );
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

    /**
     * Resets the solver internally with new parameters if necessary
     * @param [in] params
     *        SolverStrategyParameters object that is NULL by default
     * Currently every call to reset destroys the HyprePCG solver object
     * and recreates it based on the parameters object. See constructor for
     * fields required for parameter object.
     */
    void reset( std::shared_ptr<SolverStrategyParameters> ) override;

    void getFromInput( std::shared_ptr<const AMP::Database> db );

private:
    void setupBoomerAMG( void );
    void setHypreFunctionPointers();

    int d_num_functions        = 1;
    int d_min_iterations       = 0;
    int d_max_coarse_size      = 800;
    int d_min_coarse_size      = 100;
    int d_max_levels           = 10;
    int d_coarsen_type         = 8;
    int d_measure_type         = 0;
    int d_agg_num_levels       = 0;
    int d_num_paths            = 1;
    int d_cgc_iterations       = 1;
    int d_nodal                = 0;
    int d_nodal_diag           = 0;
    int d_interp_type          = 18;
    int d_P_max_elements       = 4;
    int d_separate_weights     = 0;
    int d_agg_interp_type      = 5;
    int d_agg_P_max_elements   = 0;
    int d_agg_P12_max_elements = 0;
    int d_number_samples       = 0;
    int d_cycle_type           = 1;
    int d_additive_level       = -1;
    int d_mult_additive_level  = -1;
    int d_simple_level         = -1;
    int d_add_P_max_elmts      = 0;
    int d_number_sweeps        = 1;
    int d_relax_type           = 16;
    int d_relax_order          = 0;
    int d_chebyshev_order      = 2;
    int d_smooth_type          = 6;
    int d_smooth_number_levels = 0;
    int d_smooth_number_sweeps = 1;
    int d_schwarz_variant      = 0;
    int d_schwarz_overlap      = 1;
    int d_schwarz_domain_type  = 2;
    int d_schwarz_nonsymmetric = 0;
    int d_logging              = 0;
    int d_debug_flag           = 0;
    int d_rap2                 = 0;
    int d_keep_transpose       = 1;

    double d_strong_threshold      = 0.25;
    double d_max_row_sum           = 0.9;
    double d_non_galerkin_tol      = 0.0;
    double d_trunc_factor          = 0.0;
    double d_agg_trunc_factor      = 0.0;
    double d_agg_P12_trunc_factor  = 0.0;
    double d_additive_trunc_factor = 0.0;
    double d_relax_weight          = 0.0;
    double d_outer_weight          = 1.0;
    double d_chebyshev_fraction    = 0.3;
    double d_schwarz_weight        = 0.0;

    bool d_bSetupSolver = true; //! whether to call the setup for the BoomerAMG solver
};
} // namespace AMP::Solver

#endif
