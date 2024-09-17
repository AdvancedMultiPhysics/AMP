#include "AMP/solvers/hypre/BoomerAMGSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <numeric>

DISABLE_WARNINGS
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
BoomerAMGSolver::BoomerAMGSolver() : HypreSolver() { d_bCreationPhase = true; }
BoomerAMGSolver::BoomerAMGSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    HYPRE_BoomerAMGCreate( &d_solver );
    AMP_ASSERT( parameters );
    BoomerAMGSolver::initialize( parameters );
}

BoomerAMGSolver::~BoomerAMGSolver() { HYPRE_BoomerAMGDestroy( d_solver ); }

void BoomerAMGSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    BoomerAMGSolver::getFromInput( parameters->d_db );
}

void BoomerAMGSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    d_bComputeResidual = db->getWithDefault<bool>( "compute_residual", false );

    d_num_functions = db->getWithDefault<int>( "num_functions", 1 );
    HYPRE_BoomerAMGSetNumFunctions( d_solver, d_num_functions );

    d_min_iterations = db->getWithDefault<int>( "min_iterations", 0 );
    HYPRE_BoomerAMGSetMinIter( d_solver, d_min_iterations );

    d_max_coarse_size = db->getWithDefault<int>( "max_coarse_size", 32 );
    HYPRE_BoomerAMGSetMaxCoarseSize( d_solver, d_max_coarse_size );

    d_min_coarse_size = db->getWithDefault<int>( "min_coarse_size", 10 );
    HYPRE_BoomerAMGSetMinCoarseSize( d_solver, d_min_coarse_size );

    d_max_levels = db->getWithDefault<int>( "max_levels", 10 );
    HYPRE_BoomerAMGSetMaxLevels( d_solver, d_max_levels );

    if ( db->keyExists( "strong_threshold" ) ) {
        d_strong_threshold = db->getScalar<HYPRE_Real>( "strong_threshold" );
        HYPRE_BoomerAMGSetStrongThreshold( d_solver, d_strong_threshold );
    }

    if ( db->keyExists( "max_row_sum" ) ) {
        d_max_row_sum = db->getScalar<HYPRE_Real>( "max_row_sum" );
        HYPRE_BoomerAMGSetMaxRowSum( d_solver, d_max_row_sum );
    }

    if ( db->keyExists( "coarsen_type" ) ) {
        d_coarsen_type = db->getScalar<int>( "coarsen_type" );
        HYPRE_BoomerAMGSetCoarsenType( d_solver, d_coarsen_type );
    }

    if ( db->keyExists( "non_galerkin_tol" ) ) {
        d_non_galerkin_tol = db->getScalar<HYPRE_Real>( "non_galerkin_tol" );
        HYPRE_BoomerAMGSetNonGalerkinTol( d_solver, d_non_galerkin_tol );
    }

    if ( db->keyExists( "measure_type" ) ) {
        d_measure_type = db->getScalar<int>( "measure_type" );
        HYPRE_BoomerAMGSetMeasureType( d_solver, d_measure_type );
    }

    if ( db->keyExists( "agg_num_levels" ) ) {
        d_agg_num_levels = db->getScalar<int>( "agg_num_levels" );
        HYPRE_BoomerAMGSetAggNumLevels( d_solver, d_agg_num_levels );
    }

    if ( db->keyExists( "num_paths" ) ) {
        d_num_paths = db->getScalar<int>( "num_paths" );
        HYPRE_BoomerAMGSetNumPaths( d_solver, d_num_paths );
    }

    if ( db->keyExists( "cgc_iterations" ) ) {
        d_cgc_iterations = db->getScalar<int>( "cgc_iterations" );
        HYPRE_BoomerAMGSetCGCIts( d_solver, d_cgc_iterations );
    }

    if ( db->keyExists( "nodal" ) ) {
        d_nodal = db->getScalar<int>( "nodal" );
        HYPRE_BoomerAMGSetNodal( d_solver, d_nodal );
    }

    if ( db->keyExists( "nodal_diag" ) ) {
        d_nodal_diag = db->getScalar<int>( "nodal_diag" );
        HYPRE_BoomerAMGSetNodalDiag( d_solver, d_nodal_diag );
    }

    if ( db->keyExists( "interp_type" ) ) {
        d_interp_type = db->getScalar<int>( "interp_type" );
        HYPRE_BoomerAMGSetInterpType( d_solver, d_interp_type );
    }

    if ( db->keyExists( "trunc_factor" ) ) {
        d_trunc_factor = db->getScalar<HYPRE_Real>( "trunc_factor" );
        HYPRE_BoomerAMGSetTruncFactor( d_solver, d_trunc_factor );
    }

    if ( db->keyExists( "P_max_elements" ) ) {
        d_P_max_elements = db->getScalar<int>( "P_max_elements" );
        HYPRE_BoomerAMGSetPMaxElmts( d_solver, d_P_max_elements );
    }

    if ( db->keyExists( "separate_weights" ) ) {
        d_separate_weights = db->getScalar<int>( "separate_weights" );
        HYPRE_BoomerAMGSetSepWeight( d_solver, d_separate_weights );
    }

    if ( db->keyExists( "agg_interp_type" ) ) {
        d_agg_interp_type = db->getScalar<int>( "agg_interp_type" );
        HYPRE_BoomerAMGSetAggInterpType( d_solver, d_agg_interp_type );
    }

    if ( db->keyExists( "agg_trunc_factor" ) ) {
        d_agg_trunc_factor = db->getScalar<HYPRE_Real>( "agg_trunc_factor" );
        HYPRE_BoomerAMGSetAggTruncFactor( d_solver, d_agg_trunc_factor );
    }

    if ( db->keyExists( "agg_P12_trunc_factor" ) ) {
        d_agg_P12_trunc_factor = db->getScalar<HYPRE_Real>( "agg_P12_trunc_factor" );
        HYPRE_BoomerAMGSetAggP12TruncFactor( d_solver, d_agg_P12_trunc_factor );
    }

    if ( db->keyExists( "agg_P_max_elements" ) ) {
        d_agg_P_max_elements = db->getScalar<int>( "agg_P_max_elements" );
        HYPRE_BoomerAMGSetAggPMaxElmts( d_solver, d_agg_P_max_elements );
    }

    if ( db->keyExists( "agg_P12_max_elements" ) ) {
        d_agg_P12_max_elements = db->getScalar<int>( "agg_P12_max_elements" );
        HYPRE_BoomerAMGSetAggP12MaxElmts( d_solver, d_agg_P12_max_elements );
    }

    if ( db->keyExists( "number_samples" ) ) {
        d_number_samples = db->getScalar<int>( "number_samples" );
        HYPRE_BoomerAMGSetNumSamples( d_solver, d_number_samples );
    }

    if ( db->keyExists( "cycle_type" ) ) {
        d_cycle_type = db->getScalar<int>( "cycle_type" );
        HYPRE_BoomerAMGSetCycleType( d_solver, d_cycle_type );
    }

    if ( db->keyExists( "additive_level" ) ) {
        d_additive_level = db->getScalar<int>( "additive_level" );
        HYPRE_BoomerAMGSetAdditive( d_solver, d_additive_level );
    }

    if ( db->keyExists( "mult_additive_level" ) ) {
        d_mult_additive_level = db->getScalar<int>( "mult_additive_level" );
        HYPRE_BoomerAMGSetMultAdditive( d_solver, d_mult_additive_level );
    }

    if ( db->keyExists( "simple_level" ) ) {
        d_simple_level = db->getScalar<int>( "simple_level" );
        HYPRE_BoomerAMGSetSimple( d_solver, d_simple_level );
    }

    if ( db->keyExists( "additive_trunc_factor" ) ) {
        d_additive_trunc_factor = db->getScalar<HYPRE_Real>( "additive_trunc_factor" );
        HYPRE_BoomerAMGSetMultAddTruncFactor( d_solver, d_additive_trunc_factor );
    }

    if ( db->keyExists( "add_P_max_elmts" ) ) {
        d_add_P_max_elmts = db->getScalar<int>( "add_P_max_elmts" );
        HYPRE_BoomerAMGSetMultAddPMaxElmts( d_solver, d_add_P_max_elmts );
    }

    if ( db->keyExists( "number_sweeps" ) ) {
        d_number_sweeps = db->getScalar<int>( "number_sweeps" );
        HYPRE_BoomerAMGSetNumSweeps( d_solver, d_number_sweeps );
    }

    if ( db->keyExists( "relax_type" ) ) {
        d_relax_type = db->getScalar<int>( "relax_type" );
        HYPRE_BoomerAMGSetRelaxType( d_solver, d_relax_type );
    }

    // specify Gaussian elimination on the coarsest level
    HYPRE_BoomerAMGSetCycleRelaxType( d_solver, 9, 3 );

    if ( db->keyExists( "relax_order" ) ) {
        d_relax_order = db->getScalar<int>( "relax_order" );
        HYPRE_BoomerAMGSetRelaxOrder( d_solver, d_relax_order );
    }

    if ( db->keyExists( "relax_weight" ) ) {
        d_relax_weight = db->getScalar<HYPRE_Real>( "relax_weight" );
        HYPRE_BoomerAMGSetRelaxWt( d_solver, d_relax_weight );
    }

    if ( db->keyExists( "outer_weight" ) ) {
        d_outer_weight = db->getScalar<HYPRE_Real>( "outer_weight" );
        HYPRE_BoomerAMGSetOuterWt( d_solver, d_outer_weight );
    }

    if ( db->keyExists( "chebyshev_order" ) ) {
        d_chebyshev_order = db->getScalar<int>( "chebyshev_order" );
        HYPRE_BoomerAMGSetChebyOrder( d_solver, d_chebyshev_order );
    }

    if ( db->keyExists( "chebyshev_fraction" ) ) {
        d_chebyshev_fraction = db->getScalar<HYPRE_Real>( "chebyshev_fraction" );
        HYPRE_BoomerAMGSetChebyFraction( d_solver, d_chebyshev_fraction );
    }

    if ( db->keyExists( "smooth_type" ) ) {
        d_smooth_type = db->getScalar<int>( "smooth_type" );
        HYPRE_BoomerAMGSetSmoothType( d_solver, d_smooth_type );
    }

    if ( db->keyExists( "smooth_number_levels" ) ) {
        d_smooth_number_levels = db->getScalar<int>( "smooth_number_levels" );
        HYPRE_BoomerAMGSetSmoothNumLevels( d_solver, d_smooth_number_levels );
    }

    if ( db->keyExists( "smooth_number_sweeps" ) ) {
        d_smooth_number_sweeps = db->getScalar<int>( "smooth_number_sweeps" );
        HYPRE_BoomerAMGSetSmoothNumSweeps( d_solver, d_smooth_number_sweeps );
    }

    if ( db->keyExists( "schwarz_variant" ) ) {
        d_schwarz_variant = db->getScalar<int>( "schwarz_variant" );
        HYPRE_BoomerAMGSetVariant( d_solver, d_schwarz_variant );
    }

    if ( db->keyExists( "schwarz_overlap" ) ) {
        d_schwarz_overlap = db->getScalar<int>( "schwarz_overlap" );
        HYPRE_BoomerAMGSetOverlap( d_solver, d_schwarz_overlap );
    }

    if ( db->keyExists( "schwarz_domain_type" ) ) {
        d_schwarz_domain_type = db->getScalar<int>( "schwarz_domain_type" );
        HYPRE_BoomerAMGSetDomainType( d_solver, d_schwarz_domain_type );
    }

    if ( db->keyExists( "schwarz_weight" ) ) {
        d_schwarz_weight = db->getScalar<int>( "schwarz_weight" );
        HYPRE_BoomerAMGSetSchwarzRlxWeight( d_solver, d_schwarz_weight );
    }

    if ( db->keyExists( "schwarz_nonsymmetric" ) ) {
        d_schwarz_nonsymmetric = db->getScalar<int>( "schwarz_nonsymmetric" );
        HYPRE_BoomerAMGSetSchwarzUseNonSymm( d_solver, d_schwarz_nonsymmetric );
    }

    if ( db->keyExists( "logging" ) ) {
        d_logging = db->getScalar<int>( "logging" );
        HYPRE_BoomerAMGSetLogging( d_solver, d_logging );
    }

    if ( db->keyExists( "debug_flag" ) ) {
        d_debug_flag = db->getScalar<int>( "debug_flag" );
        HYPRE_BoomerAMGSetDebugFlag( d_solver, d_debug_flag );
    }

    d_rap2 = db->getWithDefault<int>( "rap2", 0 );
    HYPRE_BoomerAMGSetRAP2( d_solver, d_rap2 );

    if ( db->keyExists( "keep_transpose" ) ) {
        d_keep_transpose = db->getScalar<int>( "keep_transpose" );
        HYPRE_BoomerAMGSetKeepTranspose( d_solver, d_keep_transpose );
    }

    HYPRE_BoomerAMGSetTol( d_solver, static_cast<HYPRE_Real>( d_dRelativeTolerance ) );
    HYPRE_BoomerAMGSetMaxIter( d_solver, d_iMaxIterations );
    HYPRE_BoomerAMGSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );
}

void BoomerAMGSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "solve" );
    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: BoomerAMGSolver::apply() operator cannot be NULL" );

    HYPRE_SetMemoryLocation( d_memory_location );
    HYPRE_SetExecutionPolicy( d_exec_policy );

    if ( d_bUseZeroInitialGuess ) {
        u->zero();
    }

    if ( d_bCreationPhase ) {
        d_bCreationPhase = false;
    }

    copyToHypre( u, d_hypre_sol );
    copyToHypre( f, d_hypre_rhs );

    std::shared_ptr<AMP::LinearAlgebra::Vector> r;

    if ( d_bComputeResidual ) {
        r = f->clone();
        d_pOperator->residual( f, u, r );
        const auto initialResNorm = r->L2Norm();

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BoomerAMGSolver::apply(), L2 norm of residual before solve "
                      << std::setprecision( 15 ) << initialResNorm << std::endl;
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        HYPRE_Real solution_norm( u->L2Norm() );
        AMP::pout << "BoomerAMGSolver : before solve solution norm: " << std::setprecision( 15 )
                  << solution_norm << std::endl;
    }

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );
    hypre_ParCSRMatrixMigrate( parcsr_A, d_memory_location );

    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_b );
    HYPRE_IJVectorGetObject( d_hypre_sol, (void **) &par_x );

    // add in code for solve here
    HYPRE_BoomerAMGSetup( d_solver, parcsr_A, par_b, par_x );
    HYPRE_BoomerAMGSolve( d_solver, parcsr_A, par_b, par_x );

    copyFromHypre( d_hypre_sol, u );

    // Check for NaNs in the solution (no communication necessary)
    auto localNorm = u->getVectorOperations()->localL2Norm( *u->getVectorData() ).get<HYPRE_Real>();
    AMP_INSIST( localNorm == localNorm, "NaNs detected in solution" );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    HYPRE_BoomerAMGGetNumIterations( d_solver, &d_iNumberIterations );
    HYPRE_Real hypre_norm;
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm( d_solver, &hypre_norm );
    d_dResidualNorm = hypre_norm;

    if ( d_dResidualNorm < d_dRelativeTolerance ) {
        d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "BoomerAMGSolver : after solve solution norm: " << std::setprecision( 15 )
                  << u->L2Norm() << std::endl;
    }

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, r );
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BoomerAMGSolver::apply(), L2 norm of residual after solve "
                      << std::setprecision( 15 ) << r->L2Norm() << std::endl;
        }
    }
}

} // namespace AMP::Solver
