#include "solvers/hypre/BoomerAMGSolver.h"

#include "ProfilerApp.h"
#include "matrices/Matrix.h"
#include "operators/LinearOperator.h"
#include "utils/Utilities.h"
#include "vectors/DataChangeFirer.h"

#include <iomanip>
#include <numeric>

namespace AMP {
namespace Solver {


/****************************************************************
* Constructors / Destructor                                     *
****************************************************************/
BoomerAMGSolver::BoomerAMGSolver()
{
    d_bCreationPhase = true;
}
BoomerAMGSolver::BoomerAMGSolver( AMP::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{

    HYPRE_BoomerAMGCreate( &d_solver );

    AMP_ASSERT( parameters.get() != nullptr );
    initialize( parameters );

}

BoomerAMGSolver::~BoomerAMGSolver()
{
    HYPRE_BoomerAMGDestroy( d_solver );
    HYPRE_IJMatrixDestroy( d_ijMatrix );
    HYPRE_IJVectorDestroy( d_hypre_rhs );
    HYPRE_IJVectorDestroy( d_hypre_sol );
}

void BoomerAMGSolver::initialize( AMP::shared_ptr<SolverStrategyParameters> const parameters )
{
    getFromInput( parameters->d_db );

    if ( d_pOperator.get() != nullptr ) {
        registerOperator( d_pOperator );
    }

    setParameters();
}

void BoomerAMGSolver::getFromInput( const AMP::shared_ptr<AMP::Database> &db )
{
    // 6.2.10 in hypre 11.2 manual 
    d_num_functions    = db->getIntegerWithDefault( "num_functions", 1);
    HYPRE_BoomerAMGSetNumFunctions       ( d_solver, d_num_functions );

    // 6.2.14 in hypre 11.2 manual 
    d_min_iterations  = db->getIntegerWithDefault( "min_iterations", 0 );
    HYPRE_BoomerAMGSetMinIter            ( d_solver, d_min_iterations);

    // 6.2.15 in hypre 11.2 manual 
    d_max_coarse_size  = db->getIntegerWithDefault( "max_coarse_size", 800 );
    HYPRE_BoomerAMGSetMaxCoarseSize      ( d_solver, d_max_coarse_size );

    // 6.2.16 in hypre 11.2 manual 
    d_min_coarse_size  = db->getIntegerWithDefault( "min_coarse_size", 100 );
    HYPRE_BoomerAMGSetMinCoarseSize      ( d_solver, d_min_coarse_size);

    // 6.2.17 in hypre 11.2 manual 
    d_max_levels       = db->getIntegerWithDefault( "max_levels", 10);
    HYPRE_BoomerAMGSetMaxLevels          ( d_solver, d_max_levels);

    // 6.2.18 in hypre 11.2 manual 
    if( db->keyExists( "strong_threshold" ) ) {
        d_strong_threshold = db->getDouble( "strong_threshold" );
        HYPRE_BoomerAMGSetStrongThreshold    ( d_solver, d_strong_threshold);
    }

    // 6.2.20 in hypre 11.2 manual 
    if( db->keyExists( "max_row_sum" ) ) {
        d_max_row_sum = db->getDouble( "max_row_sum" );
        HYPRE_BoomerAMGSetMaxRowSum          ( d_solver, d_max_row_sum);
    }

    // 6.2.21 in hypre 11.2 manual 
    if( db->keyExists( "coarsen_type" ) ) {
        d_coarsen_type = db->getInteger( "coarsen_type" );
        HYPRE_BoomerAMGSetCoarsenType        ( d_solver, d_coarsen_type);
    }

    // 6.2.23 in hypre 11.2 manual 
    if( db->keyExists( "non_galerkin_tol" ) ) {
        d_non_galerkin_tol = db->getDouble( "non_galerkin_tol" );
        HYPRE_BoomerAMGSetNonGalerkinTol     ( d_solver, d_non_galerkin_tol);
    }

    // 6.2.24 in hypre 11.2 manual 
    if( db->keyExists( "measure_type" ) ) {
        d_measure_type = db->getInteger( "measure_type" );
        HYPRE_BoomerAMGSetMeasureType        ( d_solver, d_measure_type);
    }

    // 6.2.25 in hypre 11.2 manual 
    if( db->keyExists( "agg_num_levels" ) ) {
        d_agg_num_levels = db->getInteger( "agg_num_levels" );
        HYPRE_BoomerAMGSetAggNumLevels       ( d_solver, d_agg_num_levels);
    }

    // 6.2.26 in hypre 11.2 manual 
    if( db->keyExists( "num_paths" ) ) {
        d_num_paths = db->getInteger( "num_paths" );
        HYPRE_BoomerAMGSetNumPaths           ( d_solver, d_num_paths);
    }

    // 6.2.27 in hypre 11.2 manual 
    if( db->keyExists( "cgc_iterations" ) ) {
        d_cgc_iterations = db->getInteger( "cgc_iterations" );
        HYPRE_BoomerAMGSetCGCIts             ( d_solver, d_cgc_iterations);
    }

    // 6.2.28 in hypre 11.2 manual 
    if( db->keyExists( "nodal" ) ) {
        d_nodal = db->getInteger( "nodal" );
        HYPRE_BoomerAMGSetNodal              ( d_solver, d_nodal);
    }

    // 6.2.29 in hypre 11.2 manual 
    if( db->keyExists( "nodal_diag" ) ) {
        d_nodal_diag = db->getInteger( "nodal_diag" );
        HYPRE_BoomerAMGSetNodalDiag          ( d_solver, d_nodal_diag);
    }

    // 6.2.30 in hypre 11.2 manual 
    if( db->keyExists( "interp_type" ) ) {
        d_interp_type = db->getInteger( "interp_type" );
        HYPRE_BoomerAMGSetInterpType         ( d_solver, d_interp_type);
    }

    // 6.2.31 in hypre 11.2 manual 
    if( db->keyExists( "trunc_factor" ) ) {
        d_trunc_factor = db->getDouble( "trunc_factor" );
        HYPRE_BoomerAMGSetTruncFactor        ( d_solver, d_trunc_factor);
    }

    // 6.2.32 in hypre 11.2 manual 
    if( db->keyExists( "P_max_elements" ) ) {
        d_P_max_elements = db->getInteger( "P_max_elements" );
        HYPRE_BoomerAMGSetPMaxElmts          ( d_solver, d_P_max_elements);
    }

    // 6.2.33 in hypre 11.2 manual 
    if( db->keyExists( "separate_weights" ) ) {
        d_separate_weights = db->getInteger( "separate_weights" );
        HYPRE_BoomerAMGSetSepWeight          ( d_solver, d_separate_weights);
    }

    // 6.2.34 in hypre 11.2 manual 
    if( db->keyExists( "agg_interp_type" ) ) {
        d_agg_interp_type = db->getInteger( "agg_interp_type" );
        HYPRE_BoomerAMGSetAggInterpType      ( d_solver, d_agg_interp_type);
    }

    // 6.2.35 in hypre 11.2 manual 
    if( db->keyExists( "agg_trunc_factor" ) ) {
        d_agg_trunc_factor = db->getDouble( "agg_trunc_factor" );
        HYPRE_BoomerAMGSetAggTruncFactor     ( d_solver, d_agg_trunc_factor);
    }

    // 6.2.36 in hypre 11.2 manual 
    if( db->keyExists( "agg_P12_trunc_factor" ) ) {
        d_agg_P12_trunc_factor = db->getDouble( "agg_P12_trunc_factor" );
        HYPRE_BoomerAMGSetAggP12TruncFactor  ( d_solver, d_agg_P12_trunc_factor);
    }

    // 6.2.37 in hypre 11.2 manual 
    if( db->keyExists( "agg_P_max_elements" ) ) {
        d_agg_P_max_elements = db->getInteger( "agg_P_max_elements" );
        HYPRE_BoomerAMGSetAggPMaxElmts       ( d_solver, d_agg_P_max_elements);
    }

    // 6.2.38 in hypre 11.2 manual 
    if( db->keyExists( "agg_P12_max_elements" ) ) {
        d_agg_P12_max_elements = db->getInteger( "agg_P12_max_elements" );
        HYPRE_BoomerAMGSetAggP12MaxElmts     ( d_solver, d_agg_P12_max_elements);
    }

    // 6.2.44 in hypre 11.2 manual 
    if( db->keyExists( "number_samples" ) ) {
        d_number_samples = db->getInteger( "number_samples" );
        HYPRE_BoomerAMGSetNumSamples         ( d_solver, d_number_samples);
    }

    // 6.2.45 in hypre 11.2 manual 
    if( db->keyExists( "cycle_type" ) ) {
        d_cycle_type = db->getInteger( "cycle_type" );
        HYPRE_BoomerAMGSetCycleType          ( d_solver, d_cycle_type);
    }

    // 6.2.46 in hypre 11.2 manual 
    if( db->keyExists( "additive_level" ) ) {
        d_additive_level = db->getInteger( "additive_level" );
        HYPRE_BoomerAMGSetAdditive           ( d_solver, d_additive_level);
    }

    // 6.2.47 in hypre 11.2 manual 
    if( db->keyExists( "mult_additive_level" ) ) {
        d_mult_additive_level = db->getInteger( "mult_additive_level" );
        HYPRE_BoomerAMGSetMultAdditive       ( d_solver, d_mult_additive_level);
    }

    // 6.2.48 in hypre 11.2 manual 
    if( db->keyExists( "simple_level" ) ) {
        d_simple_level = db->getInteger( "simple_level" );
        HYPRE_BoomerAMGSetSimple             ( d_solver, d_simple_level);
    }

    // 6.2.49 in hypre 11.2 manual 
    if( db->keyExists( "additive_trunc_factor" ) ) {
        d_additive_trunc_factor = db->getDouble( "additive_trunc_factor" );
        HYPRE_BoomerAMGSetMultAddTruncFactor ( d_solver, d_additive_trunc_factor);
    }

    // 6.2.50 in hypre 11.2 manual 
    if( db->keyExists( "add_P_max_elmts" ) ){
        d_add_P_max_elmts = db->getDouble( "add_P_max_elmts" );
        HYPRE_BoomerAMGSetMultAddPMaxElmts   ( d_solver, d_add_P_max_elmts);
    }

    // 6.2.55 in hypre 11.2 manual 
    if( db->keyExists( "number_sweeps" ) ) {
        d_number_sweeps = db->getInteger( "number_sweeps" );
        HYPRE_BoomerAMGSetNumSweeps          ( d_solver, d_number_sweeps);
    }

    // 6.2.58 in hypre 11.2 manual 
    if( db->keyExists( "relax_type" ) ) {
        d_relax_type = db->getInteger( "relax_type" );
        HYPRE_BoomerAMGSetRelaxType          ( d_solver, d_relax_type);
    }

    // 6.2.60 in hypre 11.2 manual 
    if( db->keyExists( "relax_order" ) ) {
        d_relax_order = db->getInteger( "relax_order" );
        HYPRE_BoomerAMGSetRelaxOrder         ( d_solver, d_relax_order);
    }

    // 6.2.61 in hypre 11.2 manual 
    if( db->keyExists( "relax_weight" ) ) {
        d_relax_weight = db->getDouble( "relax_weight" );
        HYPRE_BoomerAMGSetRelaxWt            ( d_solver, d_relax_weight);
    }

    // 6.2.64 in hypre 11.2 manual 
    if( db->keyExists( "outer_weight" ) ) {
        d_outer_weight = db->getDouble( "outer_weight" );
        HYPRE_BoomerAMGSetOuterWt            ( d_solver, d_outer_weight);
    }

    // 6.2.66 in hypre 11.2 manual 
    if( db->keyExists( "chebyshev_order" ) ) {
        d_chebyshev_order = db->getInteger( "chebyshev_order" );
        HYPRE_BoomerAMGSetChebyOrder         ( d_solver, d_chebyshev_order);
    }

    // 6.2.67 in hypre 11.2 manual 
    if( db->keyExists( "chebyshev_fraction" ) ) {
        d_chebyshev_fraction = db->getDouble( "chebyshev_fraction" );
        HYPRE_BoomerAMGSetChebyFraction      ( d_solver, d_chebyshev_fraction);
    }

    // 6.2.68 in hypre 11.2 manual 
    if( db->keyExists( "smooth_type" ) ) {
        d_smooth_type = db->getInteger( "smooth_type" );
        HYPRE_BoomerAMGSetSmoothType         ( d_solver, d_smooth_type);
    }
        
    // 6.2.69 in hypre 11.2 manual 
    if( db->keyExists( "smooth_number_levels" ) ) {
        d_smooth_number_levels = db->getInteger( "smooth_number_levels" );
        HYPRE_BoomerAMGSetSmoothNumLevels    ( d_solver, d_smooth_number_levels);
    }

    // 6.2.70 in hypre 11.2 manual 
    if( db->keyExists( "smooth_number_sweeps" ) ) {
        d_smooth_number_sweeps = db->getInteger( "smooth_number_sweeps" );
        HYPRE_BoomerAMGSetSmoothNumSweeps    ( d_solver, d_smooth_number_sweeps);
    }

    // 6.2.71 in hypre 11.2 manual 
    if( db->keyExists( "schwarz_variant" ) ) {
        d_schwarz_variant = db->getInteger( "schwarz_variant" );
        HYPRE_BoomerAMGSetVariant            ( d_solver, d_schwarz_variant);
    }

    // 6.2.72 in hypre 11.2 manual 
    if( db->keyExists( "schwarz_overlap" ) ) {
        d_schwarz_overlap = db->getInteger( "schwarz_overlap" );
        HYPRE_BoomerAMGSetOverlap            ( d_solver, d_schwarz_overlap);
    }

    // 6.2.73 in hypre 11.2 manual 
    if( db->keyExists( "schwarz_domain_type" ) ) {
        d_schwarz_domain_type = db->getInteger( "schwarz_domain_type" );
        HYPRE_BoomerAMGSetDomainType         ( d_solver, d_schwarz_domain_type);
    }

    // 6.2.74 in hypre 11.2 manual 
    if( db->keyExists( "schwarz_weight" ) ) {
        d_schwarz_weight = db->getInteger( "schwarz_weight" );
        HYPRE_BoomerAMGSetSchwarzRlxWeight   ( d_solver, d_schwarz_weight);
    }

    // 6.2.75 in hypre 11.2 manual 
    if( db->keyExists( "schwarz_nonsymmetric" ) ) {
        d_schwarz_nonsymmetric = db->getInteger( "schwarz_nonsymmetric" );
        HYPRE_BoomerAMGSetSchwarzUseNonSymm  ( d_solver, d_schwarz_nonsymmetric);
    }

    // 6.2.87 in hypre 11.2 manual 
    if( db->keyExists( "logging" ) ) {
        d_logging = db->getInteger( "logging" );
        HYPRE_BoomerAMGSetLogging            ( d_solver, d_logging);
    }

    // 6.2.88 in hypre 11.2 manual 
    if( db->keyExists( "debug_flag" ) ) {
        d_debug_flag = db->getInteger( "debug_flag" );
        HYPRE_BoomerAMGSetDebugFlag          ( d_solver, d_debug_flag);
    }

    // 6.2.90 in hypre 11.2 manual 
    d_rap2 = db->getIntegerWithDefault( "rap2", 0 );
    HYPRE_BoomerAMGSetRAP2               ( d_solver, d_rap2);

    // 6.2.91 in hypre 11.2 manual 
    if( db->keyExists( "keep_transpose" ) ) {
        d_keep_transpose = db->getInteger( "keep_transpose" );
        HYPRE_BoomerAMGSetKeepTranspose      ( d_solver, d_keep_transpose);
    }

    HYPRE_BoomerAMGSetTol                ( d_solver, d_dMaxError );
    HYPRE_BoomerAMGSetMaxIter            ( d_solver, d_iMaxIterations);
    HYPRE_BoomerAMGSetPrintLevel         ( d_solver, d_iDebugPrintInfoLevel);

}

void BoomerAMGSolver::createHYPREMatrix( const AMP::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    int ierr;
    char hypre_mesg[100];

    const auto myFirstRow = matrix->getLeftDOFManager()->beginDOF();
    const auto myEndRow   = matrix->getLeftDOFManager()->endDOF(); // check whether endDOF is truly the last -1 

    ierr = HYPRE_IJMatrixCreate( d_comm.getCommunicator(), myFirstRow, myEndRow-1, myFirstRow, myEndRow-1, &d_ijMatrix );
    HYPRE_DescribeError( ierr, hypre_mesg);

    ierr = HYPRE_IJMatrixSetObjectType( d_ijMatrix, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg);

    ierr = HYPRE_IJMatrixInitialize( d_ijMatrix );
    HYPRE_DescribeError( ierr, hypre_mesg);

    std::vector<unsigned int> cols;
    std::vector<double> values;

    // iterate over all rows
    for(auto i=myFirstRow; i!=myEndRow; ++i) {
        matrix->getRowByGlobalID(i, cols, values);
        const int nrows = 1;
        const auto irow = i;
        const auto ncols = cols.size();
        ierr = HYPRE_IJMatrixSetValues( d_ijMatrix, 
                                        nrows, 
                                        (HYPRE_Int *)&ncols, 
                                        (HYPRE_Int *)&irow, 
                                        (HYPRE_Int *) &cols[0], 
                                        (const double *) &values[0] );
        HYPRE_DescribeError( ierr, hypre_mesg);
    }

    ierr = HYPRE_IJMatrixAssemble( d_ijMatrix );
    HYPRE_DescribeError( ierr, hypre_mesg);
}

void BoomerAMGSolver::createHYPREVectors( void )
{
    char hypre_mesg[100];

    auto linearOperator =
        AMP::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator.get() != nullptr, "linearOperator cannot be NULL" );
    
    const auto &matrix = linearOperator->getMatrix();
    AMP_INSIST( matrix.get() != nullptr, "matrix cannot be NULL" );

    const auto myFirstRow = matrix->getLeftDOFManager()->beginDOF();
    const auto myEndRow   = matrix->getLeftDOFManager()->endDOF(); // check whether endDOF is truly the last -1 
    int ierr;

    // create the rhs
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow-1, &d_hypre_rhs );
    HYPRE_DescribeError( ierr, hypre_mesg);
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_rhs, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg);

    // create the solution vector
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow-1, &d_hypre_sol );
    HYPRE_DescribeError( ierr, hypre_mesg);
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_sol, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg);

}

void BoomerAMGSolver::copyToHypre( AMP::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v, 
                                   HYPRE_IJVector hypre_v )
{
    char hypre_mesg[100];
    int ierr;

    AMP_INSIST( amp_v.get() != nullptr, "vector cannot be NULL" );
    const auto &dofManager =  amp_v->getDOFManager();
    AMP_INSIST( dofManager.get() != nullptr, "DOF_Manager cannot be NULL" );
    
    const auto startingIndex = dofManager->beginDOF();
    const auto nDOFS = dofManager->numLocalDOF();
    
    std::vector<size_t> indices(nDOFS, 0);
    std::vector<HYPRE_Int> hypre_indices(nDOFS, 0);
    std::iota(indices.begin(), indices.end(), (HYPRE_Int) startingIndex);
    std::copy(indices.begin(), indices.end(), hypre_indices.begin() );

    std::vector<double> values(nDOFS, 0.0);
    
    amp_v->getValuesByGlobalID( nDOFS, &indices[0], &values[0] );
 
    ierr = HYPRE_IJVectorInitialize(hypre_v);
    HYPRE_DescribeError( ierr, hypre_mesg);
    ierr = HYPRE_IJVectorSetValues( hypre_v, nDOFS,  &hypre_indices[0], &values[0] );
    HYPRE_DescribeError( ierr, hypre_mesg);
    ierr = HYPRE_IJVectorAssemble(hypre_v);
    HYPRE_DescribeError( ierr, hypre_mesg);
}


void BoomerAMGSolver::copyFromHypre( HYPRE_IJVector hypre_v, 
                                     AMP::shared_ptr<AMP::LinearAlgebra::Vector> amp_v )
{
    char hypre_mesg[100];

    int ierr;

    AMP_INSIST( amp_v.get() != nullptr, "vector cannot be NULL" );
    const auto &dofManager =  amp_v->getDOFManager();
    AMP_INSIST( dofManager.get() != nullptr, "DOF_Manager cannot be NULL" );
    
    const auto startingIndex = dofManager->beginDOF();
    const auto nDOFS = dofManager->numLocalDOF();
    
    std::vector<size_t> indices(nDOFS, 0);
    std::vector<HYPRE_Int> hypre_indices(nDOFS, 0);
    std::iota(indices.begin(), indices.end(), startingIndex);
    std::copy(indices.begin(), indices.end(), hypre_indices.begin() );

    std::vector<double> values(nDOFS, 0.0);

    ierr = HYPRE_IJVectorGetValues( hypre_v, static_cast<HYPRE_Int>(nDOFS),  &hypre_indices[0], &values[0] );
    HYPRE_DescribeError( ierr, hypre_mesg);
    amp_v->setLocalValuesByGlobalID( nDOFS, &indices[0], &values[0] );

}

void BoomerAMGSolver::registerOperator( const AMP::shared_ptr<AMP::Operator::Operator> op )
{

    d_pOperator = op;
    AMP_INSIST( d_pOperator.get() != nullptr,
                "ERROR: BoomerAMGSolver::registerOperator() operator cannot be NULL" );

    auto linearOperator =
        AMP::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator.get() != nullptr, "linearOperator cannot be NULL" );

    auto matrix = linearOperator->getMatrix();
    AMP_INSIST( matrix.get() != nullptr, "matrix cannot be NULL" );

    // set the comm for this solver based on the comm for the matrix
    // being lazy??
    const auto &dofManager = matrix->getLeftDOFManager();
    d_comm = dofManager->getComm();

    createHYPREMatrix( matrix );
    createHYPREVectors( );

    // the next section of code should initialize a hypre IJ matrix based on the AMP matrix
    d_bCreationPhase = false;
}

void BoomerAMGSolver::setParameters( void )
{

}

void BoomerAMGSolver::resetOperator(
    const AMP::shared_ptr<AMP::Operator::OperatorParameters> params )
{
    PROFILE_START( "resetOperator" );
    AMP_INSIST( ( d_pOperator.get() != nullptr ),
                "ERROR: BoomerAMGSolver::resetOperator() operator cannot be NULL" );
    d_pOperator->reset( params );
    reset( AMP::shared_ptr<SolverStrategyParameters>() );
    PROFILE_STOP( "resetOperator" );
}


void BoomerAMGSolver::reset( AMP::shared_ptr<SolverStrategyParameters> )
{
    PROFILE_START( "reset" );
    registerOperator( d_pOperator );
    PROFILE_STOP( "reset" );
}


void BoomerAMGSolver::solve( AMP::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             AMP::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE_START( "solve" );
    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator.get() != nullptr,
                "ERROR: BoomerAMGSolver::solve() operator cannot be NULL" );

    if ( d_bUseZeroInitialGuess ) {
        u->zero();
    }

    if ( d_bCreationPhase ) {
        d_bCreationPhase = false;
    }

    copyToHypre(u, d_hypre_sol);
    copyToHypre(f, d_hypre_rhs);

    AMP::shared_ptr<AMP::LinearAlgebra::Vector> r;

    bool computeResidual = false;

    if ( computeResidual ) {
        r = f->cloneVector();
        d_pOperator->residual( f, u, r );
        auto initialResNorm = r->L2Norm();

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BoomerAMGSolver::solve(), L2 norm of residual before solve "
                      << std::setprecision( 15 ) << initialResNorm << std::endl;
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        double solution_norm = u->L2Norm();
        AMP::pout << "BoomerAMGSolver : before solve solution norm: " << std::setprecision( 15 )
                  << solution_norm << std::endl;
    }

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_IJMatrixGetObject(d_ijMatrix, (void **) &parcsr_A);
    HYPRE_IJVectorGetObject(d_hypre_rhs, (void **) &par_b);
    HYPRE_IJVectorGetObject(d_hypre_sol, (void **) &par_x);

    
    // add in code for solve here
    HYPRE_BoomerAMGSetup( d_solver, parcsr_A, par_b, par_x );
    HYPRE_BoomerAMGSolve( d_solver, parcsr_A, par_b, par_x );

    copyFromHypre( d_hypre_sol, u );

    // Check for NaNs in the solution (no communication necessary)
    double localNorm = u->localL2Norm();
    AMP_INSIST( localNorm == localNorm, "NaNs detected in solution" );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    if ( u->isA<AMP::LinearAlgebra::DataChangeFirer>() ) {
        u->castTo<AMP::LinearAlgebra::DataChangeFirer>().fireDataChange();
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        double solution_norm = u->L2Norm();
        AMP::pout << "BoomerAMGSolver : after solve solution norm: " << std::setprecision( 15 )
                  << solution_norm << std::endl;
    }

    if ( computeResidual ) {
        d_pOperator->residual( f, u, r );
        auto finalResNorm = r->L2Norm();

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BoomerAMGSolver::solve(), L2 norm of residual after solve "
                      << std::setprecision( 15 ) << finalResNorm << std::endl;
        }
    }

    PROFILE_STOP( "solve" );
}

} // Solver
} // AMP
