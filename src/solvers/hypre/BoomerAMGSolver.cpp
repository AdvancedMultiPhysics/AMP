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
extern "C" {
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
}
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
BoomerAMGSolver::BoomerAMGSolver() : SolverStrategy() { d_bCreationPhase = true; }
BoomerAMGSolver::BoomerAMGSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    HYPRE_BoomerAMGCreate( &d_solver );

#ifdef USE_CUDA
    d_memory_location = HYPRE_MEMORY_DEVICE;
    d_exec_policy     = HYPRE_EXEC_DEVICE;
#else
    d_memory_location = HYPRE_MEMORY_HOST;
    d_exec_policy     = HYPRE_EXEC_HOST;
#endif

    AMP_ASSERT( parameters );
    initialize( parameters );
}

BoomerAMGSolver::~BoomerAMGSolver()
{
    HYPRE_BoomerAMGDestroy( d_solver );
    //    HYPRE_IJMatrixDestroy( d_ijMatrix );
    HYPRE_IJVectorDestroy( d_hypre_rhs );
    HYPRE_IJVectorDestroy( d_hypre_sol );
}

void BoomerAMGSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    getFromInput( parameters->d_db );

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }

    setParameters();
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

    HYPRE_BoomerAMGSetTol( d_solver, d_dRelativeTolerance );
    HYPRE_BoomerAMGSetMaxIter( d_solver, d_iMaxIterations );
    HYPRE_BoomerAMGSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );

    if ( db->keyExists( "memory_location" ) ) {
        auto memory_location = db->getString( "memory_location" );
        AMP_INSIST( memory_location == "host" || memory_location == "device",
                    "memory_location must be either device or host" );
        d_memory_location = ( memory_location == "host" ) ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;
    }
    if ( db->keyExists( "exec_policy" ) ) {
        auto exec_policy = db->getString( "exec_policy" );
        AMP_INSIST( exec_policy == "host" || exec_policy == "device",
                    "exec_policy must be either device or host" );
        d_exec_policy = ( exec_policy == "host" ) ? HYPRE_EXEC_HOST : HYPRE_EXEC_DEVICE;
    }
}

void BoomerAMGSolver::createHYPREMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    d_HypreMatrixAdaptor =
        std::make_shared<AMP::LinearAlgebra::HypreMatrixAdaptor>( matrix->getMatrixData() );
    AMP_ASSERT( d_HypreMatrixAdaptor );
    d_ijMatrix = d_HypreMatrixAdaptor->getHypreMatrix();
    if ( d_iDebugPrintInfoLevel > 3 ) {
        HYPRE_IJMatrixPrint( d_ijMatrix, "HypreMatrix" );
    }
}

void BoomerAMGSolver::createHYPREVectors()
{
    char hypre_mesg[100];

    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

    const auto &matrix = linearOperator->getMatrix();
    AMP_INSIST( matrix, "matrix cannot be NULL" );

    const auto myFirstRow = matrix->getLeftDOFManager()->beginDOF();
    const auto myEndRow =
        matrix->getLeftDOFManager()->endDOF(); // check whether endDOF is truly the last -1
    int ierr;

    // create the rhs
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow - 1, &d_hypre_rhs );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_rhs, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg );

    // create the solution vector
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow - 1, &d_hypre_sol );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_sol, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg );
}

void BoomerAMGSolver::copyToHypre( std::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v,
                                   HYPRE_IJVector hypre_v )
{
    char hypre_mesg[100];
    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS         = dofManager->numLocalDOF();
    const auto startingIndex = dofManager->beginDOF();

    HYPRE_Real *vals = nullptr;

    if ( amp_v->numberOfDataBlocks() == 1 ) {

        if ( amp_v->isType<HYPRE_Real>( 0 ) ) {
            vals = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                       ->getRawDataBlock<HYPRE_Real>();
        } else {

            auto block0 = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                              ->getRawDataBlock<HYPRE_Real>();
            auto memType = AMP::Utilities::getMemoryType( block0 );
            AMP_INSIST( memType < AMP::Utilities::MemoryType::device,
                        "Implemented only for AMP vector memory on host" );
            std::vector<size_t> indices( nDOFS, 0 );
            std::iota( indices.begin(), indices.end(), startingIndex );
            std::vector<HYPRE_Real> values( nDOFS, 0.0 );

            amp_v->getValuesByGlobalID( nDOFS, indices.data(), values.data() );

            vals = values.data();
        }


    } else {

        auto block0 = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                          ->getRawDataBlock<HYPRE_Real>();
        auto memType = AMP::Utilities::getMemoryType( block0 );
        AMP_INSIST( memType < AMP::Utilities::MemoryType::device,
                    "Implemented only for AMP vector memory on host" );

        std::vector<HYPRE_Real> values;
        for ( auto it = amp_v->begin<HYPRE_Real>(); it != amp_v->end<HYPRE_Real>(); ++it ) {
            values.push_back( *it );
        }

        vals = values.data();
    }

    AMP_ASSERT( vals );
    ierr = HYPRE_IJVectorInitialize( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorAssemble( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );

    // this can be optimized in future so that memory is allocated based on the location
    HYPRE_ParVector par_v;
    HYPRE_IJVectorGetObject( hypre_v, (void **) &par_v );
    hypre_ParVectorMigrate( par_v, d_memory_location );
}

template<typename T>
static void copy_to_amp( std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v, HYPRE_Real *values )
{
    AMP_ASSERT( amp_v && values );
    size_t i = 0;
    for ( auto it = amp_v->begin<T>(); it != amp_v->end<T>(); ++it ) {
        *it = values[i];
        ++i;
    }
}

void BoomerAMGSolver::copyFromHypre( HYPRE_IJVector hypre_v,
                                     std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v )
{
    char hypre_mesg[100];

    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS = dofManager->numLocalDOF();

    auto block0  = amp_v->getRawDataBlock<HYPRE_Real>();
    auto memType = AMP::Utilities::getMemoryType( block0 );

    if ( memType != AMP::Utilities::MemoryType::device ) {

        // this can be optimized in future so that there's less memory movement
        // likewise we should distinguish between managed and host options
        HYPRE_ParVector par_v;
        HYPRE_IJVectorGetObject( hypre_v, (void **) &par_v );
        hypre_ParVectorMigrate( par_v, HYPRE_MEMORY_HOST );
        std::vector<HYPRE_Real> values( nDOFS, 0.0 );
        auto values_p = values.data();
        ierr =
            HYPRE_IJVectorGetValues( hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, values_p );
        HYPRE_DescribeError( ierr, hypre_mesg );

        if ( amp_v->numberOfDataBlocks() == 1 ) {
            const auto startingIndex = dofManager->beginDOF();
            std::vector<size_t> indices( nDOFS, 0 );
            std::iota( indices.begin(), indices.end(), startingIndex );
            amp_v->setLocalValuesByGlobalID( nDOFS, indices.data(), values_p );

        } else {

            if ( amp_v->isType<double>( 0 ) ) {
                copy_to_amp<double>( amp_v, values_p );
            } else if ( amp_v->isType<float>( 0 ) ) {
                copy_to_amp<float>( amp_v, values_p );
            } else {
                AMP_ERROR( "Implemented only for double and float" );
            }
        }

    } else {
        AMP_ERROR( "Not implemented for AMP vector with device memory" );
    }
}

void BoomerAMGSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{

    d_pOperator = op;
    AMP_INSIST( d_pOperator, "ERROR: BoomerAMGSolver::registerOperator() operator cannot be NULL" );

    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

    auto matrix = linearOperator->getMatrix();
    AMP_INSIST( matrix, "matrix cannot be NULL" );

    // set the comm for this solver based on the comm for the matrix
    // being lazy??
    const auto &dofManager = matrix->getLeftDOFManager();
    d_comm                 = dofManager->getComm();

    createHYPREMatrix( matrix );
    createHYPREVectors();

    // the next section of code should initialize a hypre IJ matrix based on the AMP matrix
    d_bCreationPhase = false;
}

void BoomerAMGSolver::setParameters() {}

void BoomerAMGSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    PROFILE_START( "resetOperator" );
    AMP_INSIST( ( d_pOperator ),
                "ERROR: BoomerAMGSolver::resetOperator() operator cannot be NULL" );
    d_pOperator->reset( params );
    reset( std::shared_ptr<SolverStrategyParameters>() );
    PROFILE_STOP( "resetOperator" );
}


void BoomerAMGSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{
    PROFILE_START( "reset" );
    registerOperator( d_pOperator );
    PROFILE_STOP( "reset" );
}


void BoomerAMGSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE_START( "solve" );
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
    u->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

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

    PROFILE_STOP( "solve" );
}

} // namespace AMP::Solver
