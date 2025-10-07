#include "AMP/solvers/trilinos/muelu/TrilinosMueLuSolver.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/trilinos/epetra/EpetraMatrixData.h"
#include "AMP/matrices/trilinos/epetra/EpetraMatrixHelpers.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

#include "ProfilerApp.h"

// Trilinos includes
DISABLE_WARNINGS
#include "MueLu.hpp"
#include "MueLu_CreateEpetraPreconditioner.hpp"
#include "MueLu_DirectSolver.hpp"
#include "MueLu_Ifpack2Smoother.hpp"
#include "MueLu_IfpackSmoother.hpp"
#include "MueLu_ParameterListInterpreter_decl.hpp"
#include "MueLu_RAPFactory.hpp"
#include "MueLu_SaPFactory.hpp"
#include "MueLu_SmootherFactory.hpp"
#include "MueLu_TentativePFactory.hpp"
#include "MueLu_TransPFactory.hpp"
#include "MueLu_TrilinosSmoother.hpp"
#include "Teuchos_RCP.hpp"
#include "Xpetra_EpetraCrsMatrix.hpp"
#include "Xpetra_EpetraVector.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_Operator.hpp"
#if TRILINOS_MAJOR_MINOR_VERSION >= 130400
    #include "MueLu_CreateEpetraPreconditioner.hpp"
#endif
ENABLE_WARNINGS


namespace AMP {
namespace Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
TrilinosMueLuSolver::TrilinosMueLuSolver() : d_bCreationPhase( true ), d_maxLevels( 0 ) {}
TrilinosMueLuSolver::TrilinosMueLuSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters ), d_maxLevels( 0 )
{
    AMP_ASSERT( parameters );
    initialize( parameters );
}
TrilinosMueLuSolver::~TrilinosMueLuSolver()
{
    d_mueluSolver.reset();
    d_matrix.reset(); // Need to keep a copy of the matrix alive until after the solver is destroyed
}

void TrilinosMueLuSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    getFromInput( parameters->d_db );

    if ( d_pOperator ) {

        registerOperator( d_pOperator );
    }
}

Teuchos::RCP<MueLu::TentativePFactory<SC, LO, GO, NO>>
TrilinosMueLuSolver::getTentativePFactory( void )
{
    return Teuchos::rcp( new MueLu::TentativePFactory<SC, LO, GO, NO>() );
}

Teuchos::RCP<MueLu::SaPFactory<SC, LO, GO, NO>> TrilinosMueLuSolver::getSaPFactory( void )
{
    return Teuchos::rcp( new MueLu::SaPFactory<SC, LO, GO, NO>() );
}

Teuchos::RCP<MueLu::TransPFactory<SC, LO, GO, NO>> TrilinosMueLuSolver::getRFactory( void )
{
    return Teuchos::rcp( new MueLu::TransPFactory<SC, LO, GO, NO>() );
}

Teuchos::RCP<MueLu::SmootherFactory<SC, LO, GO, NO>>
TrilinosMueLuSolver::getCoarseSolverFactory( void )
{
    auto coarseSolverPrototype = Teuchos::rcp( new MueLu::DirectSolver<SC, LO, GO, NO>() );
    return Teuchos::rcp(
        new MueLu::SmootherFactory<SC, LO, GO, NO>( coarseSolverPrototype, Teuchos::null ) );
}

Teuchos::RCP<MueLu::SmootherFactory<SC, LO, GO, NO>>
TrilinosMueLuSolver::getSmootherFactory( const int level )
{
    std::string ifpackType;
    Teuchos::RCP<MueLu::SmootherPrototype<SC, LO, GO, NO>> smootherPrototype;

    // const auto &mueLuLevel = d_mueluHierarchy->GetLevel( level );
    //    const auto A           = mueLuLevel->Get<Teuchos::RCP<Xpetra::Operator<SC, LO, GO, NO>>>(
    //    "A" );

    auto &smootherParams = getSmootherParameters( level );

    if ( d_smoother_type == "TrilinosSmoother" ) {

        ifpackType        = "RELAXATION";
        smootherPrototype = Teuchos::rcp(
            new MueLu::TrilinosSmoother<SC, LO, GO, NO>( ifpackType, smootherParams ) );

    } else if ( d_smoother_type == "IfpackSmoother" ) {

        // legacy interface to Ifpack
        ifpackType = "point relaxation stand-alone";
        smootherPrototype =
            Teuchos::rcp( new MueLu::IfpackSmoother<NO>( ifpackType, smootherParams ) );

    } else if ( d_smoother_type == "Ifpack2Smoother" ) {

        ifpackType        = "point relaxation stand-alone";
        smootherPrototype = Teuchos::rcp(
            new MueLu::Ifpack2Smoother<SC, LO, GO, NO>( ifpackType, smootherParams ) );
    }

    smootherPrototype->SetDefaultVerbLevel( getVerbosityLevel() );
    return Teuchos::rcp( new MueLu::SmootherFactory<SC, LO, GO, NO>( smootherPrototype ) );
}

Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> TrilinosMueLuSolver::getXpetraMatrix()
{
    // wrap in a Xpetra matrix
    auto epetraMatrixData =
        AMP::LinearAlgebra::EpetraMatrixData::createView( d_matrix->getMatrixData() );
    auto epA = Teuchos::rcpFromRef( epetraMatrixData->getEpetra_CrsMatrix() );
    Teuchos::RCP<Xpetra::CrsMatrix<SC, LO, GO, NO>> exA =
        Teuchos::rcp( new Xpetra::EpetraCrsMatrixT<GO, NO>( epA ) );
    auto crsWrapMat = Teuchos::rcp( new Xpetra::CrsMatrixWrap<SC, LO, GO, NO>( exA ) );
    auto xA         = Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>( crsWrapMat );

    return xA;
}

Teuchos::ParameterList &TrilinosMueLuSolver::getSmootherParameters( const int )
{
    auto &smootherParams = d_MueLuParameterList.get<Teuchos::ParameterList>( "smoother: params" );

#if 0
    // BP: will uncomment once I remember what this does!!
    if ( d_smoother_type == "IfpackSmoother" ) {

        if ( d_construct_partition ) {

            const auto &mueLuLevel = d_mueluHierarchy->GetLevel( level );
	    const auto A = mueLuLevel->Get<Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>>>( "A" );
        }
    }
#endif
    return smootherParams;
}

void TrilinosMueLuSolver::buildHierarchyFromDefaults( void )
{

    // Transfer operators
    auto TentativePFact = Teuchos::rcp( new MueLu::TentativePFactory<SC, LO, GO, NO>() );
    auto SaPFact        = Teuchos::rcp( new MueLu::SaPFactory<SC, LO, GO, NO>() );
    auto RFact          = Teuchos::rcp( new MueLu::TransPFactory<SC, LO, GO, NO>() );
    // coarsest solver.
    auto coarseSolverPrototype = Teuchos::rcp( new MueLu::DirectSolver<SC, LO, GO, NO>() );
    auto coarseSolverFact      = Teuchos::rcp(
        new MueLu::SmootherFactory<SC, LO, GO, NO>( coarseSolverPrototype, Teuchos::null ) );

    d_factoryManager.SetFactory( "Ptent", TentativePFact );
    d_factoryManager.SetFactory( "P", SaPFact );
    d_factoryManager.SetFactory( "R", RFact );
    //    d_factoryManager.SetFactory("Smoother", Teuchos::null);      //skips smoother setup
    d_factoryManager.SetFactory( "CoarseSolver", coarseSolverFact );

    // extract the Xpetra matrix from AMP
    auto fineLevelA = getXpetraMatrix();

    d_mueluHierarchy = Teuchos::rcp( new MueLu::Hierarchy<SC, LO, GO, NO>() );
    d_mueluHierarchy->SetDefaultVerbLevel( getVerbosityLevel() );
    auto finestMGLevel = d_mueluHierarchy->GetLevel();
    finestMGLevel->Set( "A", fineLevelA );

    const int startLevel = 0;
    const int maxLevels  = 10;
    d_mueluHierarchy->Setup( d_factoryManager, startLevel, maxLevels );
}

void TrilinosMueLuSolver::buildHierarchyByLevel( void )
{

    d_mueluHierarchyManager =
        Teuchos::rcp( new MueLu::ParameterListInterpreter<SC, LO, GO, NO>( d_MueLuParameterList ) );

    d_mueluHierarchy = d_mueluHierarchyManager->CreateHierarchy();
    // forced to set Max coarse size explicitly, as there does not seem to be any other sane way of
    // doing it

    d_mueluHierarchy->SetMaxCoarseSize( d_MueLuParameterList.get<int>( "coarse: max size" ) );
    //    d_mueluHierarchyManager->SetupHierarchy( *d_mueluHierarchy);

    d_mueluHierarchy->SetDefaultVerbLevel( getVerbosityLevel() );
    auto finestMGLevel = d_mueluHierarchy->GetLevel();
    // extract the Xpetra matrix from AMP
    auto fineLevelA = getXpetraMatrix();
    finestMGLevel->Set( "A", fineLevelA );

    d_levelFactoryManager.resize( d_maxLevels );

    for ( size_t i = 0u; i < d_maxLevels; ++i ) {

        d_levelFactoryManager[i] = Teuchos::rcp( new MueLu::FactoryManager<SC, LO, GO, NO> );
        d_levelFactoryManager[i]->SetFactory( "Ptent", getTentativePFactory() );
        d_levelFactoryManager[i]->SetFactory( "P", getSaPFactory() );
        d_levelFactoryManager[i]->SetFactory( "R", getRFactory() );
    }

    // seeing if smoothers can be initialized for each level
    for ( size_t i = 0u; i < d_maxLevels; ++i ) {

        auto finerLevelManager   = ( i == 0u ) ? Teuchos::null : d_levelFactoryManager[i - 1];
        auto currentLevelManager = d_levelFactoryManager[i];
        auto coarserLevelManager =
            ( i < d_maxLevels - 1 ) ? d_levelFactoryManager[i + 1] : Teuchos::null;

        // setup a level
        bool bIsLastLevel = d_mueluHierarchy->Setup(
            i, finerLevelManager, currentLevelManager, coarserLevelManager );

        auto currentLevel = d_mueluHierarchy->GetLevel( i );

        // keep some of the factories
        auto PtentFactory = currentLevelManager->GetFactory( "Ptent" );
        currentLevel->Keep( "Ptent", PtentFactory.get() );

        auto PFactory = currentLevelManager->GetFactory( "P" );
        currentLevel->Keep( "P", PFactory.get() );

        auto RFactory = currentLevelManager->GetFactory( "R" );
        currentLevel->Keep( "R", RFactory.get() );

        currentLevelManager->SetFactory( "Smoother", getSmootherFactory( i ) );
        currentLevelManager->SetFactory( "CoarseSolver", getCoarseSolverFactory() );

        coarserLevelManager = ( bIsLastLevel ) ? Teuchos::null : d_levelFactoryManager[i + 1];

        // setup a level
        bIsLastLevel = d_mueluHierarchy->Setup(
            i, finerLevelManager, currentLevelManager, coarserLevelManager );

        if ( bIsLastLevel )
            break;
    }
}

void TrilinosMueLuSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    d_bRobustMode = db->getWithDefault<bool>( "ROBUST_MODE", false );
    d_bUseEpetra  = db->getWithDefault<bool>( "USE_EPETRA", true );

    d_build_hierarchy = db->getWithDefault<bool>( "build_hierarchy", true );
    d_build_hierarchy_from_defaults =
        db->getWithDefault<bool>( "build_hierarchy_from_defaults", false );

    // general parameters
    d_MueLuParameterList.set( "verbosity", db->getWithDefault<std::string>( "verbosity", "none" ) );
    d_MueLuParameterList.set( "problem: type",
                              db->getWithDefault<std::string>( "problem_type", "unknown" ) );
    d_MueLuParameterList.set( "number of equations",
                              db->getWithDefault<int>( "number_of_equations", 1 ) );
    d_maxLevels = db->getWithDefault<int>( "max_levels", 10 );
    d_MueLuParameterList.set( "max levels", (int) d_maxLevels );
    d_MueLuParameterList.set( "cycle type", db->getWithDefault<std::string>( "cycle_type", "V" ) );
    d_MueLuParameterList.set( "problem: symmetric",
                              db->getWithDefault<bool>( "problem_symmetric", false ) );
    d_MueLuParameterList.set( "xml parameter file",
                              db->getWithDefault<std::string>( "xml_parameter_file", "" ) );

    // smoothing and coarse solver options
    d_smoother_type = db->getWithDefault<std::string>( "smoother_type", "IfpackSmoother" );

    d_MueLuParameterList.set( "smoother: pre or post",
                              db->getWithDefault<std::string>( "smoother_pre_or_post", "both" ) );
    d_MueLuParameterList.set(
        "smoother: pre type",
        db->getWithDefault<std::string>( "smoother_pre_type", "RELAXATION" ) );
    d_MueLuParameterList.set(
        "smoother: post type",
        db->getWithDefault<std::string>( "smoother_post_type", "RELAXATION" ) );
    d_MueLuParameterList.set( "smoother: overlap",
                              db->getWithDefault<int>( "smoother_overlap", 0 ) );
    d_MueLuParameterList.set( "smoother: pre overlap",
                              db->getWithDefault<int>( "smoother_pre_overlap", 0 ) );
    d_MueLuParameterList.set( "smoother: post overlap",
                              db->getWithDefault<int>( "smoother_post_overlap", 0 ) );

    Teuchos::ParameterList relaxationParams;

    if ( db->keyExists( "smoother_params" ) ) {

        const auto &smoother_db = db->getDatabase( "smoother_params" );

        relaxationParams.set(
            "relaxation: type",
            smoother_db->getWithDefault<std::string>( "relaxation_type", "Gauss-Seidel" ) );
        relaxationParams.set( "relaxation: sweeps",
                              smoother_db->getWithDefault<int>( "relaxation_sweeps", 1 ) );
        relaxationParams.set(
            "relaxation: damping factor",
            smoother_db->getWithDefault<double>( "relaxation_damping_factor", 1.0 ) );

        if ( smoother_db->keyExists( "relaxation_zero_starting_solution" ) ) {
            relaxationParams.set(
                "relaxation: zero starting solution",
                smoother_db->getScalar<bool>( "relaxation_zero_starting_solution" ) );
        }

        if ( smoother_db->keyExists( "relaxation_backward_mode" ) ) {
            relaxationParams.set( "relaxation: backward mode",
                                  smoother_db->getScalar<bool>( "relaxation_backward_mode" ) );
        }

        if ( smoother_db->keyExists( "relaxation_use_l1" ) ) {
            relaxationParams.set( "relaxation: use l1",
                                  smoother_db->getScalar<bool>( "relaxation_use_l1" ) );
        }

        if ( smoother_db->keyExists( "relaxation_l1_eta" ) ) {
            relaxationParams.set( "relaxation: l1 eta",
                                  smoother_db->getScalar<double>( "relaxation_l1_eta" ) );
        }

        if ( smoother_db->keyExists( "relaxation_min_diagonal_value" ) ) {
            relaxationParams.set(
                "relaxation: min diagonal value",
                smoother_db->getScalar<double>( "relaxation_min_diagonal_value" ) );
        }

        if ( smoother_db->keyExists( "relaxation_fix_tiny_diagonal_entries" ) ) {
            relaxationParams.set(
                "relaxation: fix tiny diagonal entries",
                smoother_db->getScalar<bool>( "relaxation_fix_tiny_diagonal_entries" ) );
        }

        if ( smoother_db->keyExists( "relaxation_check_diagonal_entries" ) ) {
            relaxationParams.set(
                "relaxation: check diagonal entries",
                smoother_db->getScalar<bool>( "relaxation_check_diagonal_entries" ) );
        }
#if 0
        if ( smoother_db->keyExists("relaxation_local_smoothing_indices") ) { 
            relaxationParams.set( "relaxation: local smoothing indices", smoother_db->getWithDefault<int>("relaxation_local_smoothing_indices") );
        }
#endif
    }

    d_MueLuParameterList.set( "smoother: params", relaxationParams );

    d_MueLuParameterList.set( "coarse: max size",
                              db->getWithDefault<int>( "coarse_max_size", 48 ) );
    d_MueLuParameterList.set( "coarse: type",
                              db->getWithDefault<std::string>( "coarse_type", "SuperLU" ) );
    d_MueLuParameterList.set( "coarse: overlap", db->getWithDefault<int>( "coarse_overlap", 0 ) );

    // aggregation options: incomplete list
    d_MueLuParameterList.set( "aggregation: type",
                              db->getWithDefault<std::string>( "aggregation_type", "uncoupled" ) );
    d_MueLuParameterList.set(
        "aggregation: ordering",
        db->getWithDefault<std::string>( "aggregation_ordering", "natural" ) );
    d_MueLuParameterList.set(
        "aggregation: drop scheme",
        db->getWithDefault<std::string>( "aggregation_drop_scheme", "classical" ) );
    d_MueLuParameterList.set( "aggregation: drop tol",
                              db->getWithDefault<double>( "aggregation_drop_tol", 0.0 ) );
    d_MueLuParameterList.set( "aggregation: min agg size",
                              db->getWithDefault<int>( "aggregation_min_agg_size", 2 ) );
    d_MueLuParameterList.set( "aggregation: max agg size",
                              db->getWithDefault<int>( "aggregation_max_agg_size", 9 ) );
    d_MueLuParameterList.set( "aggregation: brick x size",
                              db->getWithDefault<int>( "aggregation_brick_x_size", 2 ) );
    d_MueLuParameterList.set( "aggregation: brick y size",
                              db->getWithDefault<int>( "aggregation_brick_y_size", 2 ) );
    d_MueLuParameterList.set( "aggregation: brick z size",
                              db->getWithDefault<int>( "aggregation_brick_z_size", 2 ) );

    // rebalance options
    d_MueLuParameterList.set( "repartition: enable",
                              db->getWithDefault<bool>( "repartition_enable", false ) );
    d_MueLuParameterList.set(
        "repartition: partitioner",
        db->getWithDefault<std::string>( "repartition_partitioner", "zoltan2" ) );
    d_MueLuParameterList.set( "repartition: start level",
                              db->getWithDefault<int>( "repartition_start_level", 2 ) );
    d_MueLuParameterList.set( "repartition: min rows per proc",
                              db->getWithDefault<int>( "repartition_min_rows_per_proc", 50 ) );
    d_MueLuParameterList.set( "repartition: max imbalance",
                              db->getWithDefault<double>( "repartition_max_imbalance", 1.2 ) );
    d_MueLuParameterList.set( "repartition: remap parts",
                              db->getWithDefault<bool>( "repartition_remap_parts", true ) );
    d_MueLuParameterList.set( "repartition: rebalance P and R",
                              db->getWithDefault<bool>( "repartition_P_and_R", false ) );

    // mg algorithm options, incomplete list
    d_MueLuParameterList.set( "multigrid algorithm",
                              db->getWithDefault<std::string>( "multigrid_algorithm", "sa" ) );
    d_MueLuParameterList.set( "sa: damping factor",
                              db->getWithDefault<double>( "sa_damping_factor", 1.33 ) );

    // mg reuse options: none at present

    // miscellaneous options
    d_MueLuParameterList.set( "print initial parameters",
                              db->getWithDefault<bool>( "print_initial_parameters", false ) );
    d_MueLuParameterList.set( "print unused parameters",
                              db->getWithDefault<bool>( "print_unused_parameters", false ) );

    d_MueLuParameterList.set( "transpose: use implicit",
                              db->getWithDefault<bool>( "transpose_use_implicit", false ) );
}

void TrilinosMueLuSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    AMP_INSIST( d_pOperator, "ERROR: TrilinosMueLuSolver::initialize() operator cannot be NULL" );
    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );
    d_matrix = AMP::LinearAlgebra::getEpetraMatrix( linearOperator->getMatrix() );
    AMP_INSIST( d_matrix, "d_matrix cannot be NULL" );

    if ( d_bUseEpetra ) {

        if ( d_build_hierarchy ) {

            if ( d_build_hierarchy_from_defaults ) {

                buildHierarchyFromDefaults();

            } else {

                buildHierarchyByLevel();
            }

        } else {

            // MueLu expects a Teuchos ref pointer
            Teuchos::RCP<Epetra_CrsMatrix> fineLevelMatrixPtr =
                Teuchos::rcpFromRef( d_matrix->getEpetra_CrsMatrix() );
            auto mueluRCP =
                MueLu::CreateEpetraPreconditioner( fineLevelMatrixPtr, d_MueLuParameterList );

            d_mueluSolver.reset( mueluRCP.get() );
            mueluRCP.release();
        }

    } else {
        AMP_ERROR( "Only Epetra interface currently supported" );
    }
    d_bCreationPhase = false;
}


void TrilinosMueLuSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    PROFILE( "TrilinosMueLuSolver::resetOperator" );
    AMP_INSIST( ( d_pOperator ),
                "ERROR: TrilinosMueLuSolver::resetOperator() operator cannot be NULL" );
    d_pOperator->reset( params );
    reset( std::shared_ptr<SolverStrategyParameters>() );
}


void TrilinosMueLuSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{
    PROFILE( "TrilinosMueLuSolver::reset" );
    d_mueluSolver.reset();
    d_matrix.reset(); // Need to keep a copy of the matrix alive until after the solver is destroyed
    registerOperator( d_pOperator );
}

void TrilinosMueLuSolver::solveWithHierarchy( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                              std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "TrilinosMueLuSolver::solveWithHierarchy" );

    if ( d_bUseEpetra ) {


        // These functions throw exceptions if this cannot be performed.
        auto fView          = AMP::LinearAlgebra::EpetraVector::constView( f );
        auto uView          = AMP::LinearAlgebra::EpetraVector::view( u );
        Epetra_Vector &fVec = const_cast<Epetra_Vector &>( fView->getEpetra_Vector() );
        Epetra_Vector &uVec = uView->getEpetra_Vector();

        auto rcp_u = Teuchos::rcpFromRef( uVec );
        auto rcp_f = Teuchos::rcpFromRef( fVec );

        // Epetra -> Xpetra
        auto xu = Teuchos::rcp( new Xpetra::EpetraVectorT<int, NO>( rcp_u ) );
        auto xf = Teuchos::rcp( new Xpetra::EpetraVectorT<int, NO>( rcp_f ) );

        d_mueluHierarchy->Iterate( *xf, *xu, d_iMaxIterations );

    } else {
        AMP_ERROR( "Only Epetra interface currently supported" );
    }
}

void TrilinosMueLuSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "TrilinosMueLuSolver::solve" );

    AMP_ASSERT( f && u );

    // Check input vector states
    if ( u->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::UNCHANGED ) {
        u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: TrilinosMueLuSolver::apply() operator cannot be NULL" );

    auto r = f->clone();
    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        r->copyVector( f );
    } else {
        d_pOperator->residual( f, u, r );
    }

    d_dInitialResidual = d_dResidualNorm = r->L2Norm();
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TrilinosMueLuSolver::apply(), L2 norm of residual before solve "
                  << std::setprecision( 15 ) << d_dInitialResidual << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        const auto solution_norm = static_cast<double>( u->L2Norm() );
        AMP::pout << "TrilinosMueLuSolver : before solve solution norm: " << std::setprecision( 15 )
                  << solution_norm << std::endl;
    }

    if ( d_build_hierarchy ) {

        solveWithHierarchy( f, u );

    } else {

        if ( d_bCreationPhase ) {
            if ( d_bUseEpetra ) {
                // MueLu expects a Teuchos ref pointer
                auto fineLevelMatrixPtr = Teuchos::rcpFromRef( d_matrix->getEpetra_CrsMatrix() );
                auto mueluRCP =
                    MueLu::CreateEpetraPreconditioner( fineLevelMatrixPtr, d_MueLuParameterList );
                d_mueluSolver.reset( mueluRCP.get() );
                mueluRCP.release();
            } else {
                AMP_ERROR( "Only Epetra interface currently supported" );
            }
            d_bCreationPhase = false;
        }

        if ( d_bUseEpetra ) {
            // These functions throw exceptions if this cannot be performed
            auto fView                = AMP::LinearAlgebra::EpetraVector::constView( f );
            auto uView                = AMP::LinearAlgebra::EpetraVector::view( u );
            const Epetra_Vector &fVec = fView->getEpetra_Vector();
            Epetra_Vector &uVec       = uView->getEpetra_Vector();

            d_mueluSolver->ApplyInverse( fVec, uVec );
        } else {
            AMP_ERROR( "Only Epetra interface supported" );
        }
    }

    // Check for NaNs in the solution (no communication necessary)
    double localNorm = u->getVectorOperations()->localL2Norm( *u->getVectorData() ).get<double>();
    AMP_INSIST( localNorm == localNorm, "NaNs detected in solution" );

    // we are forced to update the state of u here
    // as Epetra is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->getVectorData()->fireDataChange();
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 2 ) {
        double solution_norm = static_cast<double>( u->L2Norm() );
        AMP::pout << "TrilinosMueLuSolver : after solve solution norm: " << std::setprecision( 24 )
                  << solution_norm << std::endl;
    }
    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "TrilinosMueLuSolver : after solve solution: " << std::setprecision( 24 )
                  << std::endl;
        AMP::pout << *u << std::endl;
    }

    d_pOperator->residual( f, u, r );
    d_dResidualNorm = r->L2Norm();
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TrilinosMueLuSolver::apply(), L2 norm of residual after solve "
                  << std::setprecision( 24 ) << d_dResidualNorm << std::endl;
    }

    if ( d_bRobustMode ) {
        if ( d_dResidualNorm > d_dInitialResidual ) {
            AMP::pout << "Warning: ML was not able to reduce the residual. Using LU instead."
                      << std::endl;
            reSolveWithLU( f, u );
        }
    }
}


void TrilinosMueLuSolver::reSolveWithLU( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "TrilinosMueLuSolver::reSolveWithLU" );

    if ( !d_bUseEpetra ) {
        AMP_ERROR( "Robust mode can only be used with Epetra matrices." );
    }

    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

    d_matrix = std::dynamic_pointer_cast<AMP::LinearAlgebra::ManagedEpetraMatrix>(
        linearOperator->getMatrix() );
    AMP_INSIST( d_matrix, "d_matrix cannot be NULL" );

    auto tmpMueLuParameterList = d_MueLuParameterList;
    tmpMueLuParameterList.set( "verbosity", "medium" );
    tmpMueLuParameterList.set( "max levels", 1 );
    tmpMueLuParameterList.set( "coarse: type", "SuperLU" );

    // MueLu expects a Teuchos ref pointer
    auto fineLevelMatrixPtr = Teuchos::rcpFromRef( d_matrix->getEpetra_CrsMatrix() );
    auto mueluRCP = MueLu::CreateEpetraPreconditioner( fineLevelMatrixPtr, tmpMueLuParameterList );

    d_mueluSolver.reset( mueluRCP.get() );
    mueluRCP.release();

    d_bCreationPhase = false;

    apply( f, u );

    mueluRCP = MueLu::CreateEpetraPreconditioner( fineLevelMatrixPtr, d_MueLuParameterList );

    d_mueluSolver.reset( mueluRCP.get() );
    mueluRCP.release();

    d_bCreationPhase = false;
}

MueLu::MsgType TrilinosMueLuSolver::getVerbosityLevel()
{
    MueLu::MsgType verbosityLevel = MueLu::MsgType::None;
    if ( d_iDebugPrintInfoLevel == 1 )
        verbosityLevel = MueLu::MsgType::Low;
    else if ( d_iDebugPrintInfoLevel == 2 )
        verbosityLevel = MueLu::MsgType::Medium;
    else if ( d_iDebugPrintInfoLevel == 3 )
        verbosityLevel = MueLu::MsgType::High;
    else if ( d_iDebugPrintInfoLevel >= 4 )
        verbosityLevel = MueLu::MsgType::Extreme;
    return verbosityLevel;
}

} // namespace Solver
} // namespace AMP
