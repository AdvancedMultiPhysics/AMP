#include "AMP/solvers/trilinos/nox/TrilinosNOXSolver.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/trilinos/nox/AndersonStatusTest.h"
#include "AMP/solvers/trilinos/thyra/TrilinosThyraModelEvaluator.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"

#include "ProfilerApp.h"


// Trilinos includes
DISABLE_WARNINGS

#ifdef AMP_USE_TRILINOS_BELOS
    #include "BelosTypes.hpp"
#endif
#include "NOX_MatrixFree_ModelEvaluatorDecorator.hpp"
#include "NOX_Solver_Factory.H"
#include "NOX_Solver_Generic.H"
#include "NOX_StatusTest_Combo.H"
#include "NOX_StatusTest_FiniteValue.H"
#include "NOX_StatusTest_MaxIters.H"
#include "NOX_StatusTest_NormF.H"
#include "NOX_StatusTest_NormUpdate.H"
#include "NOX_StatusTest_NormWRMS.H"
#include "NOX_StatusTest_RelativeNormF.H"
#include "NOX_Thyra.H"
#include "NOX_Thyra_Group.H"
#include "NOX_Thyra_MatrixFreeJacobianOperator.hpp"
#include "Stratimikos_DefaultLinearSolverBuilder.hpp"
#include "Teuchos_RefCountPtrDecl.hpp"
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
TrilinosNOXSolver::TrilinosNOXSolver() : SolverStrategy() {}
TrilinosNOXSolver::TrilinosNOXSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    auto params = std::dynamic_pointer_cast<TrilinosNOXSolverParameters>( parameters );
    AMP_ASSERT( params );
    initialize( params );
}
void TrilinosNOXSolver::reset( std::shared_ptr<SolverStrategyParameters> parameters )
{
    initialize( parameters );
}
TrilinosNOXSolver::~TrilinosNOXSolver() = default;


/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
void TrilinosNOXSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    if ( parameters ) {
        // Copy the parameters
        auto params = std::dynamic_pointer_cast<const TrilinosNOXSolverParameters>( parameters );
        AMP_ASSERT( params );
        AMP_ASSERT( params->d_db );
        d_comm = params->d_comm;
        if ( params->d_pInitialGuess )
            d_initialGuess = params->d_pInitialGuess;
        std::shared_ptr<AMP::Database> nonlinear_db = parameters->d_db;
        AMP_ASSERT( nonlinear_db );
        auto linear_solver_db_name =
            nonlinear_db->getWithDefault<std::string>( "linear_solver_name", "LinearSolver" );
        auto enclosing_db =
            nonlinear_db->keyExists( linear_solver_db_name ) ? nonlinear_db : d_global_db;

        std::shared_ptr<AMP::Database> linear_db;
        if ( enclosing_db )
            linear_db = enclosing_db->getDatabase( linear_solver_db_name );
        // Create a model evaluator
        auto modelParams           = std::make_shared<TrilinosThyraModelEvaluatorParameters>();
        modelParams->d_nonlinearOp = d_pOperator;
        AMP_WARNING(
            "TrilinosNOXSolver at present sets linear operator to nonlinear operator also" );
        modelParams->d_linearOp = d_pOperator;
        modelParams->d_icVec    = d_initialGuess;
        modelParams->d_preconditioner.reset();
        modelParams->d_prePostOperator = params->d_prePostOperator;
        if ( linear_db && linear_db->getWithDefault<bool>( "uses_preconditioner", false ) ) {
            if ( params->d_preconditioner ) {
                modelParams->d_preconditioner = params->d_preconditioner;
            } else {
                auto preconditionerName =
                    linear_db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
                auto pc_db                    = linear_db->keyExists( "pc_solver_name" ) ?
                                                    d_global_db->getDatabase( preconditionerName ) :
                                                    linear_db->getDatabase( preconditionerName );
                modelParams->d_preconditioner = createPreconditioner( pc_db );
            }
        }

        d_thyraModel = Teuchos::RCP<TrilinosThyraModelEvaluator>(
            new TrilinosThyraModelEvaluator( modelParams ) );
        // Create the Preconditioner operator
        d_precOp = d_thyraModel->create_W_prec();
        // Create the linear solver factory
        ::Stratimikos::DefaultLinearSolverBuilder builder;
        Teuchos::RCP<Teuchos::ParameterList> p( new Teuchos::ParameterList );
        std::string linearSolverType = linear_db->getString( "linearSolverType" );
        std::string linearSolver     = linear_db->getString( "linearSolver" );
        int maxLinearIterations      = linear_db->getWithDefault<int>( "max_iterations", 100 );
        auto linearRelativeTolerance =
            linear_db->getWithDefault<double>( "relative_tolerance", 1e-3 );
        bool flexGmres = linear_db->getWithDefault<bool>( "flexibleGmres", true );
        p->set( "Linear Solver Type", linearSolverType );
        p->set( "Preconditioner Type", "None" );
        p->sublist( "Linear Solver Types" )
            .sublist( linearSolverType )
            .set( "Solver Type", linearSolver );
        Teuchos::ParameterList &linearSolverParams =
            p->sublist( "Linear Solver Types" ).sublist( linearSolverType );
        linearSolverParams.sublist( "Solver Types" )
            .sublist( linearSolver )
            .set( "Maximum Iterations", maxLinearIterations );
        // Only "Block GMRES" recognizes the "Flexible Gmres" option, other solvers may throw an
        // input validation error
        if ( linearSolver == "Block GMRES" )
            linearSolverParams.sublist( "Solver Types" )
                .sublist( linearSolver )
                .set( "Flexible Gmres", flexGmres );
        if ( linear_db->getWithDefault<int>( "print_info_level", 0 ) >= 2 ) {
            linearSolverParams.sublist( "Solver Types" )
                .sublist( linearSolver )
                .set( "Output Frequency", 1 );
            linearSolverParams.sublist( "Solver Types" )
                .sublist( linearSolver )
                .set( "Verbosity", 10 );
            linearSolverParams.sublist( "VerboseObject" ).set( "Verbosity Level", "extreme" );
            if ( linearSolverType == "Belos" ) {
                linearSolverParams.sublist( "Solver Types" )
                    .sublist( linearSolver )
                    .set( "Verbosity",
                          Belos::Warnings + Belos::IterationDetails + Belos::OrthoDetails +
                              Belos::FinalSummary + Belos::Debug + Belos::StatusTestDetails );
            }
        } else if ( linear_db->getWithDefault<int>( "print_info_level", 0 ) >= 1 ) {
            linearSolverParams.sublist( "Solver Types" )
                .sublist( linearSolver )
                .set( "Output Frequency", 1 );
            linearSolverParams.sublist( "Solver Types" )
                .sublist( linearSolver )
                .set( "Verbosity", 10 );
            if ( linearSolverType == "Belos" ) {
                linearSolverParams.sublist( "Solver Types" )
                    .sublist( linearSolver )
                    .set( "Verbosity",
                          Belos::Warnings + Belos::IterationDetails + Belos::FinalSummary +
                              Belos::Debug );
            }
        }
        builder.setParameterList( p );
        d_lowsFactory = builder.createLinearSolveStrategy( "" );
        // d_lowsFactory->initializeVerboseObjectBase();
        d_thyraModel->set_W_factory( d_lowsFactory );
        // Create the convergence tests (these will need to be on the input database)
        Teuchos::RCP<NOX::StatusTest::NormF> absresid(
            new NOX::StatusTest::NormF( static_cast<double>( d_dAbsoluteTolerance ) ) );
        Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters(
            new NOX::StatusTest::MaxIters( d_iMaxIterations ) );
        Teuchos::RCP<NOX::StatusTest::FiniteValue> fv( new NOX::StatusTest::FiniteValue );
        Teuchos::RCP<NOX::StatusTest::NormWRMS> wrms(
            new NOX::StatusTest::NormWRMS( static_cast<double>( d_dAbsoluteTolerance ),
                                           static_cast<double>( d_dAbsoluteTolerance ) ) );
        d_status = Teuchos::rcp( new NOX::StatusTest::Combo( NOX::StatusTest::Combo::OR ) );
        d_status->addStatusTest( fv );
        d_status->addStatusTest( absresid );
        d_status->addStatusTest( maxiters );
        d_status->addStatusTest( wrms );
        // Create nox parameter list
        d_nlParams             = Teuchos::rcp( new Teuchos::ParameterList );
        std::string solverType = nonlinear_db->getString( "solver" );
        if ( solverType == "JFNK" ) {
            d_nlParams->set( "Nonlinear Solver", "Line Search Based" );
        } else if ( solverType == "Anderson" ) {
            d_nlParams->set( "Nonlinear Solver", "Anderson Accelerated Fixed-Point" );
            int depth   = nonlinear_db->getWithDefault<int>( "StorageDepth", 5 );
            auto mixing = nonlinear_db->getWithDefault<double>( "MixingParameter", 1.0 );
            d_nlParams->sublist( "Anderson Parameters" ).set( "Storage Depth", depth );
            d_nlParams->sublist( "Anderson Parameters" ).set( "Mixing Parameter", mixing );
            d_nlParams->sublist( "Anderson Parameters" )
                .sublist( "Preconditioning" )
                .set( "Precondition", d_precOp );
            Teuchos::RCP<NOX::StatusTest::RelativeNormF> relresid(
                new NOX::StatusTest::RelativeNormF( static_cast<double>( d_dRelativeTolerance ) ) );
            d_status->addStatusTest( relresid );
            Teuchos::RCP<AndersonStatusTest> andersonTest(
                new AMP::Solver::AndersonStatusTest( nonlinear_db ) );
            d_status->addStatusTest( andersonTest );
        }
        auto lineSearchMethod =
            nonlinear_db->getWithDefault<std::string>( "lineSearchMethod", "Polynomial" );
        d_nlParams->sublist( "Line Search" ).set( "Method", lineSearchMethod );
        d_nlParams->sublist( "Direction" )
            .sublist( "Newton" )
            .sublist( "Linear Solver" )
            .set( "Tolerance", linearRelativeTolerance );
        if ( params->d_prePostOperator ) {
            Teuchos::RefCountPtr<NOX::Abstract::PrePostOperator> prePostOperator(
                params->d_prePostOperator.get(),
                Teuchos::DeallocDelete<NOX::Abstract::PrePostOperator>(),
                false );
            d_nlParams->sublist( "Solver Options" )
                .set<Teuchos::RCP<NOX::Abstract::PrePostOperator>>(
                    "User Defined Pre/Post Operator", prePostOperator );
        }
        // Set the printing parameters in the "Printing" sublist
        Teuchos::ParameterList &printParams = d_nlParams->sublist( "Printing" );
        printParams.set( "Output Precision", 3 );
        printParams.set( "Output Processor", 0 );
        NOX::Utils::MsgType print_level = NOX::Utils::Error;
        if ( d_iDebugPrintInfoLevel >= 1 ) {
            print_level = static_cast<NOX::Utils::MsgType>(
                print_level + NOX::Utils::OuterIteration + NOX::Utils::OuterIterationStatusTest +
                NOX::Utils::InnerIteration + NOX::Utils::Warning );
        } else if ( d_iDebugPrintInfoLevel >= 2 ) {
            print_level = static_cast<NOX::Utils::MsgType>(
                print_level + NOX::Utils::LinearSolverDetails + NOX::Utils::Parameters +
                NOX::Utils::Details + NOX::Utils::Debug + NOX::Utils::TestDetails +
                NOX::Utils::Error );
        }
        printParams.set( "Output Information", print_level );
    }
}


/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
void TrilinosNOXSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "apply" );

    if ( d_bUseZeroInitialGuess ) {
        u->zero();
    }
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    std::shared_ptr<AMP::LinearAlgebra::Vector> rhs;
    if ( f ) {
        rhs = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( f );
    } else {
        rhs = u->clone();
        rhs->zero();
    }
    // Get thyra vectors
    if ( !d_initialGuess )
        d_initialGuess = u;
    auto initial = std::dynamic_pointer_cast<AMP::LinearAlgebra::ThyraVector>(
        AMP::LinearAlgebra::ThyraVector::view( d_initialGuess ) );
    auto U = std::dynamic_pointer_cast<AMP::LinearAlgebra::ThyraVector>(
        AMP::LinearAlgebra::ThyraVector::view( u ) );

    // auto F = std::dynamic_pointer_cast<const AMP::LinearAlgebra::ThyraVector>(
    //     AMP::LinearAlgebra::ThyraVector::constView( f ) );

    //  Set the rhs for the thyra model
    d_thyraModel->setRhs( rhs );

    // Create the JFNK operator
    Teuchos::ParameterList printParams;
    auto jfnkParams = Teuchos::parameterList();
    jfnkParams->set( "Difference Type", "Forward" );
    jfnkParams->set( "Perturbation Algorithm", "KSP NOX 2001" );
    jfnkParams->set( "lambda", 1.0e-4 );
    Teuchos::RCP<NOX::Thyra::MatrixFreeJacobianOperator<double>> jfnkOp(
        new NOX::Thyra::MatrixFreeJacobianOperator<double>( printParams ) );
    jfnkOp->setParameterList( jfnkParams );
    if ( d_iDebugPrintInfoLevel >= 3 && d_comm.getRank() == 0 )
        jfnkParams->print( AMP::pout );
    // Create the NOX::Thyra::Group
    // Teuchos::RCP<NOX::Thyra::Group> nox_group( new NOX::Thyra::Group( initial->getVec(),
    // d_thyraModel ) );
    Teuchos::RCP<::Thyra::ModelEvaluator<double>> thyraModel(
        new NOX::MatrixFreeModelEvaluatorDecorator<double>( d_thyraModel ) );
    NOX::Thyra::Vector initialGuess( initial->getVec() );
    Teuchos::RCP<NOX::Thyra::Group> nox_group( new NOX::Thyra::Group( initialGuess,
                                                                      thyraModel,
                                                                      jfnkOp,
                                                                      d_lowsFactory,
                                                                      d_precOp,
                                                                      Teuchos::null,
                                                                      Teuchos::null,
                                                                      Teuchos::null,
                                                                      Teuchos::null ) );
    nox_group->setX( U->getVec() );
    nox_group->computeF();
    // VERY IMPORTANT!!!  jfnk object needs base evaluation objects.
    // This creates a circular dependency, so use a weak pointer.
    jfnkOp->setBaseEvaluationToNOXGroup( nox_group.create_weak() );
    // Create the solver
    d_solver = NOX::Solver::buildSolver( nox_group, d_status, d_nlParams );
    // Solve
    d_nlParams->print( AMP::pout );
    NOX::StatusTest::StatusType solvStatus = d_solver->solve();
    if ( solvStatus != NOX::StatusTest::Converged )
        AMP_ERROR( "Failed to solve" );
    // Copy the solution back to u
    const auto *tmp = dynamic_cast<const NOX::Thyra::Vector *>( &( nox_group->getX() ) );
    const auto *thyraVec =
        dynamic_cast<const AMP::LinearAlgebra::ThyraVectorWrapper *>( &( tmp->getThyraVector() ) );
    AMP_ASSERT( thyraVec );
    AMP_ASSERT( thyraVec->numVecs() == 1 );
    u->copyVector( thyraVec->getVec( 0 ) );
}

// This routine or almost identical code is now present in NKA, SNESSolver and here. This needs to
// move into a base class
std::shared_ptr<SolverStrategy>
TrilinosNOXSolver::createPreconditioner( std::shared_ptr<AMP::Database> pc_solver_db )
{
    AMP_INSIST(
        pc_solver_db,
        "TrilinosNOXSolver::createPreconditioner: Database object for preconditioner is NULL" );
    std::shared_ptr<SolverStrategy> preconditionerSolver;
    auto pcSolverParameters =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( pc_solver_db );
    if ( d_pOperator ) {
        // check if this should be passed the initial guess or a solution vector that can be reset
        auto pc_params = d_pOperator->getParameters( "Jacobian", d_initialGuess );
        std::shared_ptr<AMP::Operator::Operator> pcOperator =
            AMP::Operator::OperatorFactory::create( pc_params );
        pcSolverParameters->d_pOperator = pcOperator;
    }

    preconditionerSolver = AMP::Solver::SolverFactory::create( pcSolverParameters );

    return preconditionerSolver;
}

} // namespace AMP::Solver
