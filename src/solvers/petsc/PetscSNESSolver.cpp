#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/petsc/PetscMatrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/petsc/PetscVector.h"

#include "ProfilerApp.h"

#include "petsc/private/snesimpl.h"
#include "petsc/private/vecimpl.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscversion.h"

#include <numeric>

namespace AMP::Solver {


static_assert( PETSC_VERSION_GE( 3, 15, 0 ), "AMP only supports PETSc 3.15.0 or greater" );

#if PETSC_VERSION_LT( 3, 17, 0 )
    #define SETERRQ SETERRQ1
    #define PetscInfo PetscInfo3
#endif


static inline void checkErr( PetscErrorCode ierr )
{
    AMP_INSIST( ierr == 0, "Petsc returned non-zero error code" );
}


/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
PetscSNESSolver::PetscSNESSolver() {}
PetscSNESSolver::PetscSNESSolver( std::shared_ptr<SolverStrategyParameters> params )
    : SolverStrategy( params )
{
    d_sName         = "PetscSNESSolver";
    auto parameters = std::dynamic_pointer_cast<const SolverStrategyParameters>( params );

    if ( parameters ) {
        d_pKrylovSolver =
            std::dynamic_pointer_cast<PetscKrylovSolver>( parameters->d_pNestedSolver );
        d_pSolutionVector = parameters->d_pInitialGuess;
    }

    if ( !parameters->d_comm.isNull() ) {
        d_comm = parameters->d_comm;
    } else if ( d_pOperator && d_pOperator->getMesh() ) {
        d_comm = d_pOperator->getMesh()->getComm();
    } else {
        d_comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    }

    getFromInput( params->d_db );
    createPetscObjects( params );
    initializePetscObjects();
}


/****************************************************************
 *  De-constructor                                               *
 ****************************************************************/
PetscSNESSolver::~PetscSNESSolver() { destroyPetscObjects(); }

void PetscSNESSolver::destroyPetscObjects( void )
{
    // when we are using Matrix free delete the MF PETSc Jacobian
    if ( ( !d_bUsesJacobian ) && ( d_Jacobian != nullptr ) ) {
        PETSC::matDestroy( &d_Jacobian );
        d_Jacobian = nullptr;
    }
    SNESMonitorCancel( d_SNESSolver );
    SNESDestroy( &d_SNESSolver );
    d_SNESSolver                 = nullptr;
    d_bPetscInterfaceInitialized = false;
}

void PetscSNESSolver::createPetscObjects(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    checkErr( SNESCreate( d_comm.getCommunicator(), &d_SNESSolver ) );
    bool snes_create_pc = false;
    std::string preconditionerName;

    if ( d_pKrylovSolver ) {
        SNESSetKSP( d_SNESSolver, d_pKrylovSolver->getKrylovSolver() );
    } else {
        // initialize the Krylov solver correctly
        // access the SNES internal pointer to KSP and get a pointer to KSP
        auto nonlinearSolverDB = parameters->d_db;
        std::shared_ptr<AMP::Database> linearSolverDB;

        if ( nonlinearSolverDB->keyExists( "LinearSolver" ) ) {
            linearSolverDB = nonlinearSolverDB->getDatabase( "LinearSolver" );
        } else if ( nonlinearSolverDB->keyExists( "linear_solver_name" ) ) {
            const auto name = nonlinearSolverDB->getScalar<std::string>( "linear_solver_name" );
            AMP_ASSERT( d_global_db && d_global_db->keyExists( name ) );
            linearSolverDB = d_global_db->getDatabase( name );
        } else {
            // create a default Krylov solver DB and use a Jacobian free method
            // Note that sometimes a SNES solver database will directly specify options
            // for the Krylov solver and preconditioner and so we check for that. This
            // is how the AMR tests all work at present.
            linearSolverDB = std::make_shared<AMP::Database>( "LinearSolver" );
            linearSolverDB->putScalar<std::string>( "name", "PetscKrylovSolver" );
            linearSolverDB->putScalar<int>( "print_info_level", d_iDebugPrintInfoLevel );

            std::string linear_solver_type = "fgmres";

            if ( nonlinearSolverDB->keyExists( "ksp_type" ) ||
                 nonlinearSolverDB->keyExists( "linear_solver_type" ) ) {

                linear_solver_type =
                    nonlinearSolverDB->keyExists( "ksp_type" ) ?
                        nonlinearSolverDB->getScalar<std::string>( "ksp_type" ) :
                        nonlinearSolverDB->getScalar<std::string>( "linear_solver_type" );
            }

            linearSolverDB->putScalar<std::string>( "ksp_type", linear_solver_type );

            const auto max_krylov_dim =
                nonlinearSolverDB->getWithDefault<int>( "max_krylov_dimension", 25 );
            linearSolverDB->putScalar<int>( "max_krylov_dimension", max_krylov_dim );
            const auto maximum_linear_iterations =
                nonlinearSolverDB->getWithDefault<int>( "maximum_linear_iterations", 25 );
            linearSolverDB->putScalar<int>( "maximum_linear_iterations",
                                            maximum_linear_iterations );
            const auto uses_preconditioner =
                nonlinearSolverDB->getWithDefault<bool>( "uses_preconditioner", false );
            linearSolverDB->putScalar<bool>( "uses_preconditioner", uses_preconditioner );

            std::string pc_type = "none";

            if ( uses_preconditioner ) {

                // for now restrict to shell pc's
                pc_type = "shell";

                if ( nonlinearSolverDB->keyExists( "pc_solver_name" ) ) {
                    preconditionerName =
                        nonlinearSolverDB->getScalar<std::string>( "pc_solver_name" );
                    linearSolverDB->putScalar<std::string>( "pc_solver_name", preconditionerName );
                    // set the preconditioner type to be shell if a pc solver name is given
                    snes_create_pc = true;
                }

                auto pc_side = nonlinearSolverDB->getWithDefault<std::string>( "pc_side", "RIGHT" );
                linearSolverDB->putScalar<std::string>( "pc_side", pc_side );
            }

            pc_type = nonlinearSolverDB->getWithDefault<std::string>( "pc_type", pc_type );
            linearSolverDB->putScalar<std::string>( "pc_type", pc_type );
        }

        if ( !d_bUsesJacobian ) {
            // for Jacobian free methods  the SNES solver should create the
            // preconditioner even if the parameters are specified in the
            // linear solver database by the user
            const auto uses_preconditioner =
                linearSolverDB->getWithDefault<bool>( "uses_preconditioner", false );
            // check if the linearSolverDB specifies the name of the preconditioner
            // else assume the preconditioner will be created externally and set
            // later
            if ( uses_preconditioner ) {
                snes_create_pc = linearSolverDB->keyExists( "pc_solver_name" ) ||
                                 linearSolverDB->keyExists( "Preconditioner" );
            }
            linearSolverDB->putScalar<bool>( "matrix_free", true );
        }

        std::shared_ptr<SolverStrategy> preconditionerSolver;

        if ( snes_create_pc ) {

            preconditionerName =
                linearSolverDB->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
            auto pc_db           = linearSolverDB->keyExists( "pc_solver_name" ) ?
                                       d_global_db->getDatabase( preconditionerName ) :
                                       linearSolverDB->getDatabase( preconditionerName );
            preconditionerSolver = createPreconditioner( pc_db );
        }
        AMP_ASSERT( linearSolverDB );
        auto linearSolverParams    = std::make_shared<SolverStrategyParameters>( linearSolverDB );
        linearSolverParams->d_comm = d_comm;
        linearSolverParams->d_global_db     = d_global_db;
        linearSolverParams->d_pNestedSolver = preconditionerSolver;
        std::shared_ptr<SolverStrategy> linearSolver =
            AMP::Solver::SolverFactory::create( linearSolverParams );
        d_pKrylovSolver = std::dynamic_pointer_cast<PetscKrylovSolver>( linearSolver );
        AMP_ASSERT( d_pKrylovSolver );
        SNESSetKSP( d_SNESSolver, d_pKrylovSolver->getKrylovSolver() );
    }
}

void PetscSNESSolver::initializePetscObjects()
{
    checkErr( SNESSetApplicationContext( d_SNESSolver, this ) );

    // set the type to line search, potentially modify this later to be from input
    checkErr( SNESSetType( d_SNESSolver, SNESNEWTONLS ) );
    // set the pointer to the linesearch function
    if ( d_bEnableLineSearchPreCheck ) {

        auto fnPtr = std::bind( &AMP::Solver::PetscSNESSolver::defaultLineSearchPreCheck,
                                this,
                                std::placeholders::_1,
                                std::placeholders::_2,
                                std::placeholders::_3 );
        d_lineSearchPreCheckPtr = fnPtr;

        SNESLineSearch snesLineSearch;
        SNESGetLineSearch( d_SNESSolver, &snesLineSearch );
        checkErr( SNESLineSearchSetPreCheck(
            snesLineSearch, PetscSNESSolver::wrapperLineSearchPreCheck, this ) );
    }

    KSP kspSolver;
    SNESGetKSP( d_SNESSolver, &kspSolver );
    checkErr( KSPSetPreSolve( kspSolver,
                              (PetscErrorCode( * )( KSP, Vec, Vec, void * )) KSPPreSolve_SNESEW,
                              d_SNESSolver ) );
    checkErr( KSPSetPostSolve( kspSolver,
                               (PetscErrorCode( * )( KSP, Vec, Vec, void * )) KSPPostSolve_SNESEW,
                               d_SNESSolver ) );

    // If JFNK is being employed no operator is registered with the Krylov solver
    // and so the setup and apply of the preconditioner should be taken care of by PetscSNESSolver
    if ( ( !d_bUsesJacobian ) && ( d_pKrylovSolver->usesPreconditioner() ) ) {
        PC pc_handle;
        checkErr( KSPGetPC( kspSolver, &pc_handle ) );
        checkErr( PCShellSetContext( pc_handle, this ) );
        checkErr( PCShellSetSetUp( pc_handle, PetscSNESSolver::setupPreconditioner ) );
        checkErr( PCShellSetApply( pc_handle, PetscSNESSolver::applyPreconditioner ) );
    }

    checkErr( SNESSetTolerances( d_SNESSolver,
                                 static_cast<PetscReal>( d_dAbsoluteTolerance ),
                                 static_cast<PetscReal>( d_dRelativeTolerance ),
                                 static_cast<PetscReal>( d_dStepTolerance ),
                                 d_iMaxIterations,
                                 d_iMaximumFunctionEvals ) );

    // set the convergence criteria for the Krylov solver
    if ( !( d_sForcingTermStrategy == "CONSTANT" ) ) {

        checkErr( SNESKSPSetUseEW( d_SNESSolver, PETSC_TRUE ) );
        checkErr( SNESKSPSetParametersEW( d_SNESSolver,
                                          d_iForcingTermFlag,
                                          d_dInitialForcingTerm,
                                          d_dMaximumForcingTerm,
                                          d_dEWChoice2Gamma,
                                          d_dEWChoice2Alpha,
                                          d_dEWSafeguardExponent,
                                          d_dEWSafeguardDisableThreshold ) );
    } else {

        //        checkErr( KSPSetTolerances( d_pKrylovSolver->getKrylovSolver(),
        checkErr( KSPSetTolerances(
            kspSolver, d_dConstantForcingTerm, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT ) );
    }

    if ( d_SNESAppendOptionsPrefix != "" )
        SNESAppendOptionsPrefix( d_SNESSolver, d_SNESAppendOptionsPrefix.c_str() );

    checkErr( SNESSetFromOptions( d_SNESSolver ) );

    if ( d_bPrintNonlinearResiduals ) {
        PetscViewerAndFormat *vf;
        checkErr(
            PetscViewerAndFormatCreate( PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf ) );
        checkErr( SNESMonitorSet(
            d_SNESSolver,
            (PetscErrorCode( * )( SNES, PetscInt, PetscReal, void * )) SNESMonitorDefault,
            vf,
            (PetscErrorCode( * )( void ** )) PetscViewerAndFormatDestroy ) );
    }

    if ( d_bPrintLinearResiduals ) {
        PetscViewerAndFormat *vf;
        checkErr(
            PetscViewerAndFormatCreate( PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf ) );
        checkErr( KSPMonitorSet(
            //            d_pKrylovSolver->getKrylovSolver(),
            kspSolver,
            (PetscErrorCode( * )( KSP, PetscInt, PetscReal, void * )) KSPMonitorResidual,
            vf,
            (PetscErrorCode( * )( void ** )) PetscViewerAndFormatDestroy ) );
    }

    if ( d_PetscMonitor ) {
        // Add the monitor
        SNESMonitorSet( d_SNESSolver, PetscMonitor::monitorSNES, d_PetscMonitor.get(), nullptr );
    }

    d_bPetscInterfaceInitialized = true;
}

void PetscSNESSolver::preApply( std::shared_ptr<const AMP::LinearAlgebra::Vector> v )
{
    if ( d_bDestroyCachedVecs ) {
        SNESLineSearch snesLineSearch;
        SNESGetLineSearch( d_SNESSolver, &snesLineSearch );
        // reset the SNES line search objet to deallocate previous vectors
        // This is important when the solver is being re-used with only a change
        // in the type of vectors being passed in as there will be a mismatch between the
        // vectors created and cached by the linesearch and the input
        SNESLineSearchReset( snesLineSearch );
    }

    auto spv = AMP::LinearAlgebra::PetscVector::constView( v );

    if ( spv ) {
        // a clone is done here even if d_pResidualVector is allocated
        // to guard against the possibility of two consecutive solves having
        // different vector types (see the test testPetscSNESSolver)
        auto r = spv->getManagedVec();
        if ( d_bDestroyCachedVecs || ( !d_pResidualVector ) )
            d_pResidualVector = r->clone();
        if ( d_bDestroyCachedVecs || ( !d_pScratchVector ) )
            d_pScratchVector = d_pResidualVector->clone();
    }

    AMP_ASSERT( d_pResidualVector );
    auto petscVec = AMP::LinearAlgebra::PetscVector::view( d_pResidualVector );
    AMP_INSIST( petscVec,
                "ERROR: Currently the SNES Solver can only be used with a Petsc_Vector, "
                "the supplied Vector does not appear to belong to this class" );
    Vec residualVector = petscVec->getVec();
    SNESSetFunction( d_SNESSolver, residualVector, PetscSNESSolver::apply, (void *) this );

    // Set the jacobian
    std::shared_ptr<AMP::LinearAlgebra::PetscMatrix> view1;
    if ( !d_bUsesJacobian ) {
        // at present destroying the Jacobian for the else case throws an error
        if ( d_Jacobian ) {
            PETSC::matDestroy( &d_Jacobian );
            d_Jacobian = nullptr;
        }
        checkErr( MatCreateSNESMF( d_SNESSolver, &d_Jacobian ) );
        checkErr( MatMFFDSetType( d_Jacobian, (MatMFFDType) d_sMFFDDifferencingStrategy.c_str() ) );
        checkErr( MatMFFDSetFunctionError( d_Jacobian, d_dMFFDFunctionDifferencingError ) );
        if ( d_bEnableMFFDBoundsCheck ) {
            checkErr( MatMFFDSetCheckh( d_Jacobian, PetscSNESSolver::mffdCheckBounds, this ) );
        }
        checkErr( MatSetFromOptions( d_Jacobian ) );
    } else {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>(
            d_pKrylovSolver->getOperator() );
        if ( linearOp ) {
            view1      = AMP::LinearAlgebra::PetscMatrix::view( linearOp->getMatrix() );
            d_Jacobian = view1->getMat();
        } else {
            AMP_INSIST( linearOp,
                        "ERROR: The LinearOperator pointer in the PetscKrylovSolver is NULL" );
        }
    }
    auto pcSolver  = d_pKrylovSolver->getNestedSolver();
    Mat PCJacobian = d_Jacobian;
    std::shared_ptr<AMP::LinearAlgebra::PetscMatrix> view2;
    if ( pcSolver && d_bUsesJacobian ) {
        auto linearOp =
            std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( pcSolver->getOperator() );
        if ( linearOp ) {
            auto matrix = linearOp->getMatrix();
            if ( matrix ) {
                view2      = AMP::LinearAlgebra::PetscMatrix::view( matrix );
                PCJacobian = view2->getMat();
            }
        }
    }

    checkErr( SNESSetJacobian(
        d_SNESSolver, d_Jacobian, PCJacobian, PetscSNESSolver::setJacobian, this ) );
}
void PetscSNESSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    auto petscOptions = db->getWithDefault<std::string>( "SNESOptions", "" );
    d_PetscMonitor.reset();
    if ( petscOptions.find( "monitor" ) != std::string::npos ) {
        petscOptions = PetscMonitor::removeMonitor( petscOptions );
        d_PetscMonitor.reset( new PetscMonitor( d_comm ) );
    }
    // if the solver type is specified as 'ls' replace by 'newtonls'
    // this hack is brittle and will easily fail if the string is
    // not matched exactly in the input. Temporary fix for now
    std::string solverTypeStr( "-snes_type ls" );
    auto pos = petscOptions.find( solverTypeStr );
    if ( pos != std::string::npos ) {
        petscOptions.erase( pos, solverTypeStr.length() );
        petscOptions += " -snes_type newtonls";
    }

    PetscOptionsInsertString( nullptr, petscOptions.c_str() );

    d_bUsesJacobian = db->getWithDefault<bool>( "usesJacobian", false );

    // account for different keywords
    d_sMFFDDifferencingStrategy = MATMFFD_WP;
    if ( db->keyExists( "MFFDDifferencingStrategy" ) ) {
        d_sMFFDDifferencingStrategy = db->getScalar<std::string>( "MFFDDifferencingStrategy" );
    }
    if ( db->keyExists( "differencing_parameter_strategy" ) ) {
        d_sMFFDDifferencingStrategy =
            db->getScalar<std::string>( "differencing_parameter_strategy" );
    }

    d_dMFFDFunctionDifferencingError =
        db->getWithDefault<double>( "MFFDFunctionDifferencingError", PETSC_DEFAULT );

    d_SNESAppendOptionsPrefix = db->getWithDefault<std::string>( "SNESAppendOptionsPrefix", "" );

    if ( db->keyExists( "maximumFunctionEvals" ) )
        d_iMaximumFunctionEvals = db->getScalar<int>( "maximumFunctionEvals" );
    if ( db->keyExists( "maximum_function_evals" ) )
        d_iMaximumFunctionEvals = db->getScalar<int>( "maximum_function_evals" );

    // account for different keywords
    if ( db->keyExists( "stepTolerance" ) )
        d_dStepTolerance = db->getScalar<double>( "stepTolerance" );
    if ( db->keyExists( "step_tolerance" ) )
        d_dStepTolerance = db->getScalar<double>( "step_tolerance" );

    d_bEnableLineSearchPreCheck = db->getWithDefault<bool>( "enableLineSearchPreCheck", false );

    if ( d_bEnableLineSearchPreCheck )
        d_iNumberOfLineSearchPreCheckAttempts =
            db->getWithDefault<int>( "numberOfLineSearchPreCheckAttempts", 5 );

    d_bEnableMFFDBoundsCheck = db->getWithDefault<bool>( "enableMFFDBoundsCheck", false );
    //    if ( d_bEnableMFFDBoundsCheck )
    //        d_operatorComponentToEnableBoundsCheck =
    //            db->getScalar<int>( "operatorComponentToEnableBoundsCheck" );

    d_sForcingTermStrategy = db->getWithDefault<std::string>( "forcing_term_strategy", "CONSTANT" );
    if ( d_sForcingTermStrategy == "EWCHOICE1" ) {
        d_iForcingTermFlag = 1;
    } else if ( d_sForcingTermStrategy == "EWCHOICE2" ) {
        d_iForcingTermFlag = 2;
    } else if ( d_sForcingTermStrategy == "EWCHOICE3" ) {
        d_iForcingTermFlag = 3;
    } else if ( !( d_sForcingTermStrategy == "CONSTANT" ) ) {
        AMP_ERROR( d_sName + ": Key data `forcing_term_strategy' = " + d_sForcingTermStrategy +
                   " in input not recognized." );
    }

    d_dConstantForcingTerm = db->getWithDefault<double>( "constant_forcing_term", PETSC_DEFAULT );

    d_dInitialForcingTerm = db->getWithDefault<double>( "initial_forcing_term", PETSC_DEFAULT );

    d_dMaximumForcingTerm = db->getWithDefault<double>( "maximum_forcing_term", PETSC_DEFAULT );

    d_dEWChoice2Alpha = db->getWithDefault<double>( "EW_choice2_alpha", PETSC_DEFAULT );

    d_dEWChoice2Gamma = db->getWithDefault<double>( "EW_choice2_gamma", PETSC_DEFAULT );

    d_dEWSafeguardExponent = db->getWithDefault<double>( "EW_safeguard_exponent", PETSC_DEFAULT );

    d_dEWSafeguardDisableThreshold =
        db->getWithDefault<double>( "EW_safeguard_disable_threshold", PETSC_DEFAULT );

    d_bPrintNonlinearResiduals = db->getWithDefault<bool>( "print_nonlinear_residuals", false );
    d_bPrintLinearResiduals    = db->getWithDefault<bool>( "print_linear_residuals", false );
    if ( d_iDebugPrintInfoLevel > 0 ) {
        d_bPrintNonlinearResiduals = true;
    }
    if ( d_iDebugPrintInfoLevel > 1 ) {
        d_bPrintLinearResiduals = true;
    }
    d_bDestroyCachedVecs = db->getWithDefault<bool>( "destroy_cached_vecs", false );
}

std::shared_ptr<SolverStrategy>
PetscSNESSolver::createPreconditioner( std::shared_ptr<AMP::Database> pc_solver_db )
{
    AMP_INSIST(
        pc_solver_db,
        "PetscSNESSolver::createPreconditioner: Database object for preconditioner is NULL" );
    std::shared_ptr<SolverStrategy> preconditionerSolver;
    auto pcSolverParameters =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( pc_solver_db );
    if ( d_pOperator ) {
        auto pc_params = d_pOperator->getParameters( "Jacobian", d_pSolutionVector );
        std::shared_ptr<AMP::Operator::Operator> pcOperator =
            AMP::Operator::OperatorFactory::create( pc_params );
        pcSolverParameters->d_pOperator = pcOperator;
    }

    preconditionerSolver = AMP::Solver::SolverFactory::create( pcSolverParameters );

    return preconditionerSolver;
}

/****************************************************************
 *  Apply                                                        *
 ****************************************************************/
PetscErrorCode PetscSNESSolver::apply( SNES, Vec x, Vec r, void *ctx )
{
    PROFILE( "apply" );
    int ierr = 0;

    auto sp_x = PETSC::getAMP( x );
    auto sp_r = PETSC::getAMP( r );

    std::shared_ptr<AMP::LinearAlgebra::Vector> sp_f;

    if ( sp_x )
        sp_x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    auto *pSNESSolver = reinterpret_cast<PetscSNESSolver *>( ctx );
    std::shared_ptr<AMP::Operator::Operator> op( pSNESSolver->getOperator() );

    op->apply( sp_x, sp_r );

    return ( ierr );
}


/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
void PetscSNESSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "solve" );

    if ( d_bUseZeroInitialGuess ) {
        u->zero();
    }

    if ( d_iDebugPrintInfoLevel > 2 )
        AMP::pout << "L2 Norm of u in PetscSNESSolver::solve before view " << u->L2Norm()
                  << std::endl;
    auto v = f ? f : std::const_pointer_cast<const AMP::LinearAlgebra::Vector>( u );
    preApply( v );


    // Get petsc views of the vectors
    auto spRhs = AMP::LinearAlgebra::PetscVector::constView( f );
    auto spSol = AMP::LinearAlgebra::PetscVector::view( u );
    AMP_ASSERT( spSol );

    // Check input vector states
    if ( u->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::UNCHANGED ) {
        u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        auto fNorm = f ? f->L2Norm() : 0.0;
        AMP::pout << "L2 Norm of f in PetscSNESSolver::solve after view " << fNorm << std::endl;
        AMP::pout << "L2 Norm of u in PetscSNESSolver::solve after view " << u->L2Norm()
                  << std::endl;
    }

    Vec x = spSol->getVec();
    Vec b = spRhs ? spRhs->getVec() : nullptr;

    // Solve
    {
        PROFILE( "petsc-SNESSolve" );
        checkErr( SNESSolve( d_SNESSolver, b, x ) );
    }

    // Note that an alternative would be to use the PETSc routine
    // SNESSetConvergenceHistory. At present we don't use it since
    // the solver reset which in the AMR use case is called leads to
    // deletion of the SNES solver. This is probably not optimal
    // but some care is required to re-initialize SNES so that cached
    // vectors are not kept. A refactor at some future point should
    // address this and other issues that the present implementation has
    PetscInt iters = 0;
    checkErr( SNESGetIterationNumber( d_SNESSolver, &iters ) );
    d_iNumberIterations = static_cast<int>( iters );
    d_iterationHistory.push_back( d_iNumberIterations );

    PetscInt iLinearIterations = 0;
    checkErr( SNESGetLinearSolveIterations( d_SNESSolver, &iLinearIterations ) );
    d_iLinearIterationHistory.push_back( iLinearIterations );

    setConvergenceStatus();

    if ( d_iDebugPrintInfoLevel > 0 ) {

        AMP::pout << "  SNES Iterations:  nonlinear: " << d_iNumberIterations << std::endl;
        AMP::pout << "                       linear: " << iLinearIterations << std::endl;
    }

    // Reset the solvers
    //    SNESReset( d_SNESSolver );

    spRhs.reset();
    spSol.reset();

    if ( d_iDebugPrintInfoLevel > 2 ) {

        PetscReal petscNorm;
        VecNorm( x, NORM_2, &petscNorm );
        AMP::pout << "Petsc L2 Norm of u after solve " << petscNorm << std::endl;
        AMP::pout << "AMP L2 Norm of u after solve " << u->L2Norm() << std::endl;
        AMP::pout << "Vector values" << u << std::endl;
    }

    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

void PetscSNESSolver::reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params )
{
    // BP: 02/14/2012
    // the reset call will typically happen after a regrid
    // if the number of refinement levels changes during the
    // regrid then the vector will try to deallocate data
    // on the wrong number of refinement levels
    // we can count on SAMRAI to deallocate data on patches
    // that no longer exist and keep data on patches that do
    // so the deallocate call is unnecessary and causes problems
    //   solution_vector->deallocateVectorData();
    if ( d_pSolutionVector )
        d_pSolutionVector->getVectorData()->reset();
    if ( d_pResidualVector )
        d_pResidualVector->getVectorData()->reset();
    if ( d_pScratchVector )
        d_pScratchVector->getVectorData()->reset();

    destroyPetscObjects();
    // BP: 04/5/2022
    // We need to be careful that the params object is correctly initialized for
    // the internal creation of Krylov solvers.
    createPetscObjects( params );
    initializePetscObjects();
}

int PetscSNESSolver::defaultLineSearchPreCheck( std::shared_ptr<AMP::LinearAlgebra::Vector> x,
                                                std::shared_ptr<AMP::LinearAlgebra::Vector> y,
                                                bool &changed_y )
{
    int ierr            = 1;
    auto pScratchVector = getScratchVector();

    //    pScratchVector->add( *x, *y );
    pScratchVector->subtract( *x, *y );
    auto solnScaling = this->getSolutionScaling();
    if ( solnScaling )
        pScratchVector->multiply( *pScratchVector, *solnScaling );

    bool valid_v = isVectorValid( d_pOperator, pScratchVector, x->getComm() );
    if ( valid_v ) {
        changed_y = PETSC_FALSE;
        ierr      = 0;
    } else {

        int N_line          = getNumberOfLineSearchPreCheckAttempts();
        const double lambda = 0.5;
        for ( int i = 0; i < N_line; i++ ) {
            if ( d_iDebugPrintInfoLevel > 1 ) {
                AMP::pout << "Attempting to scale search, attempt number " << i << std::endl;
            }
            y->scale( lambda, *y );
            //            pScratchVector->add( *x, *y );
            pScratchVector->subtract( *x, *y );
            if ( solnScaling )
                pScratchVector->multiply( *pScratchVector, *solnScaling );
            valid_v = isVectorValid( d_pOperator, pScratchVector, x->getComm() );
            if ( valid_v ) {
                ierr      = 0;
                changed_y = PETSC_TRUE;
                break;
            }
        }

        // if all else fails truncate to zero
        if ( !valid_v ) {
            ierr      = 0;
            changed_y = PETSC_TRUE;
            y->setMin( 0.0 );
        }
    }
    return ierr;
}

void PetscSNESSolver::printConvergenceStatus( SolverStrategy::SolverStatus status,
                                              std::ostream &os ) const
{
    std::string offset = "  ";
    if ( d_iDebugPrintInfoLevel > 0 ) {
        switch ( status ) {
        case SolverStrategy::SolverStatus::ConvergedOnAbsTol:
            os << offset << type() << " converged on absolute tolerance" << std::endl;
            break;
        case SolverStrategy::SolverStatus::ConvergedOnRelTol:
            os << offset << type() << " converged on relative tolerance" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedOther:
            os << offset << type() << " diverged, solver type specific reason" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedOnNan:
            os << offset << type() << " diverged on NaNs" << std::endl;
            break;
        case SolverStrategy::SolverStatus::MaxIterations:
            os << offset << type() << " diverged on max iterations" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedNestedSolver:
            os << offset << type() << " diverged on nested solver divergence" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedLineSearch:
            os << offset << type() << " diverged on line search" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedStepSize:
            os << offset << type() << " diverged on step size less than tolerance" << std::endl;
            break;
        case SolverStrategy::SolverStatus::DivergedFunctionCount:
            os << offset << type() << " diverged on function evaluations exceeded" << std::endl;
            break;
        default:
            os << offset << type() << " diverged, reason unknown" << std::endl;
            break;
        }
    }
}

void PetscSNESSolver::setConvergenceStatus( void )
{
    checkErr( SNESGetConvergedReason( d_SNESSolver, &d_SNES_completion_code ) );

    switch ( (int) d_SNES_completion_code ) {
    case SNES_CONVERGED_FNORM_ABS:
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        break;
    case SNES_CONVERGED_FNORM_RELATIVE:
        d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
        break;
    case SNES_CONVERGED_SNORM_RELATIVE:
        d_ConvergenceStatus = SolverStatus::DivergedStepSize;
        break;
    case SNES_DIVERGED_FUNCTION_COUNT:
        d_ConvergenceStatus = SolverStatus::DivergedFunctionCount;
        break;
    case SNES_DIVERGED_FNORM_NAN:
        d_ConvergenceStatus = SolverStatus::DivergedOnNan;
        break;
    case SNES_DIVERGED_MAX_IT:
        d_ConvergenceStatus = SolverStatus::MaxIterations;
        break;
    case SNES_DIVERGED_LINEAR_SOLVE:
        d_ConvergenceStatus = SolverStatus::DivergedNestedSolver;
        break;
    case SNES_DIVERGED_LINE_SEARCH:
        d_ConvergenceStatus = SolverStatus::DivergedLineSearch;
        break;
    default:
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        AMP_WARNING( "Unknown SNES completion code reported" );
        break;
    }

    printConvergenceStatus( d_ConvergenceStatus );
}

void PetscSNESSolver::setLineSearchPreCheck(
    std::function<int( std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       bool & )> lineSearchPreCheckPtr )
{
    d_bEnableLineSearchPreCheck = true;
    SNESLineSearch snesLineSearch;
    SNESGetLineSearch( d_SNESSolver, &snesLineSearch );
    checkErr( SNESLineSearchSetPreCheck(
        snesLineSearch, &wrapperLineSearchPreCheck, (void *) ( this ) ) );

    d_lineSearchPreCheckPtr = lineSearchPreCheckPtr;
}

/****************************************************************
 *  setJacobian                                                  *
 ****************************************************************/
PetscErrorCode PetscSNESSolver::setJacobian( SNES, Vec x, Mat A, Mat B, void *ctx )
{
    PROFILE( "setJacobian" );
    int ierr           = 0;
    auto *pSNESSolver  = reinterpret_cast<PetscSNESSolver *>( ctx );
    bool bUsesJacobian = pSNESSolver->getUsesJacobian();

    if ( !bUsesJacobian ) {
        ierr = MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        ierr = MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
        if ( A != B ) {
            ierr = MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY );
            ierr = MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY );
        }
    }

    auto pSolution = PETSC::getAMP( x );
    auto op        = pSNESSolver->getOperator();
    pSolution->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    auto op_parameters = op->getParameters( "Jacobian", pSolution );
    auto pKrylovSolver = pSNESSolver->getKrylovSolver();
    pKrylovSolver->resetOperator( op_parameters );

    return ierr;
}


/****************************************************************
 *  Check if the vector is valid                                 *
 ****************************************************************/
bool PetscSNESSolver::isVectorValid( std::shared_ptr<AMP::Operator::Operator> &op,
                                     AMP::LinearAlgebra::Vector::shared_ptr &v,
                                     const AMP_MPI &comm )
{
    bool retVal = false;
    int msg     = op->isValidVector( v ) ? 1 : 0;

    int result = comm.minReduce( msg );
    retVal     = ( result == 1 );
    return retVal;
}

// KSP Pre and Post Solve routines with Eisenstat-Walker that Petsc
// does not at present expose. These are largely copied from the PETSc
// internal routines with fixes applied for the first routine that decrease
// the number of linear iterations required
PetscErrorCode PetscSNESSolver::KSPPreSolve_SNESEW( KSP ksp, Vec, Vec, SNES snes )
{
    PetscErrorCode ierr;
    SNESKSPEW *kctx = (SNESKSPEW *) snes->kspconvctx;
    PetscReal rtol  = PETSC_DEFAULT, stol;

    PetscFunctionBegin;
    if ( !snes->ksp_ewconv )
        PetscFunctionReturn( 0 );
    if ( !snes->iter ) {
        rtol = kctx->rtol_0; /* first time in, so use the original user rtol */
        ierr = VecNorm( snes->vec_func, NORM_2, &kctx->norm_first );
        CHKERRQ( ierr );
    } else {
        if ( kctx->version == 1 ) {
            rtol = ( snes->norm - kctx->lresid_last ) / kctx->norm_last;
            if ( rtol < 0.0 )
                rtol = -rtol;
            stol = PetscPowReal( kctx->rtol_last, kctx->alpha2 );
            if ( stol > kctx->threshold )
                rtol = PetscMax( rtol, stol );
        } else if ( kctx->version == 2 ) {
            rtol = kctx->gamma * PetscPowReal( snes->norm / kctx->norm_last, kctx->alpha );
            stol = kctx->gamma * PetscPowReal( kctx->rtol_last, kctx->alpha );
            if ( stol > kctx->threshold )
                rtol = PetscMax( rtol, stol );
        } else if ( kctx->version == 3 ) { /* contributed by Luis Chacon, June 2006. */
            rtol = kctx->gamma * std::pow( snes->norm / kctx->norm_last, kctx->alpha );
            /* safeguard: avoid sharp decrease of rtol */
            stol = kctx->gamma * std::pow( kctx->rtol_last, kctx->alpha );
            if ( stol <= 0.1 ) {
                rtol = PetscMin( kctx->rtol_0, rtol );
            } else {
                stol = PetscMax( rtol, stol );
                rtol = PetscMin( kctx->rtol_0, stol );
            }
            /* safeguard: avoid oversolving */
            stol = 0.8 * ( snes->ttol ) / snes->norm;
            stol = PetscMax( rtol, stol );
            rtol = PetscMin( kctx->rtol_0, stol );
        } else
            SETERRQ( PETSC_COMM_SELF,
                     PETSC_ERR_ARG_OUTOFRANGE,
                     "Only versions 1, 2 or 3 are supported: %li",
                     static_cast<long>( kctx->version ) );
    }
    /* safeguard: avoid rtol greater than one */
    rtol = PetscMin( rtol, kctx->rtol_max );
    ierr = KSPSetTolerances( ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT );
    CHKERRQ( ierr );
    ierr = PetscInfo( snes,
                      "iter %li, Eisenstat-Walker (version %li) KSP rtol=%g\n",
                      static_cast<long>( snes->iter ),
                      static_cast<long>( kctx->version ),
                      static_cast<PetscReal>( rtol ) );
    CHKERRQ( ierr );
    PetscFunctionReturn( 0 );
}

PetscErrorCode PetscSNESSolver::KSPPostSolve_SNESEW( KSP ksp, Vec b, Vec x, SNES snes )
{
    PetscErrorCode ierr;
    SNESKSPEW *kctx = (SNESKSPEW *) snes->kspconvctx;
    PCSide pcside;
    Vec lres;

    PetscFunctionBegin;
    if ( !snes->ksp_ewconv )
        PetscFunctionReturn( 0 );
    ierr = KSPGetTolerances( ksp, &kctx->rtol_last, NULL, NULL, NULL );
    CHKERRQ( ierr );
    kctx->norm_last = snes->norm;
    if ( kctx->version == 1 ) {
        PC pc;
        PetscBool isNone;

        ierr = KSPGetPC( ksp, &pc );
        CHKERRQ( ierr );
        ierr = PetscObjectTypeCompare( (PetscObject) pc, PCNONE, &isNone );
        CHKERRQ( ierr );
        ierr = KSPGetPCSide( ksp, &pcside );
        CHKERRQ( ierr );
        if ( pcside == PC_RIGHT ||
             isNone ) { /* XXX Should we also test KSP_UNPRECONDITIONED_NORM ? */
            /* KSP residual is true linear residual */
            ierr = KSPGetResidualNorm( ksp, &kctx->lresid_last );
            CHKERRQ( ierr );
        } else {
            /* KSP residual is preconditioned residual */
            /* compute true linear residual norm */
            ierr = VecDuplicate( b, &lres );
            CHKERRQ( ierr );
            ierr = MatMult( snes->jacobian, x, lres );
            CHKERRQ( ierr );
            ierr = VecAYPX( lres, -1.0, b );
            CHKERRQ( ierr );
            ierr = VecNorm( lres, NORM_2, &kctx->lresid_last );
            CHKERRQ( ierr );
            ierr = VecDestroy( &lres );
            CHKERRQ( ierr );
        }
    }
    PetscFunctionReturn( 0 );
}


/****************************************************************
 *  Linesearch precheck                                          *
 ****************************************************************/

int PetscSNESSolver::wrapperLineSearchPreCheck(
    SNESLineSearch, Vec x, Vec y, PetscBool *changed_y, void *ctx )
{
    bool b_changed_y = false;
    int ierr         = 0;

    PROFILE( "wrapperLineSearchPreCheck" );
    AMP_ASSERT( ctx != nullptr );
    auto snesSolver = reinterpret_cast<PetscSNESSolver *>( ctx );

    auto xv = PETSC::getAMP( x );
    auto yv = PETSC::getAMP( y );
    ierr    = snesSolver->getLineSearchPreCheckAdaptor()( xv, yv, b_changed_y );

    *changed_y = static_cast<PetscBool>( b_changed_y );

    return ierr;
}

PetscErrorCode PetscSNESSolver::mffdCheckBounds( void *checkctx, Vec U, Vec a, PetscScalar *h )
{
    auto *pSNESSolver  = reinterpret_cast<PetscSNESSolver *>( checkctx );
    auto pSNESOperator = pSNESSolver->getOperator();
    std::shared_ptr<AMP::Operator::Operator> pOperator;
    auto pScratchVector = pSNESSolver->getScratchVector();

    auto sp_u = PETSC::getAMP( U );
    auto sp_a = PETSC::getAMP( a );

    // check for column operators
    auto pColumnOperator =
        std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>( pSNESOperator );
    if ( pColumnOperator ) {
        pOperator = pColumnOperator->getOperator( pSNESSolver->getBoundsCheckComponent() );
    } else {
        pOperator = pSNESOperator;
    }

    auto opVar = pOperator->getOutputVariable();
    auto scv   = pScratchVector->subsetVectorForVariable( opVar );
    auto uv    = sp_u->subsetVectorForVariable( opVar );
    auto av    = sp_a->subsetVectorForVariable( opVar );

    scv->axpy( *h, *av, *uv );
#if 1
    //    AMP_ASSERT( isVectorValid( pOperator, uv, sp_u->getComm() ) );
    while ( !isVectorValid( pOperator, scv, sp_u->getComm() ) ) {
        AMP::pout << "Scaling h back from  " << ( *h ) << " to " << 0.75 * ( *h ) << std::endl;
        *h = 0.75 * ( *h );
        scv->axpy( *h, *av, *uv );
    }
#else
    // the code below is only valid for ensuring positivity
    // will do for now
    if ( !isVectorValid( pOperator, scv, sp_u->getComm() ) ) {
        double minVal = PetscAbsScalar( ( *h ) * 1.01 );
        scv->divide( *uv, *av );
        scv->abs( *scv );
        minVal = std::min( static_cast<double>( scv->min() ), minVal );
        if ( minVal <= PetscAbsScalar( *h ) ) {
            AMP::pout << "Scaling h back from  " << ( *h ) << " to " << 0.99 * minVal << std::endl;
            if ( PetscRealPart( *h ) > 0.0 )
                *h = 0.99 * minVal;
            else
                *h = -0.99 * minVal;
        }
    }
#endif
    return ( 0 );
}

PetscErrorCode PetscSNESSolver::setupPreconditioner( PC pc )
{
    PROFILE( "PetscSNESSolver::setupPreconditioner" );

    int ierr = 0;
    Vec current_solution;
    void *ctx;
    PCShellGetContext( pc, &ctx );

    auto snesSolver = static_cast<PetscSNESSolver *>( ctx );
    AMP_ASSERT( snesSolver );
    checkErr( SNESGetSolution( snesSolver->getSNESSolver(), &current_solution ) );

    auto krylovSolver = snesSolver->getKrylovSolver();
    AMP_ASSERT( krylovSolver );

    auto preconditioner = krylovSolver->getNestedSolver();
    AMP_ASSERT( preconditioner );

    // preconditioners like MG might need to rebuild their hierarchies
    // note that at present we are passing in a null parameter object
    // in future things like the solution vector might be appropriate
    preconditioner->reset( {} );

    return ierr;
}

PetscErrorCode PetscSNESSolver::applyPreconditioner( PC pc,
                                                     Vec xin,   // input vector
                                                     Vec xout ) // output vector
{
    PROFILE( "PetscSNESSolver::applyPreconditioner" );

    void *ctx = nullptr;
    PCShellGetContext( pc, &ctx );
    auto snesSolver = static_cast<PetscSNESSolver *>( ctx );
    AMP_ASSERT( snesSolver );
    auto krylovSolver = snesSolver->getKrylovSolver();
    AMP_ASSERT( krylovSolver );
    auto preconditioner = krylovSolver->getNestedSolver();
    AMP_ASSERT( preconditioner );

    AMP_ASSERT( xin );
    AMP_ASSERT( xout );
    auto rhs         = PETSC::getAMP( xin );
    auto soln        = PETSC::getAMP( xout );
    auto solnScaling = snesSolver->getSolutionScaling();
    auto funcScaling = snesSolver->getFunctionScaling();
    if ( solnScaling )
        soln->multiply( *soln, *solnScaling );
    if ( funcScaling )
        rhs->multiply( *rhs, *funcScaling );
    // Make sure the vectors are in a consistent state
    rhs->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    soln->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // these tests were helpful in finding a bug
    if ( preconditioner->getDebugPrintInfoLevel() > 5 ) {
        PetscReal norm = 0.0;
        VecNorm( xin, NORM_2, &norm );
        auto rhs_norm = static_cast<PetscReal>( rhs->L2Norm() );
        AMP_ASSERT( AMP::Utilities::approx_equal( norm, rhs_norm ) );
    }

    // BP: 04/09/2012, to prevent norms getting cached
    checkErr( PetscObjectStateIncrease( reinterpret_cast<PetscObject>( xout ) ) );

    // BP: 04/05/2022, the SAMRSolvers version copies input to output (identity pc)
    // if the preconditioner is null. For now we don't
    preconditioner->apply( rhs, soln );
    if ( solnScaling )
        soln->divide( *soln, *solnScaling );
    if ( funcScaling )
        rhs->divide( *rhs, *funcScaling );

    // Check for nans (no communication necessary)
    double localNorm =
        static_cast<double>( soln->getVectorOperations()->localL2Norm( *soln->getVectorData() ) );
    AMP_INSIST( localNorm == localNorm, "NaNs detected in preconditioner" );

    // these tests were helpful in finding a bug
    if ( preconditioner->getDebugPrintInfoLevel() > 5 ) {
        auto ampSolnNorm = static_cast<PetscReal>( soln->L2Norm() );
        AMP::pout << "L2 Norm of soln " << ampSolnNorm << std::endl;
        PetscReal petscSolnNorm = 0.0;
        VecNorm( xout, NORM_2, &petscSolnNorm );
        AMP::pout << "L2 Norm of xout " << petscSolnNorm << std::endl;
        AMP_ASSERT( petscSolnNorm == ampSolnNorm );
    }

    //    snesSolver->logPreconditionerApply();

    return 0;
}

void PetscSNESSolver::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess )
{
    d_pSolutionVector->copyVector( initialGuess );
}


int PetscSNESSolver::getTotalNumberOfLinearIterations( void ) const
{
    return std::accumulate( d_iLinearIterationHistory.begin(), d_iLinearIterationHistory.end(), 0 );
}

} // namespace AMP::Solver
