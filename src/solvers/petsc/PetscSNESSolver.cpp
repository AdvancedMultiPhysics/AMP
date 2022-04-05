#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/matrices/petsc/PetscMatrix.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/NonlinearSolverParameters.h"
#include "AMP/solvers/SolverFactory.h"
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

namespace AMP::Solver {


#if PETSC_VERSION_LT( 3, 15, 0 )
    #error AMP only supports PETSc 3.15.0 or greater
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
    auto parameters = std::dynamic_pointer_cast<const NonlinearSolverParameters>( params );
    d_comm          = parameters->d_comm;
    d_pKrylovSolver = std::dynamic_pointer_cast<PetscKrylovSolver>( parameters->d_pNestedSolver );

    d_pSolutionVector = parameters->d_pInitialGuess;
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

void PetscSNESSolver::createPetscObjects( std::shared_ptr<const SolverStrategyParameters> params )
{
    auto parameters = std::dynamic_pointer_cast<const NonlinearSolverParameters>( params );

    checkErr( SNESCreate( d_comm.getCommunicator(), &d_SNESSolver ) );
    bool snes_create_pc = false;

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
            // create a default Krylov solver DB
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
                    linearSolverDB->putScalar<std::string>(
                        "pc_solver_name",
                        nonlinearSolverDB->getScalar<std::string>( "pc_solver_name" ) );
                    // set the preconditioner type to be shell if a pc solver name is given
                    snes_create_pc = true;
                }
            }
            pc_type = nonlinearSolverDB->getWithDefault<std::string>( "pc_type", pc_type );
            linearSolverDB->putScalar<std::string>( "pc_type", pc_type );
        }

        std::shared_ptr<SolverStrategy> preconditionerSolver;

        if ( snes_create_pc ) {
            preconditionerSolver = createPreconditioner();
        }
        AMP_ASSERT( linearSolverDB );
        auto linearSolverParams = std::make_shared<PetscKrylovSolverParameters>( linearSolverDB );
        linearSolverParams->d_comm            = d_comm;
        linearSolverParams->d_global_db       = d_global_db;
        linearSolverParams->d_pPreconditioner = preconditionerSolver;
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

    checkErr( SNESSetTolerances( d_SNESSolver,
                                 d_dAbsoluteTolerance,
                                 d_dRelativeTolerance,
                                 d_dStepTolerance,
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
        SNESMonitorSet( d_SNESSolver, PetscMonitor::monitorSNES, d_PetscMonitor.get(), PETSC_NULL );
    }

    d_bPetscInterfaceInitialized = true;
}

void PetscSNESSolver::preApply( std::shared_ptr<const AMP::LinearAlgebra::Vector> v )
{
    SNESLineSearch snesLineSearch;
    SNESGetLineSearch( d_SNESSolver, &snesLineSearch );
    // reset the SNES line search objet to deallocate previous vectors
    // This is important when the solver is being re-used with only a change
    // in the type of vectors being passed in as there will be a mismatch between the
    // vectors created and cached by the linesearch and the input
    SNESLineSearchReset( snesLineSearch );

    auto spv = AMP::LinearAlgebra::PetscVector::constView( v );

    if ( spv ) {
        // a clone is done here even if d_pResidualVector is allocated
        // to guard against the possibility of two consecutive solves having
        // different vector types (see the test testPetscSNESSolver)
        auto r            = spv->getManagedVec();
        d_pResidualVector = r->cloneVector();
        d_pScratchVector  = d_pResidualVector->cloneVector();
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
    auto pcSolver  = d_pKrylovSolver->getPreconditioner();
    Mat PCJacobian = d_Jacobian;
    std::shared_ptr<AMP::LinearAlgebra::PetscMatrix> view2;
    if ( pcSolver ) {
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

    PetscOptionsInsertString( PETSC_NULL, petscOptions.c_str() );

    d_bUsesJacobian = db->getWithDefault<bool>( "usesJacobian", false );
    d_sMFFDDifferencingStrategy =
        db->getWithDefault<std::string>( "MFFDDifferencingStrategy", MATMFFD_WP );
    d_dMFFDFunctionDifferencingError =
        db->getWithDefault<double>( "MFFDFunctionDifferencingError", PETSC_DEFAULT );

    d_SNESAppendOptionsPrefix = db->getWithDefault<std::string>( "SNESAppendOptionsPrefix", "" );

    if ( db->keyExists( "maximumFunctionEvals" ) )
        d_iMaximumFunctionEvals = db->getScalar<int>( "maximumFunctionEvals" );

    if ( db->keyExists( "stepTolerance" ) )
        d_dStepTolerance = db->getScalar<double>( "stepTolerance" );

    d_bEnableLineSearchPreCheck = db->getWithDefault<bool>( "enableLineSearchPreCheck", false );

    if ( d_bEnableLineSearchPreCheck )
        d_iNumberOfLineSearchPreCheckAttempts =
            db->getWithDefault<int>( "numberOfLineSearchPreCheckAttempts", 5 );

    d_bEnableMFFDBoundsCheck = db->getWithDefault<bool>( "enableMFFDBoundsCheck", false );
    if ( d_bEnableMFFDBoundsCheck )
        d_operatorComponentToEnableBoundsCheck =
            db->getScalar<int>( "operatorComponentToEnableBoundsCheck" );

    d_sForcingTermStrategy =
        db->getWithDefault<std::string>( "forcing_term_strategy", std::string{ "CONSTANT" } );
    if ( d_sForcingTermStrategy == "EWCHOICE1" ) {
        d_iForcingTermFlag = 1;
    } else if ( d_sForcingTermStrategy == "EWCHOICE2" ) {
        d_iForcingTermFlag = 2;
    } else if ( d_sForcingTermStrategy == "EWCHOICE3" ) {
        d_iForcingTermFlag = 3;
    } else if ( !( d_sForcingTermStrategy == "CONSTANT" ) ) {
        AMP_ERROR( d_sName << ": "
                           << "Key data `forcing_term_strategy' = " << d_sForcingTermStrategy
                           << " in input not recognized." );
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
}

std::shared_ptr<SolverStrategy> PetscSNESSolver::createPreconditioner( void )
{
    std::shared_ptr<SolverStrategy> preconditionerSolver;
    AMP_ASSERT( d_db->keyExists( "pc_solver_name" ) );
    auto pc_solver_name = d_db->getString( "pc_solver_name" );
    AMP_ASSERT( d_global_db && d_global_db->keyExists( pc_solver_name ) );
    auto pc_solver_db = d_global_db->getDatabase( pc_solver_name );
    auto pcSolverParameters =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( pc_solver_db );
    if ( d_pOperator ) {
        std::shared_ptr<AMP::LinearAlgebra::Vector> x;
        auto pc_params = d_pOperator->getParameters( "Jacobian", x );
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
    PROFILE_START( "apply" );
    int ierr = 0;

    auto sp_x = PETSC::getAMP( x );
    auto sp_r = PETSC::getAMP( r );

    std::shared_ptr<AMP::LinearAlgebra::Vector> sp_f;
    if ( sp_f )
        sp_f->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );
    if ( sp_x )
        sp_x->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );
    sp_r->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    auto *pSNESSolver = reinterpret_cast<PetscSNESSolver *>( ctx );
    std::shared_ptr<AMP::Operator::Operator> op( pSNESSolver->getOperator() );

    op->residual( sp_f, sp_x, sp_r );
    sp_r->scale( -1.0 );
    sp_r->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    PROFILE_STOP( "apply" );
    return ( ierr );
}


/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
void PetscSNESSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE_START( "solve" );

    if ( d_iDebugPrintInfoLevel > 2 )
        AMP::pout << "L2 Norm of u in PetscSNESSolver::solve before view " << u->L2Norm()
                  << std::endl;

    preApply( f );

    // Get petsc views of the vectors
    auto spRhs = AMP::LinearAlgebra::PetscVector::constView( f );
    auto spSol = AMP::LinearAlgebra::PetscVector::view( u );
    AMP_ASSERT( spSol );

    // Check input vector states
    using UpdateState = AMP::LinearAlgebra::VectorData::UpdateState;
    AMP_ASSERT( ( f->getUpdateStatus() == UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == UpdateState::LOCAL_CHANGED ) );
    u->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 2 )
        AMP::pout << "L2 Norm of u in PetscSNESSolver::solve after view " << u->L2Norm()
                  << std::endl;

    Vec x = spSol->getVec();
    Vec b = spRhs ? spRhs->getVec() : PETSC_NULL;

    // Solve
    PROFILE_START( "petsc-SNESSolve" );
    checkErr( SNESSolve( d_SNESSolver, b, x ) );
    PROFILE_STOP( "petsc-SNESSolve" );

    checkErr( SNESGetIterationNumber( d_SNESSolver, &d_iNumberIterations ) );
    d_iterationHistory.push_back( d_iNumberIterations );

    checkErr( SNESGetConvergedReason( d_SNESSolver, &d_SNES_completion_code ) );

    if ( d_iDebugPrintInfoLevel > 0 ) {

        int linearIterations = 0;
        checkErr( SNESGetLinearSolveIterations( d_SNESSolver, &linearIterations ) );
        AMP::pout << " SNES Iterations:  nonlinear: " << d_iNumberIterations << std::endl;
        AMP::pout << "                      linear: " << linearIterations << std::endl;
    }

    // Reset the solvers
    //    SNESReset( d_SNESSolver );

    spRhs.reset();
    spSol.reset();

    u->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    PROFILE_STOP( "solve" );
}

int PetscSNESSolver::defaultLineSearchPreCheck( std::shared_ptr<AMP::LinearAlgebra::Vector> x,
                                                std::shared_ptr<AMP::LinearAlgebra::Vector> y,
                                                bool &changed_y )
{
    int ierr            = 1;
    auto pScratchVector = getScratchVector();

    pScratchVector->add( *x, *y );

    if ( isVectorValid( d_pOperator, pScratchVector, x->getComm() ) ) {
        changed_y = PETSC_FALSE;
        ierr      = 0;
    } else {
        int N_line = getNumberOfLineSearchPreCheckAttempts();
        auto pColumnOperator =
            std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>( d_pOperator );
        if ( pColumnOperator ) {
            for ( int i = 0; i < N_line; i++ ) {
                AMP::pout << "Attempting to scale search, attempt number " << i << std::endl;
                double lambda = 0.5;
                y->scale( lambda, *y );
                pScratchVector->add( *x, *y );
                if ( isVectorValid( d_pOperator, pScratchVector, x->getComm() ) ) {
                    ierr      = 0;
                    changed_y = PETSC_TRUE;
                    break;
                } else {
                    lambda = lambda / 2.0;
                }
            }
        }
    }
    return ierr;
}

void PetscSNESSolver::setConvergenceStatus( void )
{
    switch ( (int) d_SNES_completion_code ) {
    case SNES_CONVERGED_FNORM_ABS:
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        break;
    case SNES_CONVERGED_FNORM_RELATIVE:
        d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
        break;
    case SNES_CONVERGED_SNORM_RELATIVE:
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        break;
    case SNES_DIVERGED_FUNCTION_COUNT:
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        break;
    case SNES_DIVERGED_FNORM_NAN:
        d_ConvergenceStatus = SolverStatus::DivergedOnNan;
        break;
    case SNES_DIVERGED_MAX_IT:
        d_ConvergenceStatus = SolverStatus::DivergedMaxIterations;
        break;
    case SNES_DIVERGED_LINE_SEARCH:
        d_ConvergenceStatus = SolverStatus::DivergedLineSearch;
        break;
    default:
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        AMP_ERROR( "Unknown SNES completion code reported" );
        break;
    }
}

void PetscSNESSolver::setLineSearchPreCheck(
    std::function<int( std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       bool & )> lineSearchPreCheckPtr )
{
    d_lineSearchPreCheckPtr = lineSearchPreCheckPtr;
}

/****************************************************************
 *  setJacobian                                                  *
 ****************************************************************/
PetscErrorCode PetscSNESSolver::setJacobian( SNES, Vec x, Mat A, Mat, void *ctx )
{
    PROFILE_START( "setJacobian" );
    int ierr           = 0;
    auto *pSNESSolver  = reinterpret_cast<PetscSNESSolver *>( ctx );
    bool bUsesJacobian = pSNESSolver->getUsesJacobian();

    if ( !bUsesJacobian ) {
        ierr = MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        ierr = MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
    }

    auto pSolution     = PETSC::getAMP( x );
    auto op            = pSNESSolver->getOperator();
    auto op_parameters = op->getParameters( "Jacobian", pSolution );
    auto pKrylovSolver = pSNESSolver->getKrylovSolver();
    pKrylovSolver->resetOperator( op_parameters );

    PROFILE_STOP( "setJacobian" );
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
    int msg     = op->isValidInput( v ) ? 1 : 0;
    int result  = comm.minReduce( msg );
    retVal      = ( result == 1 );
    return retVal;
}

// KSP Pre and Post Solve routines with Eisenstat-Walker that Petsc
// does not at present expose. These are simply copied from the PETSc
// internal routines
PetscErrorCode PetscSNESSolver::KSPPreSolve_SNESEW( KSP ksp, Vec b, Vec x, SNES snes )
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
            rtol = kctx->gamma * PetscPowReal( snes->norm / kctx->norm_last, kctx->alpha );
            /* safeguard: avoid sharp decrease of rtol */
            stol = kctx->gamma * PetscPowReal( kctx->rtol_last, kctx->alpha );
            stol = PetscMax( rtol, stol );
            rtol = PetscMin( kctx->rtol_0, stol );
            /* safeguard: avoid oversolving */
            stol = kctx->gamma * ( kctx->norm_first * snes->rtol ) / snes->norm;
            stol = PetscMax( rtol, stol );
            rtol = PetscMin( kctx->rtol_0, stol );
        } else
            SETERRQ1( PETSC_COMM_SELF,
                      PETSC_ERR_ARG_OUTOFRANGE,
                      "Only versions 1, 2 or 3 are supported: %D",
                      kctx->version );
    }
    /* safeguard: avoid rtol greater than one */
    rtol = PetscMin( rtol, kctx->rtol_max );
    ierr = KSPSetTolerances( ksp, rtol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT );
    CHKERRQ( ierr );
    ierr = PetscInfo3( snes,
                       "iter %D, Eisenstat-Walker (version %D) KSP rtol=%g\n",
                       snes->iter,
                       kctx->version,
                       (double) rtol );
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

    PROFILE_START( "wrapperLineSearchPreCheck" );
    AMP_ASSERT( ctx != nullptr );
    auto snesSolver = reinterpret_cast<PetscSNESSolver *>( ctx );

    auto xv = PETSC::getAMP( x );
    auto yv = PETSC::getAMP( y );
    ierr    = snesSolver->getLineSearchPreCheckAdaptor()( xv, yv, b_changed_y );

    *changed_y = static_cast<PetscBool>( b_changed_y );

    PROFILE_STOP( "wrapperLineSearchPreCheck" );
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

    // the code below is only valid for ensuring positivity
    // will do for now
    if ( isVectorValid( pOperator, scv, sp_u->getComm() ) ) {
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

    return ( 0 );
}

void PetscSNESSolver::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess )
{
    d_pSolutionVector->copyVector( initialGuess );
}


} // namespace AMP::Solver
