#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/IO/PIO.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/petsc/PetscMatrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/vectors/petsc/PetscVector.h"
#include "ProfilerApp.h"

#include "petsc.h"
#include "petsc/private/vecimpl.h"
#include "petscksp.h"
#include "petscpc.h"

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


static inline PCSide getPCSide( const std::string &pc_side )
{
    PCSide PcSide = PC_RIGHT;
    if ( pc_side == "RIGHT" ) {
        PcSide = PC_RIGHT;
    } else if ( pc_side == "LEFT" ) {
        PcSide = PC_LEFT;
    } else if ( pc_side == "SYMMETRIC" ) {
        PcSide = PC_SYMMETRIC;
    } else {
        AMP_ERROR( "Unknown value for pc_type" );
    }
    return PcSide;
}


/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
PetscKrylovSolver::PetscKrylovSolver()
    : d_dDivergenceTolerance( 0 ),
      d_bKSPCreatedInternally( false ),
      d_bUsesPreconditioner( false ),
      d_iMaxKrylovDimension( 0 ),
      d_KrylovSolver( nullptr )
{
}
PetscKrylovSolver::PetscKrylovSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters ),
      d_dDivergenceTolerance( 0 ),
      d_bKSPCreatedInternally( false ),
      d_bUsesPreconditioner( false ),
      d_iMaxKrylovDimension( 0 ),
      d_KrylovSolver( nullptr )
{
    d_sName = "PetscKrylovSolver";
    AMP_ASSERT( parameters );

    // Create a default KrylovSolver
    d_bKSPCreatedInternally = true;
    KSPCreate( parameters->d_comm.getCommunicator(), &d_KrylovSolver );

    // Initialize
    initialize( parameters );
}


/****************************************************************
 *  De-constructor                                               *
 ****************************************************************/
PetscKrylovSolver::~PetscKrylovSolver()
{
    if ( d_bKSPCreatedInternally )
        KSPDestroy( &d_KrylovSolver );
}


/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
void PetscKrylovSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    // the comm is set here of instead of the constructor because this routine
    // could be called by SNES directly
    d_comm = parameters->d_comm;
    AMP_ASSERT( !d_comm.isNull() );

    getFromInput( parameters->d_db );

    checkErr( KSPSetType( d_KrylovSolver, d_sKspType.c_str() ) );

    if ( d_KSPAppendOptionsPrefix != "" ) {
        KSPAppendOptionsPrefix( d_KrylovSolver, d_KSPAppendOptionsPrefix.c_str() );
        // PCAppendOptionsPrefix(pc, d_KSPAppendOptionsPrefix.c_str());
    }

    if ( ( d_sKspType == "fgmres" ) || ( d_sKspType == "gmres" ) ) {

        checkErr( KSPGMRESSetRestart( d_KrylovSolver, d_iMaxKrylovDimension ) );

        if ( d_sGmresOrthogonalizationAlgorithm == "modifiedgramschmidt" ) {
            checkErr( KSPGMRESSetOrthogonalization(
                d_KrylovSolver, KSPGMRESModifiedGramSchmidtOrthogonalization ) );
        } else if ( d_sGmresOrthogonalizationAlgorithm == "gmres_cgs_refine_ifneeded" ) {
            checkErr( KSPGMRESSetOrthogonalization(
                d_KrylovSolver, KSPGMRESClassicalGramSchmidtOrthogonalization ) );
            checkErr(
                KSPGMRESSetCGSRefinementType( d_KrylovSolver, KSP_GMRES_CGS_REFINE_IFNEEDED ) );
        } else if ( d_sGmresOrthogonalizationAlgorithm == "gmres_cgs_refine_always" ) {
            checkErr( KSPGMRESSetOrthogonalization(
                d_KrylovSolver, KSPGMRESClassicalGramSchmidtOrthogonalization ) );
            checkErr( KSPGMRESSetCGSRefinementType( d_KrylovSolver, KSP_GMRES_CGS_REFINE_ALWAYS ) );
        }
    } else if ( d_sKspType == "bcgs" ) {
    } else if ( d_sKspType == "cg" ) {
        d_PcSide = "LEFT";
    } else if ( d_sKspType == "preonly" ) {
        // if only preconditioner, override preconditioner side
        d_PcSide = "LEFT";
        checkErr( KSPSetNormType( d_KrylovSolver, KSP_NORM_NONE ) );
    }

    // PetscTruth useZeroGuess = (d_bUseZeroInitialGuess) ? PETSC_TRUE : PETSC_FALSE;
    // ierr = KSPSetInitialGuessNonzero(d_KrylovSolver, useZeroGuess);

    PetscBool useNonzeroGuess = ( !d_bUseZeroInitialGuess ) ? PETSC_TRUE : PETSC_FALSE;

    checkErr( KSPSetInitialGuessNonzero( d_KrylovSolver, useNonzeroGuess ) );

    checkErr( KSPSetTolerances( d_KrylovSolver,
                                static_cast<PetscReal>( d_dRelativeTolerance ),
                                static_cast<PetscReal>( d_dAbsoluteTolerance ),
                                static_cast<PetscReal>( d_dDivergenceTolerance ),
                                d_iMaxIterations ) );
    if ( d_bKSPCreatedInternally ) {
        //        checkErr( KSPSetFromOptions( d_KrylovSolver ) );
    }
    if ( d_PetscMonitor ) {
        // Add the monitor
        checkErr( KSPMonitorSet(
            d_KrylovSolver, PetscMonitor::monitorKSP, d_PetscMonitor.get(), nullptr ) );
    }
    // in this case we make the assumption we can access a PetscMat for now
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
    // Create the preconditioner
    PC pc;
    checkErr( KSPGetPC( d_KrylovSolver, &pc ) );
    if ( d_bUsesPreconditioner ) {

        if ( d_sPcType != "shell" ) {
            // the pointer to the preconditioner should be NULL if we are using a Petsc internal PC
            AMP_ASSERT( d_pNestedSolver.get() == nullptr );
            PCSetType( pc, d_sPcType.c_str() );
        } else {
            // for a shell preconditioner the user context is set to an instance of this class
            // and the setup and apply preconditioner functions for the PCSHELL are set to
            // static member functions of this class. By doing this we do not need to introduce
            // static member functions into every SolverStrategy that might be used as a
            // preconditioner
            // NOTE: if this is being used within a JFNK solver the outer solver sets and applies
            // the PC
            checkErr( PCSetType( pc, PCSHELL ) );
            checkErr( PCShellSetContext( pc, this ) );
            checkErr( PCShellSetSetUp( pc, PetscKrylovSolver::setupPreconditioner ) );
            checkErr( PCShellSetApply( pc, PetscKrylovSolver::applyPreconditioner ) );
            initializePreconditioner( parameters );
        }
        checkErr( KSPSetPCSide( d_KrylovSolver, getPCSide( d_PcSide ) ) );
    } else {
        checkErr( PCSetType( pc, PCNONE ) );
    }
}
// Function to get values from input
void PetscKrylovSolver::getFromInput( std::shared_ptr<AMP::Database> db )
{
    // fill this in
    auto petscOptions = db->getWithDefault<std::string>( "KSPOptions", "" );
    if ( petscOptions.find( "monitor" ) != std::string::npos ) {
        petscOptions = PetscMonitor::removeMonitor( petscOptions );
        d_PetscMonitor.reset( new PetscMonitor( d_comm ) );
    }
    PetscOptionsInsertString( nullptr, petscOptions.c_str() );

    d_sKspType             = db->getWithDefault<std::string>( "ksp_type", "fgmres" );
    d_dDivergenceTolerance = db->getWithDefault<double>( "divergence_tolerance", 1.0e+03 );
    d_iMaxIterations       = db->getWithDefault<int>( "max_iterations", 1000 );

    d_KSPAppendOptionsPrefix = db->getWithDefault<std::string>( "KSPAppendOptionsPrefix", "" );

    if ( ( d_sKspType == "fgmres" ) || ( d_sKspType == "gmres" ) ) {
        d_iMaxKrylovDimension              = db->getWithDefault<int>( "max_krylov_dimension", 20 );
        d_sGmresOrthogonalizationAlgorithm = db->getWithDefault<std::string>(
            "gmres_orthogonalization_algorithm", "modifiedgramschmidt" );
    }

    d_bMatrixFree = db->getWithDefault<bool>( "matrix_free", false );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );

    if ( d_bUsesPreconditioner ) {
        // for now restrict to shell pc's
        d_sPcType = db->getWithDefault<std::string>( "pc_type", "shell" );
        d_PcSide  = db->getWithDefault<std::string>( "pc_side", "RIGHT" );
    } else {
        d_sPcType = "none";
    }
}


/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
void PetscKrylovSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "PetscKrylovSolver::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // Check input vector states
    AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    // this register operation seems to be necessary for Petsc mat shell operations
    // to somehow work
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }

    const auto f_norm = static_cast<PetscReal>( f->L2Norm() );

    // Zero rhs implies zero solution, bail out early
    if ( f_norm == static_cast<PetscReal>( 0.0 ) ) {
        u->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "PetscKrylovSolverSolver::apply: solution is zero" << std::endl;
        }
        return;
    }

    // Compute initial residual, used mostly for reporting in this case
    // since Petsc tracks this internally
    // Can we get that value from Petsc and remove one global reduce?
    std::shared_ptr<AMP::LinearAlgebra::Vector> r;
    PetscReal current_res;
    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        current_res = f_norm;
    } else {
        r = f->clone();
        d_pOperator->residual( f, u, r );
        current_res = static_cast<PetscReal>( r->L2Norm() );
    }
    d_dInitialResidual = current_res;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "PetscKrylovSolverSolver::apply: initial L2Norm of solution vector: "
                  << u->L2Norm() << std::endl;
        AMP::pout << "PetscKrylovSolverSolver::apply: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
        AMP::pout << "PetscKrylovSolverSolver::apply: initial L2Norm of residual: " << current_res
                  << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( current_res ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "PetscKrylovSolverSolver::apply: initial residual below tolerance"
                      << std::endl;
        }
        return;
    }

    // Get petsc views of the vectors
    // This will replace any PETSc references to pointers we also track
    // After this, we are free to delet f_thisGetsAroundPETScSharedPtrIssue without memory leak.
    auto fVecView = AMP::LinearAlgebra::PetscVector::constView( f );
    auto uVecView = AMP::LinearAlgebra::PetscVector::view( u );
    Vec fVec      = fVecView->getVec();
    Vec uVec      = uVecView->getVec();
    PetscReal norm;
    VecNorm( fVec, NORM_2, &norm );
    KSPSolve( d_KrylovSolver, fVec, uVec );
    // Manually update state, needed for mixing different (tpl) solvers
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Query iteration count and store on AMP side
    PetscInt iters;
    KSPGetIterationNumber( d_KrylovSolver, &iters );
    d_iNumberIterations = iters;
    // Re-compute or query final residual
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, r );
        current_res = static_cast<PetscReal>( r->L2Norm() );
    } else {
        KSPGetResidualNorm( d_KrylovSolver, &current_res );
    }

    // Store final residual norm and update convergence flags
    d_dResidualNorm = current_res;
    checkStoppingCriteria( current_res );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "PetscKrylovSolver::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "PetscKrylovSolver::apply: final L2Norm of residual: " << current_res
                  << std::endl;
        AMP::pout << "PetscKrylovSolver::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "PetscKrylovSolver::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }

    // Reset the solvers
    KSPReset( d_KrylovSolver );
    AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
}


/****************************************************************
 *  Function to set the KrylovSolver                             *
 ****************************************************************/
void PetscKrylovSolver::setKrylovSolver( KSP *ksp )
{
    if ( d_bKSPCreatedInternally )
        KSPDestroy( &d_KrylovSolver );
    d_bKSPCreatedInternally = false;
    d_KrylovSolver          = *ksp;
}


/****************************************************************
 *  Function to set the register the operator                    *
 ****************************************************************/
void PetscKrylovSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    // in this case we make the assumption we can access a PetscMat for now
    AMP_ASSERT( op );
    d_pOperator = op;

    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );

    // some weird error preventing a single call to KSPOperators
    // will need to debug later
    if ( linearOperator ) {

        auto pMatrix = AMP::LinearAlgebra::PetscMatrix::view( linearOperator->getMatrix() );
        Mat mat      = pMatrix->getMat();
        KSPSetOperators( d_KrylovSolver, mat, mat );

    } else {

        setupPetscMatInterface( d_pOperator, d_Mat );
        AMP_ASSERT( d_Mat );
        KSPSetOperators( d_KrylovSolver, d_Mat, d_Mat );
    }
}

void PetscKrylovSolver::setupPetscMatInterface( std::shared_ptr<AMP::Operator::Operator> op,
                                                Mat &mat )
{
    AMP_ASSERT( op );

    if ( d_Mat )
        MatDestroy( &d_Mat );

    MPI_Comm comm = op->getMesh()->getComm().getCommunicator();
    checkErr( MatCreateShell( comm,
                              0, // dummy number of local rows
                              0, // dummy number of local columns
                              PETSC_DETERMINE,
                              PETSC_DETERMINE,
                              (void *) this,
                              &mat ) );
    AMP_ASSERT( mat );
    checkErr( MatShellSetOperation( mat, MATOP_MULT, (void ( * )()) PetscKrylovSolver::matVec ) );
    checkErr( MatAssemblyBegin( mat, MAT_FINAL_ASSEMBLY ) );
    checkErr( MatAssemblyEnd( mat, MAT_FINAL_ASSEMBLY ) );
}

void PetscKrylovSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_pOperator ) {
        d_pOperator->reset( params );
        auto linearOperator =
            std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        AMP_ASSERT( linearOperator );

        auto pMatrix = AMP::LinearAlgebra::PetscMatrix::view( linearOperator->getMatrix() );

        Mat mat;
        mat = pMatrix->getMat();

        KSPSetOperators( d_KrylovSolver, mat, mat );
    }

    // should add a mechanism for the linear operator to provide updated parameters for the
    // preconditioner operator though it's unclear where this might be necessary
    if ( d_pNestedSolver && ( d_pOperator != d_pNestedSolver->getOperator() ) ) {
        d_pNestedSolver->resetOperator( params );
    }
}

void PetscKrylovSolver::setZeroInitialGuess( bool use_zero_guess )
{
    d_bUseZeroInitialGuess    = use_zero_guess;
    PetscBool useNonzeroGuess = ( !d_bUseZeroInitialGuess ) ? PETSC_TRUE : PETSC_FALSE;
    checkErr( KSPSetInitialGuessNonzero( d_KrylovSolver, useNonzeroGuess ) );
}

void PetscKrylovSolver::initializePreconditioner(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    d_pNestedSolver = parameters->d_pNestedSolver;

    if ( d_pNestedSolver ) {

        // if the operator has not been initialized
        // attempt to initialize and register and operator
        if ( d_pOperator ) {
            auto op = d_pNestedSolver->getOperator();
            if ( !op ) {
                auto pcOperator = createPCOperator();
                d_pNestedSolver->registerOperator( pcOperator );
            }
        }

    } else {
        auto db     = parameters->d_db;
        auto pcName = db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
        std::shared_ptr<AMP::Database> outerDB;
        outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;
        if ( outerDB ) {
            auto pcDB = outerDB->getDatabase( pcName );
            if ( pcDB ) {
                // the user may set the preconditioner explicitly later
                auto parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
                parameters->d_pOperator = d_pOperator;
                parameters->d_comm      = d_comm;
                d_pNestedSolver         = AMP::Solver::SolverFactory::create( parameters );
                AMP_ASSERT( d_pNestedSolver );
            }
        }
    }
}

std::shared_ptr<AMP::Operator::Operator> PetscKrylovSolver::createPCOperator()
{
    std::shared_ptr<AMP::LinearAlgebra::Vector> x;
    auto pc_params = d_pOperator->getParameters( "Jacobian", x );
    std::shared_ptr<AMP::Operator::Operator> pcOperator =
        AMP::Operator::OperatorFactory::create( pc_params );
    AMP_ASSERT( pcOperator );
    return pcOperator;
}

/****************************************************************
 *  Function to setup the preconditioner                         *
 ****************************************************************/
PetscErrorCode PetscKrylovSolver::setupPreconditioner( PC pc )
{
    int ierr  = 0;
    void *ctx = nullptr;
    ierr      = PCShellGetContext( pc, &ctx );
    return ierr;
}

PetscErrorCode PetscKrylovSolver::matVec( Mat mat, Vec x, Vec y )
{
    int ierr  = 0;
    void *ctx = nullptr;
    checkErr( MatShellGetContext( mat, &ctx ) );
    AMP_ASSERT( ctx != nullptr );
    auto solver = reinterpret_cast<PetscKrylovSolver *>( ctx );
    auto op     = solver->getOperator();
    auto sp_x   = PETSC::getAMP( x );
    auto sp_y   = PETSC::getAMP( y );
    AMP_ASSERT( sp_x && sp_y );
    op->apply( sp_x, sp_y );

    return ierr;
}


/****************************************************************
 *  Function to call the preconditioner                          *
 ****************************************************************/
PetscErrorCode PetscKrylovSolver::applyPreconditioner( PC pc, Vec r, Vec z )
{
    int ierr = 0;
    void *ctx;
    PCShellGetContext( pc, &ctx );
    AMP_ASSERT( ctx != nullptr );
    auto solver = reinterpret_cast<PetscKrylovSolver *>( ctx );

    auto sp_r = PETSC::getAMP( r );
    auto sp_z = PETSC::getAMP( z );

    // Make sure the vectors are in a consistent state
    sp_r->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    sp_z->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // these tests were helpful in finding a bug
    if ( solver->getDebugPrintInfoLevel() > 5 ) {
        PetscReal norm = 0.0;
        VecNorm( r, NORM_2, &norm );
        auto sp_r_norm = static_cast<PetscReal>( sp_r->L2Norm() );
        AMP_ASSERT( AMP::Utilities::approx_equal( norm, sp_r_norm ) );
    }

    // Call the preconditioner
    auto preconditioner = solver->getNestedSolver();
    if ( preconditioner != nullptr ) {
        preconditioner->apply( sp_r, sp_z );
    } else {
        // Use the identity preconditioner
        sp_z->copyVector( sp_r );
    }

    // Check for nans (no communication necessary)
    auto localNorm = static_cast<PetscReal>(
        sp_z->getVectorOperations()->localL2Norm( *sp_z->getVectorData() ) );
    AMP_INSIST( localNorm == localNorm, "NaNs detected in preconditioner" );

    // not sure why, but the state of sp_z is not updated and petsc uses the cached norm
    sp_z->getVectorData()->fireDataChange();

    // these tests were helpful in finding a bug
    if ( solver->getDebugPrintInfoLevel() > 5 ) {
        PetscReal norm = 0.0;
        AMP::pout << "L2 Norm of sp_z " << sp_z->L2Norm() << std::endl;
        VecNorm( z, NORM_2, &norm );
        AMP::pout << "L2 Norm of z " << norm << std::endl;
        AMP_ASSERT( norm == sp_z->L2Norm() );
    }

    return ( ierr );
}


} // namespace AMP::Solver
