#include "AMP/solvers/hypre/HypreSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <numeric>

DISABLE_WARNINGS
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_config.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
HypreSolver::HypreSolver() : SolverStrategy() {}
HypreSolver::HypreSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );
    HypreSolver::initialize( parameters );
}

HypreSolver::~HypreSolver()
{
    HYPRE_IJVectorDestroy( d_hypre_rhs );
    HYPRE_IJVectorDestroy( d_hypre_sol );
    destroyHypreSolver();
}

void HypreSolver::createHypreSolver()
{
    d_hypreCreateSolver( d_comm.getCommunicator(), &d_solver );
}

void HypreSolver::destroyHypreSolver()
{
    d_hypreDestroySolver( d_solver );
    d_solver = nullptr;
}

void HypreSolver::initialize( std::shared_ptr<const SolverStrategyParameters> )
{
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void HypreSolver::setCommonParameters( std::shared_ptr<const AMP::Database> db )
{
    AMP_DEBUG_ASSERT( db );

    if ( db->keyExists( "logging" ) ) {
        d_logging = db->getScalar<HYPRE_Int>( "logging" );
    }

    d_hypreSetRelativeTolerance( d_solver, static_cast<HYPRE_Real>( d_dRelativeTolerance ) );
    if ( d_hypreSetAbsoluteTolerance )
        d_hypreSetAbsoluteTolerance( d_solver, static_cast<HYPRE_Real>( d_dAbsoluteTolerance ) );
    d_hypreSetMaxIterations( d_solver, d_iMaxIterations );
    d_hypreSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );
    d_hypreSetLogging( d_solver, d_logging );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
    d_bDiagScalePC        = db->getWithDefault<bool>( "diag_scale_pc", false );
}

void HypreSolver::createHYPREMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    d_HypreMatrixAdaptor =
        std::make_shared<AMP::LinearAlgebra::HypreMatrixAdaptor>( matrix->getMatrixData() );
    AMP_ASSERT( d_HypreMatrixAdaptor );
    d_ijMatrix = d_HypreMatrixAdaptor->getHypreMatrix();
    if ( d_iDebugPrintInfoLevel > 3 ) {
        HYPRE_IJMatrixPrint( d_ijMatrix, "HypreMatrix" );
    }
}

void HypreSolver::createHYPREVectors()
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

static AMP::Utilities::MemoryType getAMPMemorySpace( HYPRE_MemoryLocation memory_location )
{
    if ( memory_location == HYPRE_MEMORY_HOST ) {
        return AMP::Utilities::MemoryType::host;
    } else if ( memory_location == HYPRE_MEMORY_DEVICE ) {
#if defined( HYPRE_USING_DEVICE_MEMORY )
        return AMP::Utilities::MemoryType::device;
#elif defined( HYPRE_USING_UNIFIED_MEMORY )
        return AMP::Utilities::MemoryType::managed;
#else
        AMP_ERROR( "Unable to detect Hypre memory location" );
        return AMP::Utilities::MemoryType::device;
#endif
    } else {
        AMP_ERROR( "Unable to detect Hypre memory location" );
        return AMP::Utilities::MemoryType::host;
    }
}

void HypreSolver::copyToHypre( std::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v,
                               HYPRE_IJVector hypre_v )
{
    char hypre_mesg[100];
    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    AMP_INSIST( amp_v->numberOfDataBlocks() == 1,
                "Copy from AMP vectors with more than one data block to Hypre not implemented" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS = dofManager->numLocalDOF();

    ierr = HYPRE_IJVectorInitialize( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );

    HYPRE_Real *vals_p = nullptr;

    if ( amp_v->isType<HYPRE_Real>( 0 ) ) {

        vals_p = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                     ->getRawDataBlock<HYPRE_Real>();

        auto memType      = AMP::Utilities::getMemoryType( vals_p );
        auto hypreMemType = getAMPMemorySpace( d_hypre_memory_location );
        // see if memory spaces are compatible
        if ( memType == hypreMemType ) {
            AMP_ASSERT( vals_p );
            ierr = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
        } else {
            auto compat_amp_v = AMP::LinearAlgebra::createVector( amp_v, hypreMemType );
            compat_amp_v->copyVector( amp_v );
            vals_p = compat_amp_v->getRawDataBlock<HYPRE_Real>();
            ierr   = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
        }

    } else {
        AMP_ERROR( "Copy from AMP to Hypre vectors of different precision not implemented" );
    }

    ierr = HYPRE_IJVectorAssemble( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );
}

void HypreSolver::copyFromHypre( HYPRE_IJVector hypre_v,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v )
{
    AMP_INSIST( amp_v, "vector cannot be NULL" );
    AMP_INSIST( amp_v->numberOfDataBlocks() == 1,
                "Copy from Hypre to AMP vectors with more than one data block not "
                "implemented" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );
    const auto nDOFS = dofManager->numLocalDOF();

    char hypre_mesg[100];
    int ierr;

    auto vals_p = amp_v->getRawDataBlock<HYPRE_Real>();

    if ( amp_v->isType<HYPRE_Real>( 0 ) ) {

        auto memType      = AMP::Utilities::getMemoryType( vals_p );
        auto hypreMemType = getAMPMemorySpace( d_hypre_memory_location );
        // see if memory spaces are compatible
        if ( memType == hypreMemType ) {
            ierr = HYPRE_IJVectorGetValues(
                hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
            return;
        } else {

            AMP_WARN_ONCE(
                "Hypre not built with support for AMP vector memory, vector will be migrated" );

            auto tmp_amp_v = AMP::LinearAlgebra::createVector( amp_v, hypreMemType );
            AMP_ASSERT( tmp_amp_v );
            vals_p = tmp_amp_v->getRawDataBlock<HYPRE_Real>();
            ierr   = HYPRE_IJVectorGetValues(
                hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
            amp_v->copyVector( tmp_amp_v );
        }

    } else {
        AMP_ERROR( "Copy from Hypre to AMP vectors of different precision not implemented" );
    }
}

void HypreSolver::setupHypreMatrixAndRhs()
{
    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );
    auto matrix = linearOperator->getMatrix();
    // a user can choose to set the Operator without initializing the matrix
    // so check whether a valid matrix exists
    if ( matrix ) {

        // set the comm for this solver based on the comm for the matrix
        const auto &dofManager = matrix->getLeftDOFManager();
        d_comm                 = dofManager->getComm();

        // set the hypre memory and execution spaces from the operator
        if ( linearOperator->getMemoryLocation() > AMP::Utilities::MemoryType::host ) {
            d_hypre_memory_location = HYPRE_MEMORY_DEVICE;
            d_hypre_exec_policy     = HYPRE_EXEC_DEVICE;
        } else {
            d_hypre_memory_location = HYPRE_MEMORY_HOST;
            d_hypre_exec_policy     = HYPRE_EXEC_HOST;
        }

        createHYPREMatrix( matrix );
        createHYPREVectors();
        d_bMatrixInitialized = true;
    }
}

void HypreSolver::setupSolver()
{
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );

    auto op = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_ASSERT( op );
    auto matrix = op->getMatrix();
    AMP_ASSERT( matrix );
    auto f = matrix->createInputVector();
    f->zero(); // just to be safe
    AMP_ASSERT( f );
    copyToHypre( f, d_hypre_rhs );
    HYPRE_ParVector par_x;
    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_x );

    if ( d_bUsesPreconditioner ) {
        if ( d_bDiagScalePC ) {
            HYPRE_Solver precond = NULL;
            d_hypreSetPreconditioner( d_solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                      (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                      precond );
        } else {
            auto pc = std::dynamic_pointer_cast<HypreSolver>( d_pNestedSolver );
            if ( pc ) {

                auto precond = pc->getHYPRESolver();
                AMP_ASSERT( precond );

                if ( pc->type() == "BoomerAMGSolver" ) {
                    d_hypreSetPreconditioner( d_solver,
                                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                              precond );
                } else {
                    AMP_ERROR( "Currently only diagonal scaling and Boomer AMG preconditioners are "
                               "supported" );
                }

            } else {
                AMP_ERROR(
                    "Currently only native hypre preconditioners are supported for hypre solvers" );
            }
        }
    }

    d_hypreSolverSetup(
        d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_x, (HYPRE_Vector) par_x );
}

void HypreSolver::setupNestedSolver( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
    } else {
        auto db = parameters->d_db;
        if ( d_bUsesPreconditioner && !d_bDiagScalePC ) {
            auto pcName  = db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
            auto outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;
            if ( outerDB ) {
                auto pcDB   = outerDB->getDatabase( pcName );
                auto pcName = pcDB->getString( "name" );
                if ( pcName == "BoomerAMGSolver" ) {
                    pcDB->putScalar<bool>( "setup_solver", false );
                } else {
                    AMP_ERROR( "Currently only diagonal scaling and Boomer AMG preconditioners are "
                               "supported" );
                }
                auto parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
                parameters->d_pOperator = d_pOperator;
                d_pNestedSolver         = AMP::Solver::SolverFactory::create( parameters );
                AMP_ASSERT( d_pNestedSolver );
            }
        }
    }
}

void HypreSolver::preSolve( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                            std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: " + type() + "::apply() operator cannot be NULL" );

    HYPRE_SetMemoryLocation( d_hypre_memory_location );
    HYPRE_SetExecutionPolicy( d_hypre_exec_policy );

    // Compute initial residual, used mostly for reporting in this case
    // since Hypre tracks this internally
    // Can we get that value from Hypre and remove one global reduce?
    if ( !d_r )
        d_r = f->clone();

    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        d_r->copyVector( f );
    } else {
        d_pOperator->residual( f, u, d_r );
    }

    d_dResidualNorm    = static_cast<HYPRE_Real>( d_r->L2Norm() );
    d_dInitialResidual = d_dResidualNorm;

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << type() << "::apply: initial L2Norm of residual: " << d_dResidualNorm
                  << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << type() << "::apply: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        AMP::pout << type() << "::apply: initial L2Norm of rhs vector: " << f->L2Norm()
                  << std::endl;
    }


    copyToHypre( u, d_hypre_sol );
    copyToHypre( f, d_hypre_rhs );
}

void HypreSolver::hypreSolve()
{
    PROFILE( "HypreSolver::hypreSolve" );
    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( d_dResidualNorm ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << type() << "::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );

    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_b );
    HYPRE_IJVectorGetObject( d_hypre_sol, (void **) &par_x );

    d_hypreSolve( d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );
}

void HypreSolver::postSolve( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    HYPRE_Int hypre_iters;
    d_hypreGetNumIterations( d_solver, &hypre_iters );
    d_iNumberIterations = hypre_iters;

    copyFromHypre( d_hypre_sol, u );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Re-compute final residual, hypre only returns relative residual
    d_pOperator->residual( f, u, d_r );
    d_dResidualNorm = static_cast<HYPRE_Real>( d_r->L2Norm() );
    // Store final residual norm and update convergence flags
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << type() << "::apply: final L2Norm of residual: " << d_dResidualNorm
                  << std::endl;
        AMP::pout << type() << "::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << type() << "::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << type() << "::apply: final L2Norm of solution: " << u->L2Norm() << std::endl;
    }
}

void HypreSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "HypreSolver::apply" );

    preSolve( f, u );

    hypreSolve();

    postSolve( f, u );
}

void HypreSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    AMP_INSIST( d_pOperator, "ERROR: HypreSolver::registerOperator() operator cannot be NULL" );
    setupHypreMatrixAndRhs();
}

void HypreSolver::resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    PROFILE( "resetOperator" );
    AMP_INSIST( ( d_pOperator ), "ERROR: HypreSolver::resetOperator() operator cannot be NULL" );
    d_bMatrixInitialized = false;
    d_pOperator->reset( params );
}


void HypreSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{
    PROFILE( "reset" );
    registerOperator( d_pOperator );
}

} // namespace AMP::Solver
