#include "AMP/solvers/hypre/HypreBiCGSTABSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
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
HypreBiCGSTABSolver::HypreBiCGSTABSolver() : HypreSolver() {}
HypreBiCGSTABSolver::HypreBiCGSTABSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    HYPRE_ParCSRBiCGSTABCreate( d_comm.getCommunicator(), &d_solver );
    setupHypreSolver( parameters );
}

HypreBiCGSTABSolver::~HypreBiCGSTABSolver() { HYPRE_ParCSRBiCGSTABDestroy( d_solver ); }

void HypreBiCGSTABSolver::setupHypreSolver(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HypreBiCGSTABSolver::setupHypreSolver" );

    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HypreBiCGSTABSolver::initialize( parameters );
    }

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );
    hypre_ParCSRMatrixMigrate( parcsr_A, d_memory_location );

    auto op = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_ASSERT( op );
    auto matrix = op->getMatrix();
    AMP_ASSERT( matrix );
    auto f = matrix->getInputVector();
    f->zero(); // just to be safe
    AMP_ASSERT( f );
    copyToHypre( f, d_hypre_rhs );
    HYPRE_ParVector par_x;
    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_x );

    if ( d_bUsesPreconditioner ) {
        if ( d_bDiagScalePC ) {
            HYPRE_Solver bicgstab_precond = NULL;
            HYPRE_BiCGSTABSetPrecond( d_solver,
                                      (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                      (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                      bicgstab_precond );
        } else {
            auto pc = std::dynamic_pointer_cast<HypreSolver>( d_pNestedSolver );
            if ( pc ) {

                auto bicgstab_precond = pc->getHYPRESolver();
                AMP_ASSERT( bicgstab_precond );

                if ( pc->type() == "BoomerAMGSolver" ) {
                    HYPRE_BiCGSTABSetPrecond( d_solver,
                                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                              (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                              bicgstab_precond );
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

    HYPRE_BiCGSTABSetup(
        d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_x, (HYPRE_Vector) par_x );
}

void HypreBiCGSTABSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;

    HypreBiCGSTABSolver::getFromInput( db );

    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
    } else {
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

void HypreBiCGSTABSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    if ( db->keyExists( "logging" ) ) {
        const auto logging = db->getScalar<int>( "logging" );
        HYPRE_BiCGSTABSetLogging( d_solver, logging );
    }

    HYPRE_BiCGSTABSetTol( d_solver, static_cast<HYPRE_Real>( d_dRelativeTolerance ) );
    HYPRE_BiCGSTABSetAbsoluteTol( d_solver, static_cast<HYPRE_Real>( d_dAbsoluteTolerance ) );
    HYPRE_BiCGSTABSetMaxIter( d_solver, d_iMaxIterations );
    HYPRE_BiCGSTABSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
    d_bDiagScalePC        = db->getWithDefault<bool>( "diag_scale_pc", false );
}

void HypreBiCGSTABSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "HypreBiCGSTABSolver::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: HypreBiCGSTABSolver::apply() operator cannot be NULL" );

    HYPRE_SetMemoryLocation( d_memory_location );
    HYPRE_SetExecutionPolicy( d_exec_policy );

    const auto f_norm = static_cast<HYPRE_Real>( f->L2Norm() );

    // Zero rhs implies zero solution, bail out early
    if ( f_norm == static_cast<HYPRE_Real>( 0.0 ) ) {
        u->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "HypreBiCGSTABSolver::apply: solution is zero" << std::endl;
        }
        return;
    }

    // Compute initial residual, used mostly for reporting in this case
    // since Hypre tracks this internally
    // Can we get that value from Hypre and remove one global reduce?
    std::shared_ptr<AMP::LinearAlgebra::Vector> r;
    HYPRE_Real current_res;
    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        current_res = f_norm;
    } else {
        r = f->clone();
        d_pOperator->residual( f, u, r );
        current_res = static_cast<HYPRE_Real>( r->L2Norm() );
    }
    d_dInitialResidual = current_res;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "HypreBiCGSTABSolver::apply: initial L2Norm of solution vector: "
                  << u->L2Norm() << std::endl;
        AMP::pout << "HypreBiCGSTABSolver::apply: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
        AMP::pout << "HypreBiCGSTABSolver::apply: initial L2Norm of residual: " << current_res
                  << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( current_res ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "HypreBiCGSTABSolver::apply: initial residual below tolerance"
                      << std::endl;
        }
        return;
    }

    copyToHypre( u, d_hypre_sol );
    copyToHypre( f, d_hypre_rhs );

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );

    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_b );
    HYPRE_IJVectorGetObject( d_hypre_sol, (void **) &par_x );

    HYPRE_BiCGSTABSolve(
        d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

    copyFromHypre( d_hypre_sol, u );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Query iteration count and store on AMP side
    HYPRE_Int hypre_iters;
    HYPRE_BiCGSTABGetNumIterations( d_solver, &hypre_iters );
    d_iNumberIterations = hypre_iters;
    HYPRE_Real hypre_res;
    HYPRE_BiCGSTABGetFinalRelativeResidualNorm( d_solver, &hypre_res );

    // Check for NaNs
    if ( std::isnan( hypre_res ) ) {
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        AMP_WARNING( "HypreBiCGSTABSolver::apply: Residual norm is NaN" );
    }

    // Re-compute or query final residual
    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, r );
        current_res = static_cast<HYPRE_Real>( r->L2Norm() );
    } else {
        current_res = hypre_res;
    }

    // Store final residual norm and update convergence flags
    d_dResidualNorm = current_res;
    checkStoppingCriteria( current_res );

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "HypreBiCGSTABSolver::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "HypreBiCGSTABSolver::apply: final L2Norm of residual: " << current_res
                  << std::endl;
        AMP::pout << "HypreBiCGSTABSolver::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "HypreBiCGSTABSolver::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

void HypreBiCGSTABSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    HYPRE_ParCSRBiCGSTABDestroy( d_solver );
    HYPRE_ParCSRBiCGSTABCreate( d_comm.getCommunicator(), &d_solver );

    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
