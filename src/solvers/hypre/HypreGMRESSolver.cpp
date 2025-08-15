#include "AMP/solvers/hypre/HypreGMRESSolver.h"
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
HypreGMRESSolver::HypreGMRESSolver() : HypreSolver() {}
HypreGMRESSolver::HypreGMRESSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    HYPRE_ParCSRGMRESCreate( d_comm.getCommunicator(), &d_solver );
    setupHypreSolver( parameters );
}

HypreGMRESSolver::~HypreGMRESSolver() { HYPRE_ParCSRGMRESDestroy( d_solver ); }

void HypreGMRESSolver::setupHypreSolver(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HypreGMRESSolver::setupHypreSolver" );

    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HypreGMRESSolver::initialize( parameters );
    }

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
            HYPRE_Solver gmres_precond = NULL;
            HYPRE_GMRESSetPrecond( d_solver,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                   (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                   gmres_precond );
        } else {
            auto pc = std::dynamic_pointer_cast<HypreSolver>( d_pNestedSolver );
            if ( pc ) {

                auto gmres_precond = pc->getHYPRESolver();
                AMP_ASSERT( gmres_precond );

                if ( pc->type() == "BoomerAMGSolver" ) {
                    HYPRE_GMRESSetPrecond( d_solver,
                                           (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                           (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                           gmres_precond );
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

    HYPRE_GMRESSetup(
        d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_x, (HYPRE_Vector) par_x );
}

void HypreGMRESSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;

    HypreGMRESSolver::getFromInput( db );

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

void HypreGMRESSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    if ( db->keyExists( "logging" ) ) {
        const auto logging = db->getScalar<int>( "logging" );
        HYPRE_GMRESSetLogging( d_solver, logging );
    }

    d_iMaxKrylovDim = db->getWithDefault<int>( "max_krylov_dimension", 100 );
    HYPRE_GMRESSetTol( d_solver, static_cast<HYPRE_Real>( d_dRelativeTolerance ) );
    HYPRE_GMRESSetAbsoluteTol( d_solver, static_cast<HYPRE_Real>( d_dAbsoluteTolerance ) );
    HYPRE_GMRESSetMaxIter( d_solver, d_iMaxIterations );
    HYPRE_GMRESSetKDim( d_solver, static_cast<HYPRE_Int>( d_iMaxKrylovDim ) );
    HYPRE_GMRESSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
    d_bDiagScalePC        = db->getWithDefault<bool>( "diag_scale_pc", false );

    if ( !d_bComputeResidual ) {
        HYPRE_GMRESSetSkipRealResidualCheck( d_solver, d_bComputeResidual );
    }
}

void HypreGMRESSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "HypreGMRESSolver::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 0;

    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: HypreGMRESSolver::apply() operator cannot be NULL" );

    HYPRE_SetMemoryLocation( d_memory_location );
    HYPRE_SetExecutionPolicy( d_exec_policy );

    const auto f_norm = static_cast<HYPRE_Real>( f->L2Norm() );

    // Zero rhs implies zero solution, bail out early
    if ( f_norm == static_cast<HYPRE_Real>( 0.0 ) ) {
        u->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "HypreGMRESSolver::apply: solution is zero" << std::endl;
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
        AMP::pout << "HypreGMRESSolver::apply: initial L2Norm of solution vector: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "HypreGMRESSolver::apply: initial L2Norm of rhs vector: " << f_norm
                  << std::endl;
        AMP::pout << "HypreGMRESSolver::apply: initial L2Norm of residual: " << current_res
                  << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( current_res ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "HypreGMRESSolver::apply: initial residual below tolerance" << std::endl;
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

    HYPRE_GMRESSolve(
        d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

    copyFromHypre( d_hypre_sol, u );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Query iteration count and store on AMP side
    HYPRE_Int hypre_iters;
    HYPRE_GMRESGetNumIterations( d_solver, &hypre_iters );
    d_iNumberIterations = hypre_iters;
    HYPRE_Real hypre_res;
    HYPRE_GMRESGetFinalRelativeResidualNorm( d_solver, &hypre_res );

    // Check for NaNs
    if ( std::isnan( hypre_res ) ) {
        d_ConvergenceStatus = SolverStatus::DivergedOther;
        AMP_WARNING( "HypreGMRESSolver::apply: Residual norm is NaN" );
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
        AMP::pout << "HypreGMRESSolver::apply: final L2Norm of solution: " << u->L2Norm()
                  << std::endl;
        AMP::pout << "HypreGMRESSolver::apply: final L2Norm of residual: " << current_res
                  << std::endl;
        AMP::pout << "HypreGMRESSolver::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "HypreGMRESSolver::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

void HypreGMRESSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    HYPRE_ParCSRGMRESDestroy( d_solver );
    HYPRE_ParCSRGMRESCreate( d_comm.getCommunicator(), &d_solver );

    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
