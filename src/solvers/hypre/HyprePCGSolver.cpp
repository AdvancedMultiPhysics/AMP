#include "AMP/solvers/hypre/HyprePCGSolver.h"
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
HyprePCGSolver::HyprePCGSolver() : HypreSolver() { d_bCreationPhase = true; }
HyprePCGSolver::HyprePCGSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    HYPRE_ParCSRPCGCreate( d_comm.getCommunicator(), &d_solver );

    AMP_ASSERT( parameters );
    HyprePCGSolver::initialize( parameters );
}

HyprePCGSolver::~HyprePCGSolver() { HYPRE_ParCSRPCGDestroy( d_solver ); }

void HyprePCGSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;

    HyprePCGSolver::getFromInput( db );

    if ( parameters->d_pNestedSolver ) {
        d_pPreconditioner = parameters->d_pNestedSolver;
    } else {
        if ( d_bUsesPreconditioner ) {
            auto pcName  = db->getWithDefault<std::string>( "pc_solver_name", "Preconditioner" );
            auto outerDB = db->keyExists( pcName ) ? db : parameters->d_global_db;
            if ( outerDB ) {
                auto pcDB       = outerDB->getDatabase( pcName );
                auto parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcDB );
                parameters->d_pOperator = d_pOperator;
                d_pPreconditioner       = AMP::Solver::SolverFactory::create( parameters );
                AMP_ASSERT( d_pPreconditioner );
            }
        }
    }
}

void HyprePCGSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    if ( db->keyExists( "logging" ) ) {
        const auto logging = db->getScalar<int>( "logging" );
        HYPRE_PCGSetLogging( d_solver, logging );
    }

    HYPRE_PCGSetTol( d_solver, static_cast<HYPRE_Real>( d_dRelativeTolerance ) );
    HYPRE_PCGSetAbsoluteTol( d_solver, static_cast<HYPRE_Real>( d_dAbsoluteTolerance ) );
    HYPRE_PCGSetMaxIter( d_solver, d_iMaxIterations );
    HYPRE_PCGSetPrintLevel( d_solver, d_iDebugPrintInfoLevel );

    d_bUsesPreconditioner = db->getWithDefault<bool>( "uses_preconditioner", false );
    d_bDiagScalePC        = db->getWithDefault<bool>( "diag_scale_pc", false );

    if ( d_bComputeResidual ) {
        HYPRE_PCGSetRecomputeResidual( d_solver, d_bComputeResidual );
    }

    if ( db->keyExists( "compute_residual_p" ) ) {
        HYPRE_PCGSetRecomputeResidualP( d_solver, 1 );
    }
}

void HyprePCGSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                            std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "solve" );
    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: HyprePCGSolver::apply() operator cannot be NULL" );

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
            AMP::pout << "HyprePCGSolver::apply(), L2 norm of residual before solve "
                      << std::setprecision( 15 ) << initialResNorm << std::endl;
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        HYPRE_Real solution_norm( u->L2Norm() );
        AMP::pout << "HyprePCGSolver : before solve solution norm: " << std::setprecision( 15 )
                  << solution_norm << std::endl;
    }

    if ( d_bUsesPreconditioner ) {
        if ( d_bDiagScalePC ) {
            HYPRE_Solver pcg_precond = NULL;
            HYPRE_PCGSetPrecond( d_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                 pcg_precond );
        } else {
            auto pc = std::dynamic_pointer_cast<HypreSolver>( d_pPreconditioner );
            if ( pc ) {

                auto pcg_precond = pc->getHYPRESolver();
                AMP_ASSERT( pcg_precond );

                if ( pc->type() == "BoomerAMGSolver" ) {
                    HYPRE_PCGSetPrecond( d_solver,
                                         (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                         (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup,
                                         pcg_precond );

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

    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_ParVector par_b;
    HYPRE_ParVector par_x;

    HYPRE_IJMatrixGetObject( d_ijMatrix, (void **) &parcsr_A );
    hypre_ParCSRMatrixMigrate( parcsr_A, d_memory_location );

    HYPRE_IJVectorGetObject( d_hypre_rhs, (void **) &par_b );
    HYPRE_IJVectorGetObject( d_hypre_sol, (void **) &par_x );

    // add in code for solve here
    HYPRE_PCGSetup( d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );
    HYPRE_PCGSolve( d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_b, (HYPRE_Vector) par_x );

    copyFromHypre( d_hypre_sol, u );

    // Check for NaNs in the solution (no communication necessary)
    auto localNorm = u->getVectorOperations()->localL2Norm( *u->getVectorData() ).get<HYPRE_Real>();
    AMP_INSIST( localNorm == localNorm, "NaNs detected in solution" );

    // we are forced to update the state of u here
    // as Hypre is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    HYPRE_PCGGetNumIterations( d_solver, &d_iNumberIterations );
    HYPRE_Real hypre_norm;
    HYPRE_PCGGetFinalRelativeResidualNorm( d_solver, &hypre_norm );
    d_dResidualNorm = hypre_norm;

    if ( d_dResidualNorm < d_dRelativeTolerance ) {
        d_ConvergenceStatus = SolverStatus::ConvergedOnRelTol;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "HyprePCGSolver : after solve solution norm: " << std::setprecision( 15 )
                  << u->L2Norm() << std::endl;
    }

    if ( d_bComputeResidual ) {
        d_pOperator->residual( f, u, r );
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "HyprePCGSolver::apply(), L2 norm of residual after solve "
                      << std::setprecision( 15 ) << r->L2Norm() << std::endl;
        }
    }
}

} // namespace AMP::Solver
