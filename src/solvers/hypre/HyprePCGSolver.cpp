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
HyprePCGSolver::HyprePCGSolver() : HypreSolver() {}
HyprePCGSolver::HyprePCGSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    HYPRE_ParCSRPCGCreate( d_comm.getCommunicator(), &d_solver );
    setupHypreSolver( parameters );
    setHypreFunctionPointers();
}

void HyprePCGSolver::setHypreFunctionPointers()
{
    d_hypreSolve          = HYPRE_PCGSolve;
    getHypreNumIterations = HYPRE_PCGGetNumIterations;
}

HyprePCGSolver::~HyprePCGSolver() { HYPRE_ParCSRPCGDestroy( d_solver ); }

void HyprePCGSolver::setupHypreSolver( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HyprePCGSolver::setupHypreSolver" );

    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HyprePCGSolver::initialize( parameters );
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
            HYPRE_Solver pcg_precond = NULL;
            HYPRE_PCGSetPrecond( d_solver,
                                 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScale,
                                 (HYPRE_PtrToSolverFcn) HYPRE_ParCSRDiagScaleSetup,
                                 pcg_precond );
        } else {
            auto pc = std::dynamic_pointer_cast<HypreSolver>( d_pNestedSolver );
            if ( pc ) {

                auto pcg_precond = pc->getHYPRESolver();
                AMP_ASSERT( pcg_precond );

                if ( pc->type() == "BoomerAMGSolver" ) {
                    HYPRE_PCGSetPreconditioner( d_solver, pcg_precond );
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

    HYPRE_PCGSetup( d_solver, (HYPRE_Matrix) parcsr_A, (HYPRE_Vector) par_x, (HYPRE_Vector) par_x );
}

void HyprePCGSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;

    HyprePCGSolver::getFromInput( db );

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

void HyprePCGSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    HYPRE_ParCSRPCGDestroy( d_solver );
    HYPRE_ParCSRPCGCreate( d_comm.getCommunicator(), &d_solver );

    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
