#include "AMP/solvers/hypre/HyprePCGSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
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
    setHypreFunctionPointers();
    setupHypreSolver( parameters );
}

void HyprePCGSolver::setHypreFunctionPointers()
{
    d_hypreSolve                = HYPRE_PCGSolve;
    d_hypreGetNumIterations     = HYPRE_PCGGetNumIterations;
    d_hypreSetPreconditioner    = HYPRE_PCGSetPrecond;
    d_hypreSolverSetup          = HYPRE_PCGSetup;
    d_hypreSetRelativeTolerance = HYPRE_PCGSetTol;
    d_hypreSetAbsoluteTolerance = HYPRE_PCGSetAbsoluteTol;
    d_hypreSetMaxIterations     = HYPRE_PCGSetMaxIter;
    d_hypreSetPrintLevel        = HYPRE_PCGSetPrintLevel;
    d_hypreSetLogging           = HYPRE_PCGSetLogging;
    d_hypreCreateSolver         = HYPRE_ParCSRPCGCreate;
    d_hypreDestroySolver        = HYPRE_ParCSRPCGDestroy;
}

HyprePCGSolver::~HyprePCGSolver() {}

void HyprePCGSolver::setupHypreSolver( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HyprePCGSolver::setupHypreSolver" );

    createHypreSolver();
    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HyprePCGSolver::initialize( parameters );
    }

    HypreSolver::setupSolver();
}

void HyprePCGSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HyprePCGSolver::initialize" );

    AMP_ASSERT( parameters );
    HyprePCGSolver::getFromInput( parameters->d_db );
    setupNestedSolver( parameters );
}

void HyprePCGSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    HypreSolver::setCommonParameters( db );

    if ( d_bComputeResidual ) {
        HYPRE_PCGSetRecomputeResidual( d_solver, d_bComputeResidual );
    }

    if ( db->keyExists( "compute_residual_p" ) ) {
        HYPRE_PCGSetRecomputeResidualP( d_solver, 1 );
    }
}

void HyprePCGSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    PROFILE( "HyprePCGSolver::reset" );

    destroyHypreSolver();
    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
