#include "AMP/solvers/hypre/HypreBiCGSTABSolver.h"
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
HypreBiCGSTABSolver::HypreBiCGSTABSolver() : HypreSolver() {}
HypreBiCGSTABSolver::HypreBiCGSTABSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    setHypreFunctionPointers();
    setupHypreSolver( parameters );
}

HypreBiCGSTABSolver::~HypreBiCGSTABSolver() {}

void HypreBiCGSTABSolver::setHypreFunctionPointers()
{
    d_hypreSolve                = HYPRE_BiCGSTABSolve;
    d_hypreGetNumIterations     = HYPRE_BiCGSTABGetNumIterations;
    d_hypreSetPreconditioner    = HYPRE_BiCGSTABSetPrecond;
    d_hypreSolverSetup          = HYPRE_BiCGSTABSetup;
    d_hypreSetRelativeTolerance = HYPRE_BiCGSTABSetTol;
    d_hypreSetAbsoluteTolerance = HYPRE_BiCGSTABSetAbsoluteTol;
    d_hypreSetMaxIterations     = HYPRE_BiCGSTABSetMaxIter;
    d_hypreSetPrintLevel        = HYPRE_BiCGSTABSetPrintLevel;
    d_hypreSetLogging           = HYPRE_BiCGSTABSetLogging;
    d_hypreCreateSolver         = HYPRE_ParCSRBiCGSTABCreate;
    d_hypreDestroySolver        = HYPRE_ParCSRBiCGSTABDestroy;
}

void HypreBiCGSTABSolver::setupHypreSolver(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HypreBiCGSTABSolver::setupHypreSolver" );

    createHypreSolver();

    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HypreBiCGSTABSolver::initialize( parameters );
    }

    HypreSolver::setupSolver();
}

void HypreBiCGSTABSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );
    HypreBiCGSTABSolver::getFromInput( parameters->d_db );
    setupNestedSolver( parameters );
}

void HypreBiCGSTABSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    HypreSolver::setCommonParameters( db );
}

void HypreBiCGSTABSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    destroyHypreSolver();
    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
