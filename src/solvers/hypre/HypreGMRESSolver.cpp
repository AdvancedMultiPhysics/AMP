#include "AMP/solvers/hypre/HypreGMRESSolver.h"
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
HypreGMRESSolver::HypreGMRESSolver() : HypreSolver() {}
HypreGMRESSolver::HypreGMRESSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : HypreSolver( parameters )
{
    setHypreFunctionPointers();
    setupHypreSolver( parameters );
}

HypreGMRESSolver::~HypreGMRESSolver() {}

void HypreGMRESSolver::setHypreFunctionPointers()
{
    d_hypreSolve                = HYPRE_GMRESSolve;
    d_hypreGetNumIterations     = HYPRE_GMRESGetNumIterations;
    d_hypreSetPreconditioner    = HYPRE_GMRESSetPrecond;
    d_hypreSolverSetup          = HYPRE_GMRESSetup;
    d_hypreSetRelativeTolerance = HYPRE_GMRESSetTol;
    d_hypreSetAbsoluteTolerance = HYPRE_GMRESSetAbsoluteTol;
    d_hypreSetMaxIterations     = HYPRE_GMRESSetMaxIter;
    d_hypreSetPrintLevel        = HYPRE_GMRESSetPrintLevel;
    d_hypreSetLogging           = HYPRE_GMRESSetLogging;
    d_hypreCreateSolver         = HYPRE_ParCSRGMRESCreate;
    d_hypreDestroySolver        = HYPRE_ParCSRGMRESDestroy;
}

void HypreGMRESSolver::setupHypreSolver(
    std::shared_ptr<const SolverStrategyParameters> parameters )
{
    PROFILE( "HypreGMRESSolver::setupHypreSolver" );

    createHypreSolver();

    // this routine should assume that the solver, matrix and vectors have been created
    // so that it can be used both in the constructor and in reset
    if ( parameters ) {

        HypreGMRESSolver::initialize( parameters );
    }

    HypreSolver::setupSolver();
}

void HypreGMRESSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );
    HypreGMRESSolver::getFromInput( parameters->d_db );
    setupNestedSolver( parameters );
}

void HypreGMRESSolver::getFromInput( std::shared_ptr<const AMP::Database> db )
{
    HypreSolver::setCommonParameters( db );

    d_iMaxKrylovDim = db->getWithDefault<int>( "max_krylov_dimension", 100 );
    HYPRE_GMRESSetKDim( d_solver, static_cast<HYPRE_Int>( d_iMaxKrylovDim ) );

    if ( !d_bComputeResidual ) {
        HYPRE_GMRESSetSkipRealResidualCheck( d_solver, d_bComputeResidual );
    }
}

void HypreGMRESSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    destroyHypreSolver();
    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
