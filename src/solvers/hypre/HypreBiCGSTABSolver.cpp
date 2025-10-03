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
    setHypreFunctionPointers();
}

HypreBiCGSTABSolver::~HypreBiCGSTABSolver() { HYPRE_ParCSRBiCGSTABDestroy( d_solver ); }

void HypreBiCGSTABSolver::setHypreFunctionPointers()
{
    d_hypreSolve          = HYPRE_BiCGSTABSolve;
    getHypreNumIterations = HYPRE_BiCGSTABGetNumIterations;
}

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

void HypreBiCGSTABSolver::reset( std::shared_ptr<SolverStrategyParameters> params )
{
    HYPRE_ParCSRBiCGSTABDestroy( d_solver );
    HYPRE_ParCSRBiCGSTABCreate( d_comm.getCommunicator(), &d_solver );

    HypreSolver::reset( params );
    setupHypreSolver( params );
}

} // namespace AMP::Solver
