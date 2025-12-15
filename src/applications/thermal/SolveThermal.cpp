#include "AMP/applications/thermal/SolveThermal.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/NonlinearKrylovAccelerator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"


#define USE_NKA 1


namespace AMP::applications {


/***************************************************************************
 * Create the input database for solvers                                    *
 ***************************************************************************/
static std::shared_ptr<AMP::Database> createSolverDatabase()
{
    auto NonlinearDB = std::make_unique<AMP::Database>( "NonlinearSolver" );
#if USE_NKA
    NonlinearDB->putScalar( "name", "NKASolver" );
    NonlinearDB->putScalar( "max_iterations", 2000 );
    NonlinearDB->putScalar( "max_error", 1e-10 );
    NonlinearDB->putScalar( "max_vectors", 100 );      // 3-10
    NonlinearDB->putScalar( "angle_tolerance", 0.15 ); // 0.1-0.2
    NonlinearDB->putScalar( "uses_preconditioner", true );
    NonlinearDB->putScalar( "print_info_level", 1 );
    // NonlinearDB->putScalar( "absolute_tolerance", 1e-12 );
    // NonlinearDB->putScalar( "relative_tolerance", 1e-5 );
    NonlinearDB->putScalar( "absolute_tolerance", 5e-5 );
    NonlinearDB->putScalar( "relative_tolerance", 1e-4 );
#else
    NonlinearDB->putScalar( "name", "PetscSNESSolver" );
    NonlinearDB->putScalar( "max_iterations", 500 );
    NonlinearDB->putScalar( "max_error", 1e-10 );
    NonlinearDB->putScalar( "absolute_tolerance", 1e-10 );
    NonlinearDB->putScalar( "relative_tolerance", 1e-9 );
    NonlinearDB->putScalar( "stepTolerance", 1e-10 );
    NonlinearDB->putScalar( "maximumFunctionEvals", 100 );
    NonlinearDB->putScalar( "usesJacobian", false );
    NonlinearDB->putScalar( "SNESOptions",
                            "-snes_monitor -snes_type ls -snes_converged_reason -snes_ksp_ew" );
    auto LinearDB = NonlinearDB->createAddDatabase( "LinearSolver" );
    #if 1
    LinearDB->putScalar( "name", "PetscKrylovSolver" );
    LinearDB->putScalar( "max_iterations", 500 );
    LinearDB->putScalar( "max_error", 1e-10 );
    LinearDB->putScalar( "ksp_type", "fgmres" );
    LinearDB->putScalar( "absolute_tolerance", 1e-12 );
    LinearDB->putScalar( "relative_tolerance", 1e-4 );
    LinearDB->putScalar( "divergence_tolerance", 1e3 );
    LinearDB->putScalar( "max_krylov_dimension", 40 );
    // LinearDB->putScalar( "uses_preconditioner", false );
    LinearDB->putScalar( "uses_preconditioner", true );
    LinearDB->putScalar( "pc_type", "shell" );
    LinearDB->putScalar( "pc_side", "RIGHT" );
    LinearDB->putScalar(
        "KSPOptions",
        "-ksp_monitor -ksp_converged_reason -ksp_max_it 500 -ksp_rtol 1e-3 -ksp_atol 1e-13 " );
    #else
    LinearDB->putScalar( "name", "GMRESSolver" );
    LinearDB->putScalar( "uses_preconditioner", false );
    LinearDB->putScalar( "print_info_level", 1 );
    LinearDB->putScalar( "max_iterations", 30 );
    LinearDB->putScalar( "max_error", 1e-10 );
    #endif
#endif
    return NonlinearDB;
}


/***************************************************************************
 * Solve for the temperature                                                *
 ***************************************************************************/
std::tuple<std::shared_ptr<AMP::LinearAlgebra::Vector>, std::shared_ptr<AMP::Operator::Operator>>
solveTemperature( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                  std::shared_ptr<const AMP::LinearAlgebra::Vector> rhs,
                  std::shared_ptr<const AMP::Database> input_db,
                  std::shared_ptr<const AMP::LinearAlgebra::Vector> initialGuess )
{
    PROFILE( "solveTemperature" );

    // Create the operator and solution vector
    auto [nonlinearOp, solVec] = createThermalOperatorsFE( mesh, input_db );

    // Integrate Rhs
    auto rhsVec = solVec->clone();
    rhsVec->copyVector( rhs );

    // Initial guess
    if ( initialGuess )
        solVec->copy( *initialGuess );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Create the linear operator
    auto linearOpParams = nonlinearOp->getParameters( "Jacobian", solVec );
    std::shared_ptr<AMP::Operator::Operator> linearOp =
        AMP::Operator::OperatorFactory::create( linearOpParams );
    auto linearColumnOperator =
        std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>( linearOp );
    AMP_ASSERT( linearColumnOperator );

    // Create the preconditioner
    auto precond_db = std::make_shared<AMP::Database>( "Preconditioner" );
    precond_db->putScalar( "max_iterations", 3 );
    precond_db->putScalar( "max_levels", 5 );
    precond_db->putScalar( "max_error", 1e-15 );
    precond_db->putScalar( "name", "TrilinosMLSolver" );
    auto precondParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( precond_db );
    precondParams->d_pOperator = linearColumnOperator;
    auto preconditioner        = std::make_shared<AMP::Solver::ColumnSolver>( precondParams );
    for ( auto op : *linearColumnOperator ) {
        auto params         = std::make_shared<AMP::Solver::SolverStrategyParameters>( precond_db );
        params->d_pOperator = op;
        preconditioner->append( AMP::Solver::SolverFactory::create( params ) );
    }

    // Create the solvers
    auto nonlinearSolver_db = createSolverDatabase();
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = mesh->getComm();
    nonlinearSolverParams->d_pOperator     = nonlinearOp;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    nonlinearSolverParams->d_vectors.resize( 1 );
    nonlinearSolverParams->d_vectors[0] = solVec;
    auto nonlinearSolver = AMP::Solver::SolverFactory::create( nonlinearSolverParams );
    nonlinearSolver->initialize( nonlinearSolverParams );
    if ( nonlinearSolver->type() == "PetscSNESSolver" ) {
        auto krylovSolver = nonlinearSolver->getNestedSolver();
        AMP_ASSERT( krylovSolver );
        krylovSolver->setNestedSolver( preconditioner );
    } else {
        nonlinearSolver->setNestedSolver( preconditioner );
    }

    // Solve
    auto resVec = solVec->clone();
    nonlinearOp->residual( rhsVec, solVec, resVec );
    AMP::pout << "Initial Residual Norm: " << resVec->L2Norm() << std::endl;
    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->apply( rhsVec, solVec );
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    resVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    nonlinearOp->residual( rhsVec, solVec, resVec );
    AMP::pout << "Final Residual Norm: " << resVec->L2Norm() << std::endl;
    AMP::pout << "Final Solution Norm: " << solVec->L2Norm() << std::endl;
    AMP::pout << "Final Rhs Norm: " << rhsVec->L2Norm() << std::endl;

    // Return
    std::tuple<std::shared_ptr<AMP::LinearAlgebra::Vector>,
               std::shared_ptr<AMP::Operator::Operator>>
        rtn;
    std::get<0>( rtn ) = solVec;
    std::get<1>( rtn ) = nonlinearOp;
    return rtn;
}


} // namespace AMP::applications
