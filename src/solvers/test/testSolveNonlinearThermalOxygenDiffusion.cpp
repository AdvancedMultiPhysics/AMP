#include "AMP/IO/PIO.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &inputName )
{
    std::string input_file = inputName;
    AMP::pout << "Running with input " << input_file << std::endl;
    std::string log_file = "output_" + inputName;

    AMP::logOnlyNodeZero( log_file );

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // create the Mesh
    const auto mesh = createMesh( input_db );

    auto [nodalDOFMap, gaussDOFMap] = getDofMaps( mesh );

    // create a nonlinear BVP operator for nonlinear oxygen diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearOxygenOperator" ), "key missing!" );

    auto nonlinearOxygenOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearOxygenOperator", input_db ) );

    // create a nonlinear BVP operator for nonlinear thermal diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearThermalOperator" ), "key missing!" );

    auto nonlinearThermalDatabase = input_db->getDatabase( "testNonlinearThermalOperator" );
    auto nonlinearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearThermalOperator", input_db ) );

    // create a column operator object for nonlinear thermal-oxygen diffusion
    auto nonlinearThermalOxygenOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    nonlinearThermalOxygenOperator->append( nonlinearOxygenOperator );
    nonlinearThermalOxygenOperator->append( nonlinearThermalOperator );

    auto oxygenVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearOxygenOperator->getVolumeOperator() );
    auto thermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );

    auto oxygenVariable  = oxygenVolumeOperator->getOutputVariable();
    auto thermalVariable = thermalVolumeOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto solVec = AMP::LinearAlgebra::createVector(
        nodalDOFMap, nonlinearThermalOxygenOperator->getOutputVariable(), true );
    auto rhsVec = solVec->clone();
    auto resVec = solVec->clone();

    // just making sure
    rhsVec->zero();

    // initialize the output multi-variable
    auto columnVariable = nonlinearThermalOxygenOperator->getOutputVariable();

    // create the following shared pointers for ease of use
    auto oxygenNlSolVec  = solVec->subsetVectorForVariable( oxygenVariable );
    auto thermalNlSolVec = solVec->subsetVectorForVariable( thermalVariable );

    // Random initial guess
    solVec->setRandomValues();
    const double referenceTemperature = 750.0;
    thermalNlSolVec->addScalar( *thermalNlSolVec, referenceTemperature );
    oxygenNlSolVec->scale( 0.01, *oxygenNlSolVec );

    nonlinearOxygenOperator->modifyInitialSolutionVector( solVec );
    nonlinearThermalOperator->modifyInitialSolutionVector( solVec );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, solVec, nonlinearThermalOxygenOperator );

    nonlinearThermalOxygenOperator->residual( rhsVec, solVec, resVec );
    auto initialResidualNorm = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    nonlinearSolver->setZeroInitialGuess( false );

    nonlinearSolver->apply( rhsVec, solVec );

    nonlinearThermalOxygenOperator->residual( rhsVec, solVec, resVec );

    auto finalResidualNorm = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > initialResidualNorm * 1.0e-10 + 1.0e-05 ) {
        ut->failure( "the Final Residual is larger than the tolerance" );
    } else {
        ut->passes( "PetscSNES Solver successfully solves a nonlinear thermal-oxygen diffusion "
                    "equation with JFNK, "
                    "FGMRES for Krylov, block diagonal preconditioning with ML solvers" );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> inputNames;

    if ( argc > 1 ) {
        inputNames.push_back( argv[1] );
    } else {

#ifdef AMP_USE_PETSC
    #ifdef AMP_USE_TRILINOS_ML
        //  inputNames.push_back("testPetscSNESSolver-NonlinearThermalOxygenDiffusion-1");
        inputNames.emplace_back( "input_testPetscSNESSolver-NonlinearThermalOxygenDiffusion-2" );
    #endif
#endif
    }

    for ( auto &inputName : inputNames )
        myTest( &ut, inputName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
