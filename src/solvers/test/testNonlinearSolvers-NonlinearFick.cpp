#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>

static void myTest( AMP::UnitTest *ut, const std::string &inputName )
{
    std::string input_file = inputName;
    std::string log_file   = "output_" + inputName;

    AMP::pout << "Running with input " << input_file << std::endl;
    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    auto DOF_scalar = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    // create a nonlinear BVP operator for nonlinear fick diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearFickOperator" ), "key missing!" );

    auto nonlinearFickOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearFickOperator", input_db ) );

    // initialize the input variable
    auto fickVariable = nonlinearFickOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto solVec = AMP::LinearAlgebra::createVector( DOF_scalar, fickVariable );
    auto rhsVec = AMP::LinearAlgebra::createVector( DOF_scalar, fickVariable );
    auto resVec = AMP::LinearAlgebra::createVector( DOF_scalar, fickVariable );

    // Initial guess
    solVec->setToScalar( .05 );
    AMP::pout << "initial guess norm = " << solVec->L2Norm() << "\n";
    nonlinearFickOperator->modifyInitialSolutionVector( solVec );
    AMP::pout << "initial guess norm  after apply = " << solVec->L2Norm() << "\n";

    rhsVec->setToScalar( 0.0 );
    nonlinearFickOperator->modifyRHSvector( rhsVec );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, solVec, nonlinearFickOperator );

    nonlinearFickOperator->residual( rhsVec, solVec, resVec );

    AMP::pout << "Initial Residual Norm: " << resVec->L2Norm() << std::endl;

    nonlinearSolver->setZeroInitialGuess( false );

    nonlinearSolver->apply( rhsVec, solVec );

    nonlinearFickOperator->residual( rhsVec, solVec, resVec );

    auto finalResidualNorm = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > 1.0e-08 ) {
        ut->failure( inputName );
    } else {
        ut->passes( inputName );
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> fileNames;

    if ( argc > 1 ) {
        fileNames.push_back( argv[1] );
    } else {
#ifdef AMP_USE_PETSC
    #ifdef AMP_USE_HYPRE
        fileNames.emplace_back( "input_testPetscSNESSolver-NonlinearFick-cylinder-1a" );
        fileNames.emplace_back( "input_testPetscSNESSolver-NonlinearFick-cylinder-1b" );
        fileNames.emplace_back( "input_testPetscSNESSolver-NonlinearFick-cylinder-1c" );
        fileNames.emplace_back( "input_testPetscSNESSolver-NonlinearFick-cylinder-1d" );
    #endif
#endif
    }

    for ( auto &fileName : fileNames )
        myTest( &ut, fileName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
