#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>


static void
fickTest( AMP::UnitTest *ut, const std::string &inputName, std::vector<double> &results )
{
    std::string input_file = inputName;
    std::string log_file   = "output_" + inputName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( params );
    auto mesh    = manager->Subset( "cylinder" );

    // create a nonlinear BVP operator for nonlinear fick diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearFickOperator" ), "key missing!" );

    auto nonlinearFickOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testNonlinearFickOperator", input_db ) );

    // initialize the input variable
    auto fickVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearFickOperator->getVolumeOperator() );

    auto fickVariable = fickVolumeOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto solVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, fickVariable, true );
    auto rhsVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, fickVariable, true );
    auto resVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, fickVariable, true );

    // Initial guess
    solVec->setToScalar( .05 );
    std::cout << "initial guess norm = " << solVec->L2Norm() << "\n";
    nonlinearFickOperator->modifyInitialSolutionVector( solVec );
    std::cout << "initial guess norm  after apply = " << solVec->L2Norm() << "\n";
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
    std::cout << "Final Residual Norm: " << resVec->L2Norm() << std::endl;

    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // store result
    {
        auto iterator   = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        size_t numNodes = iterator.size();
        results.resize( numNodes );
        std::vector<size_t> dofs;
        for ( size_t iNode = 0; iNode < numNodes; iNode++ ) {
            nodalScalarDOF->getDOFs( iterator->globalID(), dofs );
            size_t gid     = dofs[0];
            results[iNode] = solVec->getValueByGlobalID( gid );
            ++iterator;
        }
    }

    ut->passes( inputName );
}


static void
fickSoretTest( AMP::UnitTest *ut, const std::string &inputName, std::vector<double> &results )
{
    std::string input_file = inputName;
    std::string log_file   = "output_" + inputName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( params );
    auto mesh    = manager->Subset( "cylinder" );

    // create a nonlinear BVP operator for nonlinear Fick-Soret diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearFickSoretBVPOperator" ), "key missing!" );

    // Create nonlinear FickSoret BVP operator and access volume nonlinear FickSoret operator
    auto nlinBVPOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "testNonlinearFickSoretBVPOperator", input_db );
    auto nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    auto nlinOp = std::dynamic_pointer_cast<AMP::Operator::FickSoretNonlinearFEOperator>(
        nlinBVPOp->getVolumeOperator() );
    auto fickOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinOp->getFickOperator() );
    auto soretOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinOp->getSoretOperator() );

    auto tVar = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
    auto cVar = std::make_shared<AMP::LinearAlgebra::Variable>( *fickOp->getOutputVariable() );
    auto fsOutVar =
        std::make_shared<AMP::LinearAlgebra::Variable>( *nlinBVPOp->getOutputVariable() );

    // create solution, rhs, and residual vectors
    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto solVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, cVar, true );
    auto rhsVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, fsOutVar, true );
    auto resVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, fsOutVar, true );

    // create parameters for reset test and reset fick and soret operators
    auto tVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, tVar, true );
    tVec->setToScalar( 300. );

    fickOp->setVector( "temperature", tVec );
    soretOp->setVector( "temperature", tVec );

    // Initial guess
    solVec->setToScalar( .05 );
    std::cout << "initial guess norm = " << solVec->L2Norm() << "\n";
    nlinBVPOp->modifyInitialSolutionVector( solVec );
    std::cout << "initial guess norm  after apply = " << solVec->L2Norm() << "\n";

    rhsVec->setToScalar( 0.0 );
    nlinBVPOp->modifyRHSvector( rhsVec );

    // Create the solver
    auto nonlinearSolver = AMP::Solver::Test::buildSolver(
        "NonlinearSolver", input_db, globalComm, solVec, nlinBVPOperator );

    nlinBVPOp->residual( rhsVec, solVec, resVec );
    AMP::pout << "Initial Residual Norm: " << resVec->L2Norm() << std::endl;

    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->apply( rhsVec, solVec );
    nlinBVPOp->residual( rhsVec, solVec, resVec );
    std::cout << "Final Residual Norm: " << resVec->L2Norm() << std::endl;

    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // store result
    {
        auto iterator   = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        size_t numNodes = iterator.size();
        results.resize( numNodes );
        std::vector<size_t> dofs;
        for ( size_t iNode = 0; iNode < numNodes; iNode++ ) {
            nodalScalarDOF->getDOFs( iterator->globalID(), dofs );
            size_t gid     = dofs[0];
            results[iNode] = solVec->getValueByGlobalID( gid );
            ++iterator;
        }
    }

    ut->passes( inputName );
}


int testNonlinearSolvers_NonlinearFickSoret_1( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<double> fickOnly, fickSoretOff, fickSoretZero, fickOnlyReal, fickSoretOffReal;

    fickTest( &ut, "input_testPetscSNESSolver-NonlinearFick-cylinder-TUI-1", fickOnly );
    fickTest( &ut, "input_testPetscSNESSolver-NonlinearFick-cylinder-TUI-2", fickOnlyReal );
    fickSoretTest(
        &ut, "input_testPetscSNESSolver-NonlinearFickSoret-cylinder-TUI-1", fickSoretOff );
    fickSoretTest(
        &ut, "input_testPetscSNESSolver-NonlinearFickSoret-cylinder-TUI-2", fickSoretZero );
    fickSoretTest(
        &ut, "input_testPetscSNESSolver-NonlinearFickSoret-cylinder-TUI-3", fickSoretOffReal );

    AMP_INSIST( fickOnly.size() == fickSoretOff.size() &&
                    fickSoretOff.size() == fickSoretZero.size() &&
                    fickOnlyReal.size() == fickSoretOffReal.size(),
                "sizes of results do not match" );

    double l2err1 = 0., l2err2 = 0.;
    for ( size_t i = 0; i < fickOnly.size(); i++ ) {
        double err = fickOnly[i] - fickSoretOff[i];
        l2err1 += err * err;
        err = fickSoretOff[i] - fickSoretZero[i];
        l2err2 += err * err;
    }
    l2err1 = std::sqrt( l2err1 );
    l2err2 = std::sqrt( l2err2 );

    std::cout << "fick/soretOff err = " << l2err1 << "  soretOff/soretZero err = " << l2err2
              << std::endl;

    double l2err3 = 0.;
    for ( size_t i = 0; i < fickOnlyReal.size(); i++ ) {
        double err = fickOnlyReal[i] - fickSoretOffReal[i];
        l2err3 += err * err;
    }
    l2err3 = std::sqrt( l2err3 );

    std::cout << "fick/soretOff real err = " << l2err3 << std::endl;

    if ( ( l2err1 < 1.e-6 ) && ( l2err2 < 1.e-6 ) && ( l2err3 < 1.e-6 ) ) {
        ut.passes( "fick, fick-soret/off, and fick-soret/zero all agree" );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
