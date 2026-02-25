#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>


void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    AMP::pout << "Running with input " << input_file << std::endl;

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    //   Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // now construct the linear BVP operator for Fick
    AMP_INSIST( input_db->keyExists( "testLinearFickOperator" ), "key missing!" );
    auto linearFickOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testLinearFickOperator", input_db ) );

    // now construct the linear BVP operator for thermal
    AMP_INSIST( input_db->keyExists( "testLinearThermalOperator" ), "key missing!" );
    auto linearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "testLinearThermalOperator", input_db ) );

    // create a column operator object for linear Thermal-Fick
    auto linearThermalFickOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    linearThermalFickOperator->append( linearThermalOperator );
    linearThermalFickOperator->append( linearFickOperator );

    // initialize the input multi-variable
    auto thermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
            linearThermalOperator->getVolumeOperator() );
    auto FickVolumeOperator = std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
        linearFickOperator->getVolumeOperator() );

    auto inputVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "inputVariable" );
    inputVariable->add( thermalVolumeOperator->getInputVariable() );
    inputVariable->add( FickVolumeOperator->getInputVariable() );

    // initialize the output multi-variable
    auto outputVariable = linearThermalFickOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto solVec = AMP::LinearAlgebra::createVector( nodalDofMap, inputVariable );
    auto rhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable );
    auto resVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    // Initial-Guess for thermal
    auto dirichletThermalInVecOp =
        std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "ThermalInitialGuess", input_db ) );
    dirichletThermalInVecOp->setVariable( thermalVolumeOperator->getInputVariable() );

    // Initial-Guess for Fick
    auto dirichletFickInVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "FickInitialGuess", input_db ) );
    dirichletFickInVecOp->setVariable( FickVolumeOperator->getInputVariable() );

    // Random initial guess
    solVec->setRandomValues();

    // Initial guess for thermal must satisfy the thermal Dirichlet boundary conditions
    dirichletThermalInVecOp->apply( nullVec, solVec );

    // Initial guess for Fick must satisfy the Fick Dirichlet boundary conditions
    dirichletFickInVecOp->apply( nullVec, solVec );

    rhsVec->setToScalar( 0.0 );

    // ---- initialize the solvers
    auto columnSolver_db = input_db->getDatabase( "ColumnSolver" );
    auto columnSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( columnSolver_db );
    auto columnSolver = std::make_shared<AMP::Solver::ColumnSolver>( columnSolverParams );

    auto thermalSolver_db = columnSolver_db->getDatabase( "ThermalSolver" );
    auto thermalSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( thermalSolver_db );
    thermalSolverParams->d_pOperator = linearThermalOperator;
    auto linearThermalSolver =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( thermalSolverParams );

    auto FickSolver_db = columnSolver_db->getDatabase( "FickSolver" );
    auto FickSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( FickSolver_db );
    FickSolverParams->d_pOperator = linearFickOperator;
    auto linearFickSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( FickSolverParams );

    columnSolver->append( linearThermalSolver );
    columnSolver->append( linearFickSolver );

    double initialResidualNorm = static_cast<double>( resVec->L2Norm() );

    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    columnSolver->apply( rhsVec, solVec );

    linearThermalFickOperator->residual( rhsVec, solVec, resVec );

    double finalResidualNorm = static_cast<double>( resVec->L2Norm() );

    std::cout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > 1.0e-08 ) {
        ut->failure( "ColumnSolver unsuccessfully solves two linear operators" );
    } else {
        ut->passes( "ColumnSolver successfully solves two linear operators" );
    }
    ut->passes( exeName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testColumnSolver-LinearThermalAndFick-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
