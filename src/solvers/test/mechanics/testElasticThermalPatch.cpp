#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/ThermalStrainMaterialModel.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <string>

static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file  = "input_" + exeName;
    std::string output_file = "output_" + exeName + ".txt";
    std::string log_file    = "log_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Read the mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // Create a nonlinear BVP operator for mechanics
    AMP_INSIST( input_db->keyExists( "NonlinearMechanicsOperator" ), "key missing!" );
    auto nonlinearMechanicsBVPoperator =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "NonlinearMechanicsOperator", input_db ) );
    auto nonlinearMechanicsVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() );
    auto mechanicsMaterialModel = nonlinearMechanicsVolumeOperator->getMaterialModel();

    // Create the variables
    auto mechanicsNonlinearVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::MechanicsNonlinearFEOperator>(
            nonlinearMechanicsBVPoperator->getVolumeOperator() );
    auto dispVar    = mechanicsNonlinearVolumeOperator->getOutputVariable();
    auto tempVar    = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
    auto tempDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto dispDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto solVec       = AMP::LinearAlgebra::createVector( dispDofMap, dispVar, true );
    auto rhsVec       = solVec->clone();
    auto resVec       = solVec->clone();
    auto initTempVec  = AMP::LinearAlgebra::createVector( tempDofMap, tempVar, true );
    auto finalTempVec = initTempVec->clone();

    // Set Initial Temperature
    AMP_INSIST( input_db->keyExists( "INIT_TEMP_CONST" ), "key missing!" );
    auto initTempVal = input_db->getScalar<double>( "INIT_TEMP_CONST" );
    initTempVec->setToScalar( initTempVal );
    mechanicsNonlinearVolumeOperator->setReferenceTemperature( initTempVec );

    // Set Final Temperature
    AMP_INSIST( input_db->keyExists( "FINAL_TEMP_CONST" ), "key missing!" );
    auto finalTempVal = input_db->getScalar<double>( "FINAL_TEMP_CONST" );
    finalTempVec->setToScalar( finalTempVal );
    mechanicsNonlinearVolumeOperator->setVector( AMP::Operator::Mechanics::TEMPERATURE,
                                                 finalTempVec );

    // Initial guess
    solVec->zero();
    nonlinearMechanicsBVPoperator->modifyInitialSolutionVector( solVec );

    // RHS
    rhsVec->zero();
    nonlinearMechanicsBVPoperator->modifyRHSvector( rhsVec );

    // Create a Linear BVP operator for mechanics
    auto linearMechanicsBVPoperator = std::make_shared<AMP::Operator::LinearBVPOperator>(
        nonlinearMechanicsBVPoperator->getParameters( "Jacobian", nullptr ) );

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    AMP_INSIST( nonlinearSolver_db->keyExists( "LinearSolver" ), "key missing!" );
    auto linearSolver_db = nonlinearSolver_db->getDatabase( "LinearSolver" );
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver;

    AMP_INSIST( nonlinearSolver_db->keyExists( "usesJacobian" ), "key missing!" );
    bool usesJacobian = nonlinearSolver_db->getScalar<bool>( "usesJacobian" );

    if ( usesJacobian ) {
        auto linearSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
        linearSolverParams->d_pOperator = linearMechanicsBVPoperator;
        linearSolverParams->d_comm      = globalComm;
        linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );
    }

    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinearMechanicsBVPoperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    if ( usesJacobian ) {
        nonlinearSolverParams->d_pNestedSolver = linearSolver;
    }
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );
    nonlinearSolver->setZeroInitialGuess( false );

    if ( !usesJacobian ) {
        linearSolver = nonlinearSolver->getKrylovSolver();
    }

    // We need to reset the linear operator before the solve since TrilinosML does
    // the factorization of the matrix during construction and so the matrix must
    // be correct before constructing the TrilinosML object.
    nonlinearMechanicsBVPoperator->apply( solVec, resVec );
    linearMechanicsBVPoperator->reset(
        nonlinearMechanicsBVPoperator->getParameters( "Jacobian", solVec ) );

    AMP_INSIST( linearSolver_db->keyExists( "Preconditioner" ), "key missing!" );
    auto preconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );
    auto preconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( preconditioner_db );
    preconditionerParams->d_pOperator = linearMechanicsBVPoperator;
    auto preconditioner = std::make_shared<AMP::Solver::TrilinosMLSolver>( preconditionerParams );

    linearSolver->setNestedSolver( preconditioner );

    nonlinearSolver->apply( rhsVec, solVec );

    mechanicsNonlinearVolumeOperator->printStressAndStrain( solVec, output_file );

    ut->passes( "testElasticThermalPatch" );
}


int testElasticThermalPatch( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> mesh_files;
    mesh_files.emplace_back( "testElasticThermalPatch-1" );
    mesh_files.emplace_back( "testElasticThermalPatch-2" );
    mesh_files.emplace_back( "testElasticThermalPatch-3" );
    mesh_files.emplace_back( "testElasticThermalPatch-4" );

    for ( auto &mesh_file : mesh_files )
        myTest( &ut, mesh_file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
