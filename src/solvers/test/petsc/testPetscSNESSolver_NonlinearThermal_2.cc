#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolverParameters.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolverParameters.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/InputDatabase.h"
#include "AMP/utils/InputManager.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/utils/shared_ptr.h"
#include "AMP/vectors/VectorBuilder.h"


#include <iostream>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );

    auto input_db = AMP::make_shared<AMP::InputDatabase>( "input_db" );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );
    input_db->printClassData( AMP::plog );

    // Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = AMP::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::Mesh::buildMesh( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Volume, gaussPointGhostWidth, DOFsPerElement, split );
    AMP::pout << "Constructing Nonlinear Thermal Operator..." << std::endl;

    // create a nonlinear BVP operator for nonlinear thermal diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearThermalOperator" ), "key missing!" );
    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel;
    auto nonlinearThermalOperator = AMP::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "testNonlinearThermalOperator", input_db, thermalTransportModel ) );

    // initialize the input variable
    auto thermalVolumeOperator =
        AMP::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );
    auto thermalVariable = thermalVolumeOperator->getOutputVariable();

    // create solution, rhs, and residual vectors
    auto solVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );
    auto rhsVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );
    auto resVec = AMP::LinearAlgebra::createVector( nodalDofMap, thermalVariable );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    AMP::pout << "Constructing Linear Thermal Operator..." << std::endl;

    // now construct the linear BVP operator for thermal
    AMP_INSIST( input_db->keyExists( "testLinearThermalOperator" ), "key missing!" );
    auto linearThermalOperator = AMP::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "testLinearThermalOperator", input_db, thermalTransportModel ) );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams =
        AMP::make_shared<AMP::Operator::NeutronicsRhsParameters>( neutronicsOp_db );
    auto neutronicsOperator = AMP::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );
    auto SpecificPowerVar   = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Rhs over Desnity * GeomType::Volume
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
    auto sourceOperator = AMP::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "VolumeIntegralOperator", input_db, stransportModel ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVar = sourceOperator->getOutputVariable();
    auto PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );
    rhsVec->copyVector( PowerInWattsVec );
    auto precision = std::setprecision( 10 );
    AMP::pout << "RHS L2 norm before corrections = " << precision << rhsVec->L2Norm() << "\n";
    AMP::pout << "RHS max before corrections = " << precision << rhsVec->max() << "\n";
    AMP::pout << "RHS min before corrections = " << precision << rhsVec->min() << "\n";
    nonlinearThermalOperator->modifyRHSvector( rhsVec );
    AMP::pout << "RHS L2 norm after corrections = " << precision << rhsVec->L2Norm() << "\n";
    AMP::pout << "RHS max after corrections = " << precision << rhsVec->max() << "\n";
    AMP::pout << "RHS min after corrections = " << precision << rhsVec->min() << "\n";

    // Initial guess
    double initGuess = input_db->getDoubleWithDefault( "InitialGuess", 400.0 );
    solVec->setToScalar( initGuess );
    AMP::pout << "initial guess L2 norm before corrections = " << solVec->L2Norm() << "\n";
    AMP::pout << "initial guess max before corrections = " << solVec->max() << "\n";
    AMP::pout << "initial guess min before corrections = " << solVec->min() << "\n";

    nonlinearThermalOperator->modifyInitialSolutionVector( solVec );
    AMP::pout << "initial guess L2 norm after corrections = " << solVec->L2Norm() << "\n";
    AMP::pout << "initial guess max after corrections = " << solVec->max() << "\n";
    AMP::pout << "initial guess min after corrections = " << solVec->min() << "\n";

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db    = nonlinearSolver_db->getDatabase( "LinearSolver" );

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        AMP::make_shared<AMP::Solver::PetscSNESSolverParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nonlinearThermalOperator;
    nonlinearSolverParams->d_pInitialGuess = solVec;

    auto nonlinearSolver = AMP::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    AMP::pout << "Calling Get Jacobian Parameters..." << std::endl;

    nonlinearThermalOperator->modifyInitialSolutionVector( solVec );
    linearThermalOperator->reset( nonlinearThermalOperator->getParameters( "Jacobian", solVec ) );

    AMP::pout << "Finished reseting the jacobian." << std::endl;

    auto thermalPreconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );
    auto thermalPreconditionerParams =
        AMP::make_shared<AMP::Solver::SolverStrategyParameters>( thermalPreconditioner_db );
    thermalPreconditionerParams->d_pOperator = linearThermalOperator;
    auto linearThermalPreconditioner =
        AMP::make_shared<AMP::Solver::TrilinosMLSolver>( thermalPreconditionerParams );

    // register the preconditioner with the Jacobian free Krylov solver
    auto linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setPreconditioner( linearThermalPreconditioner );
    nonlinearThermalOperator->residual( rhsVec, solVec, resVec );

    double initialResidualNorm = resVec->L2Norm();
    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    double expectedVal = 20.7018;
    if ( !AMP::Utilities::approx_equal( expectedVal, initialResidualNorm, 1e-5 ) ) {
        ut->failure( "the Initial Residual Norm has changed." );
    }

    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->solve( rhsVec, solVec );

    std::cout << "Final Solution Norm: " << solVec->L2Norm() << std::endl;
    expectedVal = 45612;
    if ( !AMP::Utilities::approx_equal( expectedVal, solVec->L2Norm(), 1e-5 ) ) {
        ut->failure( "the Final Solution Norm has changed." );
    }

    AMP::pout << " Solution Max: " << precision << solVec->max() << std::endl;
    AMP::pout << " Solution Min: " << precision << solVec->min() << std::endl;
    AMP::pout << " Solution L1 Norm: " << precision << solVec->L1Norm() << std::endl;
    AMP::pout << " Solution L2 Norm: " << precision << solVec->L2Norm() << std::endl;

    nonlinearThermalOperator->residual( rhsVec, solVec, resVec );

    double finalResidualNorm = resVec->L2Norm();
    double finalSolutionNorm = solVec->L2Norm();
    AMP::pout << "Final Residual Norm: " << precision << finalResidualNorm << std::endl;
    AMP::pout << "Final Solution Norm: " << precision << finalSolutionNorm << std::endl;

    expectedVal = 4.561204386863e4;
    if ( fabs( finalResidualNorm ) > 1e-8 )
        ut->failure( "the Final Residual is larger than the tolerance" );
    if ( !AMP::Utilities::approx_equal( expectedVal, finalSolutionNorm, 1e-7 ) ) {
        ut->failure( "the Final Residual Norm has changed." );
    }

#ifdef USE_EXT_SILO
    auto siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( meshAdapter );
    siloWriter->registerVector( solVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Solution" );
    siloWriter->registerVector( resVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Residual" );
    siloWriter->writeFile( exeName, 0 );
#endif

    ut->passes( exeName );
}


int testPetscSNESSolver_NonlinearThermal_2( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testPetscSNESSolver-NonlinearThermal-cylinder_MATPRO2" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}