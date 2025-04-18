#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/solvers/trilinos/muelu/TrilinosMueLuSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


void linearThermalTest( AMP::UnitTest *ut,
                        const std::string &input_file,
                        const std::string &log_file )
{

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    // Create the Mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams =
        std::make_shared<AMP::Operator::NeutronicsRhsParameters>( neutronicsOp_db );
    auto neutronicsOperator = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Source over Desnity * Volume
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "VolumeIntegralOperator", input_db, stransportModel ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVar = sourceOperator->getOutputVariable();
    auto PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    // CREATE THE THERMAL BVP OPERATOR
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel;
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "DiffusionBVPOperator", input_db, transportModel ) );

    auto TemperatureInKelvinVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getInputVariable() );
    auto RightHandSideVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getOutputVariable() );
    auto ResidualVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getOutputVariable() );

    RightHandSideVec->setToScalar( 0.0 );

    // Add the boundary conditions corrections
    auto boundaryOpCorrectionVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getOutputVariable() );

    auto boundaryOp = diffusionOperator->getBoundaryOperator();
    boundaryOp->addRHScorrection( boundaryOpCorrectionVec );

    RightHandSideVec->subtract( *PowerInWattsVec, *boundaryOpCorrectionVec );

    std::cout << "RHS Norm after BC Correction " << RightHandSideVec->L2Norm() << std::endl;
    std::cout << "RHS Norm 1: " << RightHandSideVec->L2Norm() << std::endl;
    std::cout << "RHS Norm 2: " << PowerInWattsVec->L2Norm() << std::endl;
    std::cout << "RHS Norm 3: " << boundaryOpCorrectionVec->L2Norm() << std::endl;

    // make sure the database on theinput file exists for the linear solver
    AMP_INSIST( input_db->keyExists( "LinearSolver" ), "Key ''LinearSolver'' is missing!" );

    // Read the input file onto a database
    auto mlSolver_db = input_db->getDatabase( "LinearSolver" );

    // Fill in the parameters fo the class with the info on the database
    auto mlSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );

    // Define the operature to be used by the Solver.
    mlSolverParams->d_pOperator = diffusionOperator;

    // Set initial guess
    TemperatureInKelvinVec->setToScalar( 1.0 );

    // Check the initial L2 norm of the solution
    std::cout << "Initial Solution Norm: " << TemperatureInKelvinVec->L2Norm() << std::endl;
    std::cout << "RHS Norm: " << RightHandSideVec->L2Norm() << std::endl;

    // Create the ML Solver
    auto mlSolver = std::make_shared<AMP::Solver::TrilinosMueLuSolver>( mlSolverParams );

    // Use a random initial guess?
    mlSolver->setZeroInitialGuess( false );

    // Solve the prblem.
    mlSolver->apply( RightHandSideVec, TemperatureInKelvinVec );

    // Compute the residual
    diffusionOperator->residual( RightHandSideVec, TemperatureInKelvinVec, ResidualVec );

    // Check the L2 norm of the final residual.
    double finalResidualNorm = static_cast<double>( ResidualVec->L2Norm() );
    std::cout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > 10.0 ) {
        ut->failure( input_file );
    } else {
        ut->passes( input_file );
    }

    input_db.reset();
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {
        files.push_back( "input_testTrilinosMueLuSolver-LinearThermalRobin" );
    }

    for ( auto &fileName : files ) {
        linearThermalTest( &ut, fileName, fileName + ".log" );
    }
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
