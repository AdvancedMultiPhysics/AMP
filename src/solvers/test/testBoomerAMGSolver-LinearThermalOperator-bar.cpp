#include "AMP/IO/PIO.h"
#include "AMP/IO/Writer.h"
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
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/solvers/hypre/BoomerAMGSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "testSolverHelpers.h"

#include <limits>
#include <memory>
#include <string>

#define ITFAILS ut.failure( __LINE__ );
#define UNIT_TEST( a ) \
    if ( !( a ) )      \
        ut.failure( __LINE__ );

void linearThermalTest( AMP::UnitTest *ut )
{
    // Input and output file names
    //  #include <string>
    std::string exeName( "testBoomerAMGSolver-LinearThermalOperator-bar" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    //   Create the Mesh.
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

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> unusedModel;
    auto neutronicsOperator = std::dynamic_pointer_cast<AMP::Operator::NeutronicsRhs>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "NeutronicsOperator", input_db, unusedModel ) );

    neutronicsOperator->setTimeStep( 0. );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

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

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    // CREATE THE THERMAL OPERATOR
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

    RightHandSideVec->copyVector( PowerInWattsVec );

    auto boundaryOp = diffusionOperator->getBoundaryOperator();

    boundaryOp->addRHScorrection( RightHandSideVec );
    boundaryOp->setRHScorrection( RightHandSideVec );

    // make sure the database on theinput file exists for the linear solver
    AMP_INSIST( input_db->keyExists( "LinearSolver" ), "Key ''LinearSolver'' is missing!" );

    // Read the input file onto a database.
    auto mlSolver_db = input_db->getDatabase( "LinearSolver" );

    // Fill in the parameters fo the class with the info on the database.
    auto mlSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );

    // Define the operature to be used by the Solver.
    mlSolverParams->d_pOperator = diffusionOperator;

    // Set initial guess
    TemperatureInKelvinVec->setToScalar( 1.0 );

    // Check the initial L2 norm of the solution
    double initSolNorm = static_cast<double>( TemperatureInKelvinVec->L2Norm() );
    std::cout << "Initial Solution Norm: " << initSolNorm << std::endl;

    double rhsNorm = static_cast<double>( RightHandSideVec->L2Norm() );
    std::cout << "RHS Norm: " << rhsNorm << std::endl;

    // Create the ML Solver
    auto mlSolver = std::make_shared<AMP::Solver::BoomerAMGSolver>( mlSolverParams );

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
        ut->failure( "BoomerAMGSolver successfully solves a linear thermal problem with a nuclear "
                     "source term." );
    } else {
        ut->passes( "BoomerAMGSolver successfully solves a linear thermal problem with a nuclear "
                    "source term." );
    }

    // check the solution
    int zeroGhostWidth = 0;
    auto iterator      = meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, zeroGhostWidth );

    // The analytical solution is:  T = a + b*z + c*z*z
    //   c = -power/2
    //   b = -10*power
    //   a = 300 + 150*power
    auto fun = []( double, double, double z ) {
        double power = 1.;
        double c     = -power / 2.;
        double b     = -10. * power;
        double a     = 300. + 150. * power;
        return a + b * z + c * z * z;
    };
    bool passes = checkAnalyticalSolution( exeName, fun, iterator, TemperatureInKelvinVec );
    if ( passes )
        ut->passes( "The linear thermal solve is verified" );

    // Plot the results
    auto siloWriter = AMP::IO::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( meshAdapter );
    siloWriter->registerVector(
        PowerInWattsVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "PowerInWatts" );
    siloWriter->registerVector(
        TemperatureInKelvinVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "TemperatureInKelvin" );
    siloWriter->registerVector( ResidualVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Residual" );
    siloWriter->writeFile( input_file, 0 );

    input_db.reset();
    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    linearThermalTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
