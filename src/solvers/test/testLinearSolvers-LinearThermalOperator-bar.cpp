#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>
#include <limits>
#include <string>

void linearThermalTest( AMP::UnitTest *ut, const std::string &inputFileName )
{
    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;

    AMP::pout << "Running with input " << input_file << std::endl;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    // Create the Mesh
    const auto mesh = createMesh( input_db );

    // Create the power source
    auto PowerInWattsVec = constructNeutronicsPowerSource( input_db, mesh );

    // CREATE THE THERMAL OPERATOR
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db ) );

    auto nodalDofMap    = PowerInWattsVec->getDOFManager();
    auto inputVariable  = diffusionOperator->getInputVariable();
    auto outputVariable = diffusionOperator->getOutputVariable();
    auto memoryLocation = diffusionOperator->getMemoryLocation();

    auto TemperatureInKelvinVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, inputVariable, true, memoryLocation );

    auto RightHandSideVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, outputVariable, true, memoryLocation );

    RightHandSideVec->copyVector( PowerInWattsVec );
    diffusionOperator->modifyRHSvector( RightHandSideVec );

    // Set initial guess
    TemperatureInKelvinVec->setToScalar( 1.0 );

    // Check the initial L2 norm of the solution
    const auto initSolNorm = static_cast<double>( TemperatureInKelvinVec->L2Norm() );
    AMP::pout << "Initial Solution Norm: " << initSolNorm << std::endl;

    const auto rhsNorm = static_cast<double>( RightHandSideVec->L2Norm() );
    AMP::pout << "RHS Norm: " << rhsNorm << std::endl;

    auto &comm    = mesh->getComm();
    auto mlSolver = AMP::Solver::Test::buildSolver(
        "LinearSolver", input_db, comm, nullptr, diffusionOperator );

    // Solve the problem.
    mlSolver->setZeroInitialGuess( false );
    mlSolver->apply( RightHandSideVec, TemperatureInKelvinVec );

    checkConvergence( mlSolver.get(), input_db, inputFileName, *ut );

    // check the solution
    auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );

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

    std::string exeName = inputFileName;
    auto pos            = exeName.find( "input_" );
    exeName.erase( pos, 6 );

    bool passes = checkAnalyticalSolution( exeName, fun, iterator, TemperatureInKelvinVec );
    if ( passes )
        ut->passes( "The linear thermal solve is verified" );

    input_db.reset();
    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {
#ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testBoomerAMGSolver-LinearThermalOperator-bar" );
#endif
#ifdef AMP_USE_TRILINOS_MUELU
        files.emplace_back( "input_testTrilinosMueLuSolver-LinearThermalOperator-bar" );
#endif
    }

    {
        PROFILE( "DRIVER::main(test loop)" );
        for ( auto &file : files ) {
            linearThermalTest( &ut, file );
        }
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinearThermalOperator-bar_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
