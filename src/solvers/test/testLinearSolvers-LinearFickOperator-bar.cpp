#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <string>


void linearFickTest( AMP::UnitTest *ut, const std::string &inputFileName )
{
    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;
    const auto globalComm  = AMP::AMP_MPI( AMP_COMM_WORLD );

    AMP::pout << "Running with input " << input_file << std::endl;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    // create the Mesh
    const auto mesh = createMesh( input_db );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto nodalDofMap    = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    // CREATE THE DIFFUSION OPERATOR
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db ) );

    auto SolutionVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getInputVariable() );
    auto RightHandSideVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getOutputVariable() );
    auto ResidualVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, diffusionOperator->getOutputVariable() );

    RightHandSideVec->setToScalar( 0. );

    auto boundaryOp = diffusionOperator->getBoundaryOperator();

    boundaryOp->addRHScorrection( RightHandSideVec );
    boundaryOp->setRHScorrection( RightHandSideVec );

    // FIND THE SOLUTION

    // Set initial guess
    SolutionVec->setToScalar( 1.0 );

    // Check the initial L2 norm of the solution
    auto initSolNorm = static_cast<double>( SolutionVec->L2Norm() );
    AMP::pout << "Initial Solution Norm: " << initSolNorm << std::endl;

    auto rhsNorm = static_cast<double>( RightHandSideVec->L2Norm() );
    AMP::pout << "RHS Norm: " << rhsNorm << std::endl;

    auto mlSolver = AMP::Solver::Test::buildSolver(
        "LinearSolver", input_db, globalComm, nullptr, diffusionOperator );

    // Use a random initial guess?
    mlSolver->setZeroInitialGuess( false );

    // Solve the problem.
    mlSolver->apply( RightHandSideVec, SolutionVec );

    // Compute the residual
    diffusionOperator->residual( RightHandSideVec, SolutionVec, ResidualVec );

    // Check the L2 norm of the final residual.
    auto finalResidualNorm = static_cast<double>( ResidualVec->L2Norm() );
    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    if ( finalResidualNorm > 10.0 ) {
        ut->failure( mlSolver->type() + " fails to solve a linear Fick problem." );
    } else {
        ut->passes( mlSolver->type() + " successfully solves a linear Fick problem." );
    }

    // CHECK THE SOLUTION
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
    bool passes = checkAnalyticalSolution( exeName, fun, iterator, SolutionVec );
    if ( passes )
        ut->passes( "The linear Fick solve is verified" );

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
        files.emplace_back( "input_testBoomerAMGSolver-LinearFickOperator-bar" );
#endif
#ifdef AMP_USE_TRILINOS_MUELU
        files.emplace_back( "input_testTrilinosMueLuSolver-LinearFickOperator-bar" );
#endif
    }

    {
        PROFILE( "DRIVER::main(test loop)" );
        for ( auto &file : files ) {
            linearFickTest( &ut, file );
        }
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinearFick_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
