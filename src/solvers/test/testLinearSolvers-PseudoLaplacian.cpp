#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <chrono>
#include <iomanip>
#include <memory>
#include <string>

#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()

void linearThermalTest( AMP::UnitTest *ut,
                        const std::string &inputFileName,
                        std::string &accelerationBackend,
                        std::string &memoryLocation )
{
    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;

    AMP::pout << "Running linearThermalTest with input " << input_file << " with "
              << accelerationBackend << " backend on " << memoryLocation << " memory" << std::endl;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    AMP::logAllNodes( log_file );

    auto nReps = input_db->getWithDefault<int>( "repetitions", 1 );

    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );

    // Create the Mesh and DOFManager
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( comm );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );
    auto scalarDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Create variables and vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = inVar;

    std::shared_ptr<AMP::LinearAlgebra::Vector> inVec, outVec;

    if ( memoryLocation == "host" ) {
        inVec  = AMP::LinearAlgebra::createVector( scalarDOFs, inVar );
        outVec = AMP::LinearAlgebra::createVector( scalarDOFs, outVar );
    } else {
        auto mem_loc = AMP::Utilities::memoryLocationFromString( memoryLocation );
        inVec        = AMP::LinearAlgebra::createVector( scalarDOFs, inVar, true, mem_loc );
        outVec       = AMP::LinearAlgebra::createVector( scalarDOFs, outVar, true, mem_loc );
    }

    auto backend = AMP::Utilities::backendFromString( accelerationBackend );

    // Create the matrix
    auto matrix = AMP::LinearAlgebra::createMatrix( inVec, outVec, backend, "CSRMatrix" );

    fillWithPseudoLaplacian( matrix, scalarDOFs );

    // Create operator to wrap matrix
    auto op_db = input_db->getDatabase( "LinearOperator" );
    op_db->putScalar<std::string>( "AccelerationBackend", accelerationBackend );
    op_db->putScalar<std::string>( "MemoryLocation", memoryLocation );

    auto opParams       = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
    auto linearOperator = std::make_shared<AMP::Operator::LinearOperator>( opParams );
    linearOperator->setMatrix( matrix );
    linearOperator->setVariables( inVar, outVar );

    auto solver_db = input_db->getDatabase( "LinearSolver" );
    solver_db->putScalar( "MemoryLocation", memoryLocation );
    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, linearOperator );

    // Set initial guess and rhs
    auto sol = matrix->createInputVector();
    auto rhs = matrix->createOutputVector();

    auto t1 = std::chrono::high_resolution_clock::now();

    for ( int i = 0; i < nReps; ++i ) {

        AMP::pout << "Iteration " << i << ", system size: " << rhs->getGlobalSize() << std::endl;
        sol->setToScalar( 0.0 );
        rhs->setRandomValues();
        // Check the initial L2 norm of the solution
        double initSolNorm = static_cast<double>( sol->L2Norm() );
        AMP::pout << "Initial Solution Norm: " << initSolNorm << std::endl;
        AMP::pout << "RHS Norm: " << rhs->L2Norm() << std::endl;
        AMP::pout << "System size: " << rhs->getGlobalSize() << std::endl;


        // Use a random initial guess?
        linearSolver->setZeroInitialGuess( true );

        // Solve the problem.
        linearSolver->apply( rhs, sol );

        checkConvergence( linearSolver.get(), input_db, inputFileName, *ut );
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    AMP::pout << std::endl
              << "linearThermalTest with " << inputFileName << "  average time: ("
              << 1e-3 * to_ms( t2 - t1 ) / nReps << " s)" << std::endl;
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
        files.emplace_back( "input_testLinearSolvers-PL-AmpMesh-CG" );

#ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testLinearSolvers-PL-AmpMesh-BoomerAMG" );
#endif

#ifdef AMP_USE_LIBMESH
        files.emplace_back( "input_testLinearSolvers-PL-CylMesh-CG" );
    #ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testLinearSolvers-PL-CylMesh-BoomerAMG" );
    #endif
#endif
    }

    std::vector<std::pair<std::string, std::string>> backendsAndMemory;
    if ( argc == 4 ) {
        backendsAndMemory.emplace_back( std::make_pair( argv[2], argv[3] ) );
    } else {
        backendsAndMemory.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef AMP_USE_OPENMP
        backendsAndMemory.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#ifdef AMP_USE_KOKKOS
        backendsAndMemory.emplace_back( std::make_pair( "kokkos", "host" ) );
    #ifdef AMP_USE_DEVICE
        backendsAndMemory.emplace_back( std::make_pair( "kokkos", "managed" ) );
    //    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "device" ) );
    #endif
#endif
#ifdef AMP_USE_DEVICE
        backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "managed" ) );
        //    backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "device" ) );
#endif
    }
    for ( auto &file : files ) {
        for ( auto &[backend, memory] : backendsAndMemory )
            linearThermalTest( &ut, file, backend, memory );
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinSolvePsuedoLap_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
