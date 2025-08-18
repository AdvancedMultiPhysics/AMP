#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

#include <chrono>
#include <iomanip>
#include <memory>
#include <string>


#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()

std::vector<std::pair<std::string, std::string>> getBackendsAndMemory( std::string memory_space )
{
    std::vector<std::pair<std::string, std::string>> rvec;
    AMP_INSIST( memory_space == "host" || memory_space == "managed" || memory_space == "device",
                "Memory space has to be one of host, managed, or device" );

    rvec.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef AMP_USE_OPENMP
    rvec.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#ifdef AMP_USE_KOKKOS
    rvec.emplace_back( std::make_pair( "kokkos", "host" ) );
#endif

    if ( memory_space == "managed" || memory_space == "device" ) {
#ifdef AMP_USE_DEVICE
        rvec.emplace_back( std::make_pair( "hip_cuda", "managed" ) );
    #if defined( AMP_USE_KOKKOS )
        rvec.emplace_back( std::make_pair( "kokkos", "managed" ) );
    #endif
#endif
    }

    if ( memory_space == "device" ) {
#ifdef AMP_USE_DEVICE
        rvec.emplace_back( std::make_pair( "hip_cuda", "device" ) );
    #ifdef AMP_USE_KOKKOS
        rvec.emplace_back( std::make_pair( "kokkos", "device" ) );
    #endif
#endif
    }

    return rvec;
}

void linearThermalTest( AMP::UnitTest *ut,
                        const std::string &inputFileName,
                        std::string &accelerationBackend,
                        std::string &memoryLocation )
{
    PROFILE( "DRIVER::linearThermalTest" );

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( inputFileName );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    auto logFile = AMP::Utilities::stringf( "output_testLinSolveRobin_r%03i",
                                            AMP::AMPManager::getCommWorld().getSize() );
    AMP::logAllNodes( logFile );

    auto nReps = input_db->getWithDefault<int>( "repetitions", 1 );
    AMP::pout << std::endl
              << "linearThermalTest input: " << inputFileName
              << ",  backend: " << accelerationBackend << ",  memory: " << memoryLocation
              << ", repetitions: " << nReps << std::endl;

    // SASolver and UASolver do not support any type of device memory yet
    if ( ( inputFileName.find( "SASolver" ) != std::string::npos ||
           inputFileName.find( "UASolver" ) != std::string::npos ) &&
         memoryLocation != "host" ) {
        ut->expected_failure( "Skipping SASolver or UASolver on non-host memory" );
        return;
    }

    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    auto volumeOp_db     = input_db->getDatabase( "VolumeIntegralOperator" );

    // Create the Mesh
    const auto mesh = createMesh( input_db );

    auto PowerInWattsVec = constructNeutronicsPowerSource( input_db, mesh );

    // Set appropriate acceleration backend
    auto op_db = input_db->getDatabase( "DiffusionBVPOperator" );
    // Create the Thermal BVP Operator
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db ) );

    auto linearOp               = diffusionOperator->getVolumeOperator();
    auto TemperatureInKelvinVec = linearOp->createOutputVector();
    auto RightHandSideVec       = linearOp->createInputVector();

    auto boundaryOpCorrectionVec = RightHandSideVec->clone();

    // Add the boundary conditions corrections
    auto boundaryOp = diffusionOperator->getBoundaryOperator();
    boundaryOp->addRHScorrection( boundaryOpCorrectionVec );
    RightHandSideVec->subtract( *PowerInWattsVec, *boundaryOpCorrectionVec );

    auto &comm     = mesh->getComm();
    auto solver_db = input_db->getDatabase( "LinearSolver" );
    solver_db->putScalar( "MemoryLocation", memoryLocation );
    auto mem_loc = AMP::Utilities::memoryLocationFromString( memoryLocation );

    std::shared_ptr<AMP::Operator::LinearOperator> migratedOperator = diffusionOperator;

    if ( memoryLocation != "host" ) {

        auto inVar  = migratedOperator->getInputVariable();
        auto outVar = migratedOperator->getOutputVariable();

        // Create operator to wrap matrix
        auto op_db = std::make_shared<AMP::Database>( "LinearOperator" );
        op_db->putScalar<std::string>( "AccelerationBackend", accelerationBackend );
        op_db->putScalar<std::string>( "MemoryLocation", memoryLocation );

        auto opParams       = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
        migratedOperator    = std::make_shared<AMP::Operator::LinearOperator>( opParams );
        auto matrix         = diffusionOperator->getMatrix();
        auto migratedMatrix = AMP::LinearAlgebra::createMatrix( matrix, mem_loc );
        migratedOperator->setMatrix( migratedMatrix );
        migratedOperator->setVariables( inVar, outVar );
    }

    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, migratedOperator );

    auto t1 = std::chrono::high_resolution_clock::now();

    auto op_mem_loc = diffusionOperator->getMemoryLocation();
    std::shared_ptr<AMP::LinearAlgebra::Vector> u, f;
    if ( op_mem_loc != mem_loc ) {
        u = AMP::LinearAlgebra::createVector( TemperatureInKelvinVec, mem_loc );
        f = AMP::LinearAlgebra::createVector( RightHandSideVec, mem_loc );
        f->copyVector( RightHandSideVec );
    } else {
        u = TemperatureInKelvinVec;
        f = RightHandSideVec;
    }


    for ( int i = 0; i < nReps; ++i ) {
        // Set initial guess
        u->setToScalar( 1.0 );

        AMP::pout << "Iteration " << i << ", system size: " << f->getGlobalSize() << std::endl;

        // Use a random initial guess?
        linearSolver->setZeroInitialGuess( false );

        // Solve the problem.
        {
            PROFILE( "DRIVER::linearThermalTest(solve call)" );
            linearSolver->apply( f, u );
        }

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

    std::vector<std::string> deviceInputs;
    std::vector<std::string> hostOnlyInputs;
    std::vector<std::string> managedAndHostInputs;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        for ( int i = 1; i < argc; i++ )
            deviceInputs.emplace_back( argv[i] );

    } else {

        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CG" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-IPCG" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FCG" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-CG" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRES" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FGMRES" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BiCGSTAB" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-TFQMR" );

        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-IPCG" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FCG" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG-FCG" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRES" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FGMRES" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-GMRES" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-BiCGSTAB" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-TFQMR" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-BiCGSTAB" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-TFQMR" );

#ifdef AMP_USE_PETSC
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscCG" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscFGMRES" );
        deviceInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscBiCGSTAB" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscCG" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscFGMRES" );
        deviceInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscBiCGSTAB" );
#endif

#ifdef AMP_USE_HYPRE
        managedAndHostInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-HypreBiCGSTAB" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-HypreGMRES" );

        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreGMRES" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreBiCGSTAB" );

        managedAndHostInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG" );

        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-IPCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG-FCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG-CG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRES" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FGMRES" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GCR" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GMRES" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-BiCGSTAB" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-TFQMR" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-BiCGSTAB" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-TFQMR" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreCG" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreGMRES" );
        managedAndHostInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreBiCGSTAB" );
        if ( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix" ) {
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-SASolver-BoomerAMG" );
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-UASolver-FCG" );
        }
    #ifdef AMP_USE_PETSC
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscCG" );
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscFGMRES" );
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscBiCGSTAB" );
    #endif
#endif
        if ( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix" ) {
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-SASolver-HybridGS" );
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-SASolver-HybridGS-FCG" );
        }

#ifdef AMP_USE_TRILINOS_ML
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-CG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-IPCG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-FCG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-GMRES" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-FGMRES" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-BiCGSTAB" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-TFQMR" );
    #ifdef AMP_USE_PETSC
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-PetscCG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-PetscFGMRES" );
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-ML-PetscBiCGSTAB" );
    #endif
#endif

#ifdef AMP_USE_TRILINOS_MUELU
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-CG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-IPCG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-FCG" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-GMRES" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-FGMRES" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-BiCGSTAB" );
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-TFQMR" );
    #ifdef AMP_USE_PETSC
        hostOnlyInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscCG" );
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscFGMRES" );
        hostOnlyInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscBiCGSTAB" );
    #endif
#endif
    }

    {
        PROFILE( "DRIVER::main(test loop for all backends on device memory)" );
        auto backendsAndMemory = getBackendsAndMemory( "device" );

        for ( auto &file : deviceInputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( &ut, file, backend, memory );
        }
    }

    {
        PROFILE( "DRIVER::main(test loop for backends on host and managed memory)" );
        auto inputs = deviceInputs;
        inputs.insert( inputs.end(), managedAndHostInputs.begin(), managedAndHostInputs.end() );

        auto backendsAndMemory = getBackendsAndMemory( "managed" );

        for ( auto &file : inputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( &ut, file, backend, memory );
        }
    }

    {
        PROFILE( "DRIVER::main(test loop for host backends and memory)" );
        auto backendsAndMemory = getBackendsAndMemory( "host" );

        auto inputs = hostOnlyInputs;
        inputs.insert( inputs.end(), managedAndHostInputs.begin(), managedAndHostInputs.end() );
        inputs.insert( inputs.end(), deviceInputs.begin(), deviceInputs.end() );

        for ( auto &file : inputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( &ut, file, backend, memory );
        }
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinSolveRobin_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
