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

    // SASolver does not support any type of device memory yet
    if ( inputFileName.find( "SASolver" ) != std::string::npos && memoryLocation != "host" ) {
        ut->expected_failure( "Skipping SASolver on non-host memory" );
        return;
    }

    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    neutronicsOp_db->putScalar( "AccelerationBackend", accelerationBackend );
    neutronicsOp_db->putScalar( "MemoryLocation", memoryLocation );
    auto volumeOp_db = input_db->getDatabase( "VolumeIntegralOperator" );
    volumeOp_db->putScalar( "AccelerationBackend", accelerationBackend );
    volumeOp_db->putScalar( "MemoryLocation", memoryLocation );

    // Create the Mesh
    const auto mesh = createMesh( input_db );

    auto PowerInWattsVec = constructNeutronicsPowerSource( input_db, mesh );

    // Set appropriate acceleration backend
    auto op_db = input_db->getDatabase( "DiffusionBVPOperator" );
    op_db->putScalar( "AccelerationBackend", accelerationBackend );
    op_db->putScalar( "MemoryLocation", memoryLocation );
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

    auto &comm = mesh->getComm();

    auto linearSolver = AMP::Solver::Test::buildSolver(
        "LinearSolver", input_db, comm, nullptr, diffusionOperator );

    auto t1 = std::chrono::high_resolution_clock::now();

    for ( int i = 0; i < nReps; ++i ) {
        // Set initial guess
        TemperatureInKelvinVec->setToScalar( 1.0 );

        AMP::pout << "Iteration " << i << ", system size: " << RightHandSideVec->getGlobalSize()
                  << std::endl;

        // Use a random initial guess?
        linearSolver->setZeroInitialGuess( false );

        // Solve the problem.
        {
            PROFILE( "DRIVER::linearThermalTest(solve call)" );
            linearSolver->apply( RightHandSideVec, TemperatureInKelvinVec );
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

    std::vector<std::string> files;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        for ( int i = 1; i < argc; i++ )
            files.emplace_back( argv[i] );

    } else {

        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-IPCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-TFQMR" );

        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-IPCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FGMRES" );
        files.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-GMRES" );
        files.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-BiCGSTAB" );
        files.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-TFQMR" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-TFQMR" );
        if ( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix" ) {
            files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-SASolver-HybridGS" );
            files.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-SASolver-HybridGS-FCG" );
        }
#ifdef AMP_USE_PETSC
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscCG" );
        files.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscFGMRES" );
        //        files.emplace_back(
        //        "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscBiCGSTAB" );
#endif

#ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreBiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreGMRES" );

        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreBiCGSTAB" );

        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG" );

        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-IPCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GCR" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GMRES" );
        files.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-TFQMR" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-TFQMR" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreBiCGSTAB" );
        if ( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix" ) {
            files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-SASolver-BoomerAMG" );
        }
    #ifdef AMP_USE_PETSC
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscFGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscBiCGSTAB" );
    #endif
#endif

#ifdef AMP_USE_PETSC
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscFGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscBiCGSTAB" );
#endif

#ifdef AMP_USE_TRILINOS_ML
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-IPCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-GMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-FGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-TFQMR" );
    #ifdef AMP_USE_PETSC
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-PetscCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-PetscFGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-ML-PetscBiCGSTAB" );
    #endif
#endif

#ifdef AMP_USE_TRILINOS_MUELU
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-CG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-IPCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-FCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-GMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-FGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-BiCGSTAB" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-TFQMR" );
    #ifdef AMP_USE_PETSC
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscCG" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscFGMRES" );
        files.emplace_back( "input_testLinearSolvers-LinearThermalRobin-MueLu-PetscBiCGSTAB" );
    #endif
#endif
    }

    std::vector<std::pair<std::string, std::string>> backendsAndMemory;
    backendsAndMemory.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef AMP_USE_OPENMP
    backendsAndMemory.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "managed" ) );
#endif
#ifdef AMP_USE_KOKKOS
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "host" ) );
    #ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "managed" ) );
    #endif
#endif

    {
        PROFILE( "DRIVER::main(test loop)" );
        for ( auto &file : files ) {
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
