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

#ifdef AMP_USE_HYPRE
    #include "HYPRE_config.h"
#endif

#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()

std::vector<std::pair<std::string, std::string>> getBackendsAndMemory( std::string memory_space )
{
    std::vector<std::pair<std::string, std::string>> rvec;
    AMP_INSIST( memory_space == "host" || memory_space == "managed" || memory_space == "device",
                "Memory space has to be one of host, managed, or device" );

    if ( memory_space == "device" ) {
#ifdef AMP_USE_DEVICE
        rvec.emplace_back( std::make_pair( "hip_cuda", "device" ) );
    #ifdef AMP_USE_KOKKOS
        rvec.emplace_back( std::make_pair( "kokkos", "device" ) );
    #endif
#endif
    }

    if ( memory_space == "managed" ) {
#ifdef AMP_USE_DEVICE
        rvec.emplace_back( std::make_pair( "hip_cuda", "managed" ) );
    #if defined( AMP_USE_KOKKOS )
        rvec.emplace_back( std::make_pair( "kokkos", "managed" ) );
    #endif
#endif
    }

    if ( memory_space == "host" ) {
        rvec.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef AMP_USE_OPENMP
        rvec.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#ifdef AMP_USE_KOKKOS
        rvec.emplace_back( std::make_pair( "kokkos", "host" ) );
#endif
    }

    return rvec;
}

std::vector<std::string> getHypreMemorySpaces()
{
#ifdef AMP_USE_HYPRE
    std::vector<std::string> memspaces;
    #if defined( HYPRE_USING_HOST_MEMORY )
    memspaces.emplace_back( "host" );
    #elif defined( HYPRE_USING_DEVICE_MEMORY )
    memspaces.emplace_back( "host" );
    memspaces.emplace_back( "device" );
    #elif defined( HYPRE_USING_UNIFIED_MEMORY )
    memspaces.emplace_back( "host" );
    memspaces.emplace_back( "managed" );
    #else
    memspaces.emplace_back( "host" );
    #endif
    return memspaces;
#else
    return { "host" };
#endif
}

std::tuple<std::shared_ptr<AMP::Operator::LinearOperator>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>>
constructLinearSystem( std::string physicsFileName )
{
    PROFILE( "DRIVER::linearThermalPhysics" );

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( physicsFileName );

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

    return std::make_tuple( linearOp, TemperatureInKelvinVec, RightHandSideVec );
}

void linearThermalTest( AMP::UnitTest *ut,
                        const std::string &inputFileName,
                        std::tuple<std::shared_ptr<AMP::Operator::LinearOperator>,
                                   std::shared_ptr<AMP::LinearAlgebra::Vector>,
                                   std::shared_ptr<AMP::LinearAlgebra::Vector>> linearSystem,
                        std::string &accelerationBackend,
                        std::string &memoryLocation )
{
    PROFILE( "DRIVER::linearThermalTest" );

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( inputFileName );
    input_db->print( AMP::plog );

    // Print from all cores into the output files
    // auto logFile = AMP::Utilities::stringf( "output_testLinSolveRobin_r%03i",
    //                                         AMP::AMPManager::getCommWorld().getSize() );
    // AMP::logAllNodes( logFile );

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

    auto [linearOperator, sol, rhs] = linearSystem;

    auto &comm     = rhs->getComm();
    auto solver_db = input_db->getDatabase( "LinearSolver" );
    solver_db->putScalar( "MemoryLocation", memoryLocation );
    auto mem_loc = AMP::Utilities::memoryLocationFromString( memoryLocation );

    std::shared_ptr<AMP::Operator::LinearOperator> migratedOperator = linearOperator;

    if ( memoryLocation != "host" ) {

        auto inVar  = migratedOperator->getInputVariable();
        auto outVar = migratedOperator->getOutputVariable();

        // Create operator to wrap matrix
        auto op_db = std::make_shared<AMP::Database>( "LinearOperator" );
        op_db->putScalar<std::string>( "AccelerationBackend", accelerationBackend );
        op_db->putScalar<std::string>( "MemoryLocation", memoryLocation );

        auto opParams       = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
        migratedOperator    = std::make_shared<AMP::Operator::LinearOperator>( opParams );
        auto matrix         = linearOperator->getMatrix();
        auto migratedMatrix = AMP::LinearAlgebra::createMatrix( matrix, mem_loc );
        migratedOperator->setMatrix( migratedMatrix );
        migratedOperator->setVariables( inVar, outVar );
    }

    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, migratedOperator );

    auto t1 = std::chrono::high_resolution_clock::now();

    auto op_mem_loc = linearOperator->getMemoryLocation();
    std::shared_ptr<AMP::LinearAlgebra::Vector> u, f;
    if ( op_mem_loc != mem_loc ) {
        u = AMP::LinearAlgebra::createVector( sol, mem_loc );
        f = AMP::LinearAlgebra::createVector( rhs, mem_loc );
        f->copyVector( rhs );
    } else {
        u = sol;
        f = rhs;
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

void runTestOnInputs( AMP::UnitTest *ut,
                      const std::string &physicsInput,
                      std::vector<std::string> generalInputs,
                      std::vector<std::string> deviceInputs,
                      std::vector<std::string> hostOnlyInputs,
                      std::vector<std::string> managedInputs )
{

    auto linearSystem = constructLinearSystem( physicsInput );

    {
        PROFILE( "DRIVER::main(test loop for all backends on device memory)" );
        auto backendsAndMemory = getBackendsAndMemory( "device" );

        auto inputs = deviceInputs;
        inputs.insert( inputs.end(), generalInputs.begin(), generalInputs.end() );

        for ( auto &file : inputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( ut, file, linearSystem, backend, memory );
        }
    }

    {
        PROFILE( "DRIVER::main(test loop for backends on host and managed memory)" );

        auto backendsAndMemory = getBackendsAndMemory( "managed" );
        auto inputs            = managedInputs;
        inputs.insert( inputs.end(), generalInputs.begin(), generalInputs.end() );

        for ( auto &file : inputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( ut, file, linearSystem, backend, memory );
        }
    }

    {
        PROFILE( "DRIVER::main(test loop for host backends and memory)" );
        auto backendsAndMemory = getBackendsAndMemory( "host" );

        auto inputs = hostOnlyInputs;
        inputs.insert( inputs.end(), managedInputs.begin(), managedInputs.end() );
        inputs.insert( inputs.end(), generalInputs.begin(), generalInputs.end() );

        for ( auto &file : inputs ) {
            for ( auto &[backend, memory] : backendsAndMemory )
                linearThermalTest( ut, file, linearSystem, backend, memory );
        }
    }
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> generalInputs;
    std::vector<std::string> deviceInputs;
    std::vector<std::string> hostOnlyInputs;
    std::vector<std::string> managedInputs;
    std::vector<std::string> hypreInputs;

    std::string physicsInput;

    PROFILE_ENABLE();

    auto hypre_memspaces = getHypreMemorySpaces();

    if ( argc > 2 ) {
        physicsInput = argv[1];
        for ( int i = 2; i < argc; i++ )
            generalInputs.emplace_back( argv[i] );

    } else {

        physicsInput = "input_LinearThermalRobinOperator";

        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CG" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-IPCG" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FCG" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRES" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-FGMRES" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BiCGSTAB" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-TFQMR" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRESWithCGS" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRESWithCGS2" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-GMRESWithRestart" );
        //        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-QMRCGSTAB"
        //        );

        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-IPCG" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FCG" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-CG-FCG" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRES" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-FGMRES" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-GMRES" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-BiCGSTAB" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-GMRESR-TFQMR" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-BiCGSTAB" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-TFQMR" );
        // generalInputs.emplace_back(
        //     "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-QMRCGSTAB" );

        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-LeftPC-DiagonalSolver-GMRES" );

#ifdef AMP_USE_PETSC
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscCG" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscFGMRES" );
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-PetscBiCGSTAB" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscCG" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscFGMRES" );
        generalInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalSolver-PetscBiCGSTAB" );
#endif

#ifdef AMP_USE_HYPRE
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreCG" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreBiCGSTAB" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-HypreGMRES" );

        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreCG" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreGMRES" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-DiagonalPC-HypreBiCGSTAB" );

        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG" );

        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-IPCG" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FCG" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-CG-FCG" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRES" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-FGMRES" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GCR" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-GMRES" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-BiCGSTAB" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-TFQMR" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-GMRESR-QMRCGSTAB" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-BiCGSTAB" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-TFQMR" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-QMRCGSTAB" );
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreCG" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreGMRES" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-HypreBiCGSTAB" );

        if ( hypre_memspaces.size() == 1 ) {
            if ( hypre_memspaces[0] == "device" ) {
                deviceInputs.insert( deviceInputs.end(), hypreInputs.begin(), hypreInputs.end() );
            } else {
                hostOnlyInputs.insert(
                    hostOnlyInputs.end(), hypreInputs.begin(), hypreInputs.end() );
            }
        } else {
            managedInputs.insert( managedInputs.end(), hypreInputs.begin(), hypreInputs.end() );
        }

        if ( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix" ) {
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-SASolver-BoomerAMG" );
            hostOnlyInputs.emplace_back(
                "input_testLinearSolvers-LinearThermalRobin-UASolver-FCG" );
        }

    #ifdef AMP_USE_PETSC
        // hostOnlyInputs.emplace_back(
        //     "input_testLinearSolvers-LinearThermalRobin-BoomerAMG-PetscCG" );
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
        runTestOnInputs(
            &ut, physicsInput, generalInputs, deviceInputs, hostOnlyInputs, managedInputs );
    }

    {
        generalInputs.clear();
        deviceInputs.clear();
        managedInputs.clear();
        hostOnlyInputs.clear();
        generalInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-CG" );

#ifdef AMP_USE_HYPRE
        hypreInputs.clear();
        hypreInputs.emplace_back( "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG" );
        hypreInputs.emplace_back(
            "input_testLinearSolvers-LinearThermalRobin-CylMesh-BoomerAMG-CG" );

        if ( hypre_memspaces.size() == 1 ) {
            if ( hypre_memspaces[0] == "device" ) {
                deviceInputs.insert( deviceInputs.end(), hypreInputs.begin(), hypreInputs.end() );
            } else {
                hostOnlyInputs.insert(
                    hostOnlyInputs.end(), hypreInputs.begin(), hypreInputs.end() );
            }
        } else {
            managedInputs.insert( managedInputs.end(), hypreInputs.begin(), hypreInputs.end() );
        }
#endif

        runTestOnInputs( &ut,
                         "input_LinearThermalRobinOperator-CylMesh",
                         generalInputs,
                         deviceInputs,
                         hostOnlyInputs,
                         managedInputs );
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
