#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"

#include "AMP/IO/AsciiWriter.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#define to_ms( x ) std::chrono::duration_cast<std::chrono::milliseconds>( x ).count()

void driver( AMP::AMP_MPI comm,
             AMP::UnitTest *ut,
             const std::string &inputFileName,
             const std::string &memoryLocation,
             const std::string &accelerationBackend )
{

    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;

    AMP::logOnlyNodeZero( log_file );
    AMP::pout << "Running driver with input " << input_file << std::endl;

    auto input_db = AMP::Database::parseInputFile( input_file );
    AMP::plog << "Input database:" << std::endl;
    AMP::plog << "---------------" << std::endl;
    input_db->print( AMP::plog );

    // Unpack databases from the input file
    auto RACoefficients_db = input_db->getDatabase( "RACoefficients" );
    auto Mesh_db           = input_db->getDatabase( "Mesh" );

    AMP_INSIST( RACoefficients_db, "A ''RACoefficients'' database must be provided" );
    AMP_INSIST( Mesh_db, "A ''Mesh'' database must be provided" );


    /****************************************************************
     * Create a mesh                                                 *
     ****************************************************************/
    // Create MeshParameters
    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( Mesh_db );
    mesh_params->setComm( comm );
    // Create Mesh
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = AMP::Mesh::BoxMesh::generate( mesh_params );

    // Print basic problem information
    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;
    AMP::plog << "Building " << static_cast<int>( mesh->getDim() )
              << "D Poisson problem on mesh with "
              << mesh->numGlobalElements( AMP::Mesh::GeomType::Vertex ) << " total DOFs across "
              << mesh->getComm().getSize() << " ranks" << std::endl;
    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;


    /*******************************************************************
     * Create manufactured rotated-anisotropic diffusion equation model *
     ********************************************************************/
    auto myRADiffusionModel =
        std::make_shared<AMP::Operator::ManufacturedRotatedAnisotropicDiffusionModel>(
            RACoefficients_db );

    // Create hassle-free wrappers around source term and exact solution
    auto PDESourceFun = std::bind( &AMP::Operator::RotatedAnisotropicDiffusionModel::sourceTerm,
                                   &( *myRADiffusionModel ),
                                   std::placeholders::_1 );
    auto uexactFun    = std::bind( &AMP::Operator::RotatedAnisotropicDiffusionModel::exactSolution,
                                &( *myRADiffusionModel ),
                                std::placeholders::_1 );


    /****************************************************************
     * Create the DiffusionFDOperator over the mesh             *
     ****************************************************************/
    const auto Op_db = std::make_shared<AMP::Database>( "linearOperatorDB" );
    Op_db->putScalar<int>( "print_info_level", 0 );
    Op_db->putScalar<std::string>( "name", "DiffusionFDOperator" );
    // Our operator requires the DiffusionCoefficients
    Op_db->putDatabase( "DiffusionCoefficients", myRADiffusionModel->d_c_db->cloneDatabase() );
    // Op_db->putDatabase( "Mesh", Mesh_db->cloneDatabase() );

    auto OpParameters    = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    OpParameters->d_name = "DiffusionFDOperator";
    OpParameters->d_Mesh = mesh;

    auto myPoissonOpHost = std::make_shared<AMP::Operator::DiffusionFDOperator>( OpParameters );

    /****************************************************************
     * Set up relevant vectors over the mesh                         *
     ****************************************************************/

    // Wrap exact solution function so that it also takes an int
    auto DirichletValue = [&]( const AMP::Mesh::Point &p, int ) { return uexactFun( p ); };
    // Create RHS vector
    auto rhsVecHost = myPoissonOpHost->createRHSVector( PDESourceFun, DirichletValue );
    rhsVecHost->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // migrate if needed
    auto mem_loc = AMP::Utilities::memoryLocationFromString( memoryLocation );
    auto backend = AMP::Utilities::backendFromString( accelerationBackend );
    std::shared_ptr<AMP::LinearAlgebra::Vector> rhsVec;
    std::shared_ptr<AMP::Operator::LinearOperator> myPoissonOp = myPoissonOpHost;
    if ( memoryLocation != "host" ) {
        auto inVar  = myPoissonOp->getInputVariable();
        auto outVar = myPoissonOp->getOutputVariable();

        // Create operator to wrap matrix
        auto op_db = std::make_shared<AMP::Database>( "LinearOperator" );
        op_db->putScalar<std::string>( "AccelerationBackend", accelerationBackend );
        op_db->putScalar<std::string>( "MemoryLocation", memoryLocation );

        auto opParams       = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
        myPoissonOp         = std::make_shared<AMP::Operator::LinearOperator>( opParams );
        auto matrix         = myPoissonOpHost->getMatrix();
        auto migratedMatrix = AMP::LinearAlgebra::createMatrix( matrix, mem_loc, backend );
        myPoissonOp->setMatrix( migratedMatrix );
        myPoissonOp->setVariables( inVar, outVar );
        rhsVec = AMP::LinearAlgebra::createVector( rhsVecHost, mem_loc, backend );
        rhsVec->copyVector( rhsVecHost );
    } else {
        myPoissonOpHost->getMatrix()->setBackend( backend );
        rhsVec = rhsVecHost;
    }
    auto unumVec = myPoissonOp->createInputVector();

    /****************************************************************
     * Construct linear solver and apply it                          *
     ****************************************************************/
    auto solver_db = input_db->getDatabase( "LinearSolver" );
    AMP_INSIST( solver_db, "A ''LinearSolver'' database must be provided" );

    // Get the linear solver for operator myPoissonOp
    auto t1_setup = std::chrono::high_resolution_clock::now();
    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, myPoissonOp );
    auto t2_setup = std::chrono::high_resolution_clock::now();

    auto nReps    = input_db->getWithDefault<int>( "repetitions", 1 );
    auto t1_solve = std::chrono::high_resolution_clock::now();
    for ( int i = 0; i < nReps; ++i ) {
        // Set initial guess
        unumVec->setToScalar( 0.0 );

        AMP::pout << "Iteration " << i << ", system size: " << rhsVec->getGlobalSize() << std::endl;

        // Use zero initial iterate and apply solver
        linearSolver->apply( rhsVec, unumVec );
    }
    auto t2_solve = std::chrono::high_resolution_clock::now();

    // Compute disretization error
    if ( myRADiffusionModel->d_exactSolutionAvailable && memoryLocation == "host" ) {
        AMP::plog << "\nDiscretization error post linear solve: ";
        // Fill exact solution vector
        auto uexactVec = myPoissonOp->createInputVector();
        myPoissonOpHost->fillVectorWithFunction( uexactVec, uexactFun );
        auto e = uexactVec->clone();
        e->axpy( -1.0, *unumVec, *uexactVec );
        auto enorms = getDiscreteNorms( myPoissonOpHost->getMeshSize(), e );
        AMP::plog << "||e|| = (" << enorms[0] << ", " << enorms[1] << ", " << enorms[2] << ")"
                  << std::endl;
    }

    // No specific solution is implemented for this problem, so this will just check that the solver
    // converged.
    checkConvergence( linearSolver.get(), input_db, input_file, *ut );


    AMP::pout << std::endl
              << "DiffusionFD test with " << inputFileName << " setup time: ("
              << 1e-3 * to_ms( t2_setup - t1_setup ) << "s), solve time: ("
              << 1e-3 * to_ms( t2_solve - t1_solve ) / nReps << " s)" << std::endl;
}
// end of driver()


/*  The input file must contain the following databases:

    Mesh : Describes parameters required to build a "cube" BoxMesh
    RACoefficients : Provides parameters required to build a RotatedAnisotropicDiffusionModel
*/
int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    PROFILE_ENABLE();

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> hostExeNames;
    [[maybe_unused]] std::vector<std::string> managedExeNames, deviceExeNames;
    if ( argc > 2 ) {
        std::string memloc( argv[1] );
        if ( memloc == "host" ) {
            hostExeNames.emplace_back( argv[2] );
        } else if ( memloc == "managed" ) {
            managedExeNames.emplace_back( argv[2] );
        } else if ( memloc == "device" ) {
            deviceExeNames.emplace_back( argv[2] );
        } else {
            AMP_ERROR( "unrecognized memloc" );
        }
    } else if ( argc == 2 ) {
        AMP_ERROR( "Too few inputs" );
    } else {
        // relaxation solvers alone, only for troubleshooting
        // hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-JacobiL1" );
        // hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-HybridGS" );
        // hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-JacobiL1" );
        // hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-HybridGS" );

        // SASolver with/without FCG acceleration
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-SASolver-HybridGS" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-SASolver-HybridGS-FCG" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-SASolver-HybridGS" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-SASolver-HybridGS-FCG" );
        // hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-DiagonalSolver-CG" );
#ifdef AMP_USE_DEVICE
        // managedExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-DiagonalSolver-CG"
        // ); deviceExeNames.emplace_back(
        // "input_testLinearSolvers-DiffusionFD-3D-DiagonalSolver-CG" );
#endif
#ifdef AMP_USE_HYPRE
        // Boomer with/without CG acceleration
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-BoomerAMG" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-2D-BoomerAMG-CG" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-BoomerAMG" );
        hostExeNames.emplace_back( "input_testLinearSolvers-DiffusionFD-3D-BoomerAMG-CG" );
    #ifdef AMP_USE_DEVICE
            // managedExeNames.emplace_back(
            // "input_testLinearSolvers-DiffusionFD-3D-DiagonalPC-HypreCG" );
            // deviceExeNames.emplace_back(
            // "input_testLinearSolvers-DiffusionFD-3D-DiagonalPC-HypreCG" );
    #endif
#endif
    }

    for ( auto &exeName : hostExeNames ) {
        driver( comm, &ut, exeName, "host", "serial" );
    }
#ifdef AMP_USE_DEVICE
    for ( auto &exeName : managedExeNames ) {
        driver( comm, &ut, exeName, "managed", "kokkos" );
    }
    for ( auto &exeName : deviceExeNames ) {
        driver( comm, &ut, exeName, "device", "kokkos" );
    }
#endif

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinearSolvers-DiffusionFD_r" << std::setw( 3 ) << std::setfill( '0' )
       << comm.getSize();
    PROFILE_SAVE( ss.str() );

    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
