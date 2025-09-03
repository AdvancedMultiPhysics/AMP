#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/testHelpers/MatrixDataTransforms.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// This test is adapted from testMatVec.cpp and is set up to give some basic
// profiling information regarding matvec products with different matrix
// classes

// Number of products to evaluate to average out timings
#define NUM_PRODUCTS 1000
#define NUM_PRODUCTS_TRANS 100

size_t matVecTestWithDOFs( AMP::UnitTest *ut,
                           std::string type,
                           std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                           bool testTranspose,
                           const std::string &accelerationBackend,
                           const std::string &memoryLocation )
{
    AMP::pout << "matVecTestWithDOFs with " << type << ", backend " << accelerationBackend
              << ", memory " << memoryLocation << std::endl;
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );

    std::shared_ptr<AMP::LinearAlgebra::Vector> inVec, outVec;

    // create on host and migrate as the Pseudo-Laplacian fill routines are still host based
    inVec         = AMP::LinearAlgebra::createVector( dofManager, inVar );
    outVec        = AMP::LinearAlgebra::createVector( dofManager, outVar );
    auto matrix_h = AMP::LinearAlgebra::createMatrix( inVec, outVec, type );
    fillWithPseudoLaplacian( matrix_h, dofManager );

    auto memLoc  = AMP::Utilities::memoryLocationFromString( memoryLocation );
    auto backend = AMP::Utilities::backendFromString( accelerationBackend );

    auto matrix = ( memoryLocation == "host" || type != "CSRMatrix" ) ?
                      matrix_h :
                      AMP::LinearAlgebra::createMatrix( matrix_h, memLoc, backend );

    size_t nGlobalRows = matrix->numGlobalRows();
    size_t nLocalRows  = matrix->numLocalRows();
    AMP::pout << type << " Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

#ifdef AMP_USE_HYPRE
    using scalar_t = typename AMP::LinearAlgebra::scalar_info<AMP::LinearAlgebra::hypre_real>::type;
#else
    using scalar_t = double;
#endif

    auto x = matrix->createInputVector();
    auto y = matrix->createOutputVector();

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    for ( int nProd = 0; nProd < NUM_PRODUCTS; ++nProd ) {
        matrix->mult( x, y );
    }

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );

    if ( yNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
        ut->passes( type + ", " + memoryLocation + ", " + accelerationBackend +
                    ": Passes 1 norm test with pseudo Laplacian" );
    } else {
        AMP::pout << type << ", " << memoryLocation << ", " << accelerationBackend << ", 1 Norm "
                  << yNorm << ", number of rows " << matrix->numGlobalRows() << std::endl;
        ut->failure( type + ", " + memoryLocation + ", " + accelerationBackend +
                     ": Fails 1 norm test with pseudo Laplacian" );
    }

    if ( testTranspose && NUM_PRODUCTS_TRANS ) {
        // Repeat test with transpose multiply (Laplacian is symmetric)
        y->setToScalar( 1.0 );
        y->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        x->zero();
        for ( int nProd = 0; nProd < NUM_PRODUCTS_TRANS; ++nProd ) {
            matrix->multTranspose( y, x );
        }

        auto xNorm = static_cast<scalar_t>( x->L1Norm() );

        if ( xNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
            ut->passes( type + ", " + memoryLocation + ", " + accelerationBackend +
                        ": Passes 1 norm test with pseudo Laplacian transpose" );
        } else {
            AMP::pout << type << ", " << memoryLocation << ", " << accelerationBackend
                      << ", transpose 1 Norm " << xNorm << ", number of rows "
                      << matrix->numGlobalRows() << std::endl;
            ut->failure( type + ", " + memoryLocation + ", " + accelerationBackend +
                         ": Fails 1 norm test with pseudo Laplacian transpose" );
        }
    }

    return nGlobalRows;
}

size_t matVecTest( AMP::UnitTest *ut, std::string input_file )
{
    std::string log_file = "output_testMatVecPerf";
    //    AMP::logOnlyNodeZero( log_file );
    AMP::logAllNodes( log_file );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    auto comm     = AMP::AMP_MPI( AMP_COMM_WORLD );
    params->setComm( comm );

    // Get the acceleration backend for the matrix
    std::vector<std::string> backends;
    if ( input_db->keyExists( "MatrixAccelerationBackend" ) ) {
        backends.emplace_back( input_db->getString( "MatrixAccelerationBackend" ) );
    } else {
        backends.emplace_back( "serial" );
#ifdef AMP_USE_KOKKOS
        backends.emplace_back( "kokkos" );
#endif
#ifdef AMP_USE_DEVICE
        backends.emplace_back( "hip_cuda" );
#endif
    }

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create the DOF manager
    auto scalarDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Test on defined matrix types
#if defined( AMP_USE_TRILINOS )
    matVecTestWithDOFs( ut, "ManagedEpetraMatrix", scalarDOFs, true, "serial", "host" );
#endif
#if defined( AMP_USE_PETSC )
    matVecTestWithDOFs( ut, "NativePetscMatrix", scalarDOFs, true, "serial", "host" );
#endif

    std::vector<std::pair<std::string, std::string>> backendsAndMemory;
    backendsAndMemory.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef USE_OPENMP
    backendsAndMemory.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#if defined( AMP_USE_KOKKOS )
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "host" ) );
    #ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "managed" ) );
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "device" ) );
    #endif
#endif
#ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "managed" ) );
    backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "device" ) );
#endif

    size_t nGlobal = 0;
    for ( auto &[backend, memory] : backendsAndMemory ) {
        const bool testTranspose = backend != "hip_cuda";
        nGlobal +=
            matVecTestWithDOFs( ut, "CSRMatrix", scalarDOFs, testTranspose, backend, memory );
    }
    return nGlobal;
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

        files.emplace_back( "input_testMatVecPerf-1" );
    }

    size_t nGlobal = 0;
    for ( auto &file : files )
        nGlobal = matVecTest( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testMatVecPerf_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 ) << std::setfill( '0' )
       << nGlobal;
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
