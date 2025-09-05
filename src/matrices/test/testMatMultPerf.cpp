#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
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

// This test is adapted from testMatVecPerf.cpp and is set up to give some basic
// profiling information regarding SpGEMMs with different matrix classes

// Number of products to evaluate to average out timings
#define NUM_PRODUCTS_NOREUSE 10
#define NUM_PRODUCTS_REUSE 0

size_t matMatTestWithDOFs( AMP::UnitTest *ut,
                           std::string type,
                           std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                           const std::string &accelerationBackend,
                           const std::string &memoryLocation )
{
    AMP::pout << "matMatTestWithDOFs with " << type << ", backend " << accelerationBackend
              << ", memory " << memoryLocation << std::endl;

    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );

    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );

    std::shared_ptr<AMP::LinearAlgebra::Vector> inVec, outVec;

    // create on host and migrate as the Pseudo-Laplacian fill routines are still host based
    inVec         = AMP::LinearAlgebra::createVector( dofManager, inVar );
    outVec        = AMP::LinearAlgebra::createVector( dofManager, outVar );
    auto matrix_h = AMP::LinearAlgebra::createMatrix( inVec, outVec, type );
    fillWithPseudoLaplacian( matrix_h, dofManager );

    // migrate matrix if requested and possible
    auto memLoc  = AMP::Utilities::memoryLocationFromString( memoryLocation );
    auto backend = AMP::Utilities::backendFromString( accelerationBackend );

    auto A = ( memoryLocation == "host" || type != "CSRMatrix" ) ?
                 matrix_h :
                 AMP::LinearAlgebra::createMatrix( matrix_h, memLoc, backend );

    size_t nGlobalRows = A->numGlobalRows();
    size_t nLocalRows  = A->numLocalRows();
    AMP::pout << type << " Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

#ifdef AMP_USE_HYPRE
    using scalar_t = typename AMP::LinearAlgebra::scalar_info<AMP::LinearAlgebra::hypre_real>::type;
#else
    using scalar_t = double;
#endif

    // First do products without allowing reuse of result matrix
    // skip one and do it outside the loop to produce a result
    // matrix for the next phase with reuse
    const auto yNormExpect = static_cast<scalar_t>( A->numGlobalRows() );
    scalar_t yNormFail     = 0.0;
    bool allPass           = true;
    for ( int nProd = 1; nProd < NUM_PRODUCTS_NOREUSE; ++nProd ) {
        PROFILE( "SpGEMM test no reuse" );
        auto Asq = AMP::LinearAlgebra::Matrix::matMatMult( A, A );
        auto x   = Asq->createInputVector();
        auto y   = Asq->createOutputVector();
        x->setToScalar( 1.0 );
        x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        y->zero();
        Asq->mult( x, y );
        const auto yNorm = static_cast<scalar_t>( y->L1Norm() );
        if ( yNorm != yNormExpect ) {
            allPass   = false;
            yNormFail = yNorm;
        }
    }
    auto Asq = AMP::LinearAlgebra::Matrix::matMatMult( A, A );
    auto x   = Asq->createInputVector();
    auto y   = Asq->createOutputVector();
    x->setToScalar( 1.0 );
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();
    Asq->mult( x, y );
    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm != yNormExpect ) {
        allPass   = false;
        yNormFail = yNorm;
    }

    if ( allPass ) {
        ut->passes( type + ", " + memoryLocation + ", " + accelerationBackend +
                    ": Passes 1 norm test with squared pseudo Laplacian, no re-use" );
    } else {
        AMP::pout << type << ", " << memoryLocation << ", " << accelerationBackend << ", 1 Norm "
                  << yNormFail << ", number of rows " << A->numGlobalRows() << std::endl;
        ut->failure( type + ", " + memoryLocation + ", " + accelerationBackend +
                     ": Fails 1 norm test with squared pseudo Laplacian, no re-use" );
    }

    // now do products where reuse of result matrix is supported
    yNormFail = 0.0;
    allPass   = true;
    for ( int nProd = 0; nProd < NUM_PRODUCTS_REUSE; ++nProd ) {
        PROFILE( "SpGEMM test reuse" );
        AMP::LinearAlgebra::Matrix::matMatMult( A, A, Asq );
        x->setToScalar( 1.0 );
        x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        y->zero();
        Asq->mult( x, y );
        const auto yNorm = static_cast<scalar_t>( y->L1Norm() );
        if ( yNorm != yNormExpect ) {
            allPass   = false;
            yNormFail = yNorm;
        }
    }

    if ( allPass ) {
        ut->passes( type + ", " + memoryLocation + ", " + accelerationBackend +
                    ": Passes 1 norm test with squared pseudo Laplacian, with re-use" );
    } else {
        AMP::pout << type << ", " << memoryLocation << ", " << accelerationBackend << ", 1 Norm "
                  << yNormFail << ", number of rows " << A->numGlobalRows() << std::endl;
        ut->failure( type + ", " + memoryLocation + ", " + accelerationBackend +
                     ": Fails 1 norm test with squared pseudo Laplacian, with re-use" );
    }

    return nGlobalRows;
}

size_t matMatTest( AMP::UnitTest *ut, std::string input_file )
{
    std::string log_file = "output_testMatVecPerf";
    AMP::logOnlyNodeZero( log_file );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    auto comm     = AMP::AMP_MPI( AMP_COMM_WORLD );
    params->setComm( comm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create the DOF manager
    auto scalarDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Test on defined matrix types
#if defined( AMP_USE_TRILINOS )
    matMatTestWithDOFs( ut, "ManagedEpetraMatrix", scalarDOFs, "serial", "host" );
#endif
#if defined( AMP_USE_PETSC )
    matMatTestWithDOFs( ut, "NativePetscMatrix", scalarDOFs, "serial", "host" );
#endif

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

    std::vector<std::pair<std::string, std::string>> backendsAndMemory;
    backendsAndMemory.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "hip_cuda", "device" ) );
#endif

    size_t nGlobal = 0;
    for ( auto &[backend, memory] : backendsAndMemory )
        nGlobal += matMatTestWithDOFs( ut, "CSRMatrix", scalarDOFs, backend, memory );
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
        nGlobal = matMatTest( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testMatMultPerf_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 ) << std::setfill( '0' )
       << nGlobal;
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
