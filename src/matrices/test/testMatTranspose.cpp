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

size_t matTransposeTestWithDOFs( AMP::UnitTest *ut,
                                 std::string type,
                                 std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                                 const std::string &accelerationBackend,
                                 const std::string &memoryLocation )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );

    std::shared_ptr<AMP::LinearAlgebra::Vector> inVec, outVec;

    if ( memoryLocation == "host" ) {
        inVec  = AMP::LinearAlgebra::createVector( dofManager, inVar );
        outVec = AMP::LinearAlgebra::createVector( dofManager, outVar );
    } else {
        AMP_ASSERT( memoryLocation == "managed" );
        auto mem_loc = AMP::Utilities::memoryLocationFromString( memoryLocation );
        inVec        = AMP::LinearAlgebra::createVector( dofManager, inVar, true, mem_loc );
        outVec       = AMP::LinearAlgebra::createVector( dofManager, outVar, true, mem_loc );
    }

    // Create the matrix
    auto matrix = AMP::LinearAlgebra::createMatrix(
        inVec, outVec, AMP::Utilities::backendFromString( accelerationBackend ), type );
    if ( matrix ) {
        ut->passes( type + ": Able to create a square matrix" );
    } else {
        ut->failure( type + ": Unable to create a square matrix" );
    }

    fillWithPseudoLaplacian( matrix, dofManager );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // Get matrix transpose
    auto matrix_t = matrix->transpose();
    if ( matrix ) {
        ut->passes( type + ": Able to create transpose" );
    } else {
        ut->failure( type + ": Unable to create transpose" );
    }

    size_t nGlobalRows = matrix->numGlobalRows();
    size_t nLocalRows  = matrix->numLocalRows();
    AMP::pout << type << " Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

    // simple test on dimensions of transpose
    if ( nGlobalRows == matrix_t->numGlobalColumns() &&
         nLocalRows == matrix_t->numLocalColumns() ) {
        ut->passes( type + ": transpose has correct column counts" );
    } else {
        ut->failure( type + ": transpose has wrong column counts" );
    }
    if ( matrix->numGlobalColumns() == matrix_t->numGlobalRows() &&
         matrix->numLocalColumns() == matrix_t->numLocalRows() ) {
        ut->passes( type + ": transpose has correct row counts" );
    } else {
        ut->failure( type + ": transpose has wrong row counts" );
    }

#if defined( AMP_USE_HYPRE )
    using scalar_t = typename AMP::LinearAlgebra::scalar_info<AMP::LinearAlgebra::hypre_real>::type;
#else
    using scalar_t = double;
#endif

    auto x  = matrix->createInputVector();
    auto y  = matrix->createOutputVector();
    auto xt = matrix_t->createInputVector();
    auto yt = matrix_t->createOutputVector();

    x->setToScalar( 1.0 );
    xt->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    xt->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();
    yt->zero();

    matrix->mult( x, y );
    matrix_t->mult( xt, yt );

    auto yNorm  = static_cast<scalar_t>( y->L1Norm() );
    auto ytNorm = static_cast<scalar_t>( yt->L1Norm() );

    if ( yNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) && yNorm == ytNorm ) {
        ut->passes( type + ": Passes 1 norm test with pseudo Laplacian" );
    } else {
        AMP::pout << "1 Norms " << yNorm << " and " << ytNorm << ", number of rows "
                  << matrix->numGlobalRows() << std::endl;
        ut->failure( type + ": Fails 1 norm test with pseudo Laplacian" );
    }

    return nGlobalRows;
}

size_t matTransposeTest( AMP::UnitTest *ut, std::string input_file )
{
    std::string log_file = "output_testMatTranspose";
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

    std::vector<std::pair<std::string, std::string>> backendsAndMemory;
    backendsAndMemory.emplace_back( std::make_pair( "serial", "host" ) );
#ifdef USE_OPENMP
    backendsAndMemory.emplace_back( std::make_pair( "openmp", "host" ) );
#endif
#if defined( AMP_USE_KOKKOS )
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "host" ) );
    #ifdef AMP_USE_DEVICE
    backendsAndMemory.emplace_back( std::make_pair( "kokkos", "managed" ) );
    #endif
#endif
    size_t nGlobal = 0;
    for ( auto &[backend, memory] : backendsAndMemory )
        nGlobal += matTransposeTestWithDOFs( ut, "CSRMatrix", scalarDOFs, backend, memory );
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
        nGlobal = matTransposeTest( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testMatTranspose_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 ) << std::setfill( '0' )
       << nGlobal;
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
