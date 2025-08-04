#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
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

#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )
    #include "Kokkos_Core.hpp"
#endif

#include "ProfilerApp.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// This test is adapted from testMatVec.cpp and is set up to give some basic
// profiling information regarding matvec products with different matrix
// classes


template<typename Config>
void createMatrixAndVectors( AMP::UnitTest *ut,
                             const std::string &test_name,
                             AMP::Utilities::Backend backend,
                             std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                             std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> &matrix,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &x,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &y )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    // clang-format off
#ifdef USE_DEVICE
    // using AMP::ManagedAllocator<void>;
    auto inVec = AMP::LinearAlgebra::createVector(
        dofManager, inVar, true, AMP::Utilities::MemoryType::managed );
    auto outVec = AMP::LinearAlgebra::createVector(
        dofManager, outVar, true, AMP::Utilities::MemoryType::managed );
#else
    // using AMP::HostAllocator<void>;
    auto inVec  = AMP::LinearAlgebra::createVector( dofManager, inVar );
    auto outVec = AMP::LinearAlgebra::createVector( dofManager, outVar );
#endif
    // clang-format on

    ///// Temporary before updating create matrix
    // Get the DOFs
    auto leftDOF  = inVec->getDOFManager();
    auto rightDOF = outVec->getDOFManager();

    const auto _leftDOF  = leftDOF.get();
    const auto _rightDOF = rightDOF.get();
    std::function<std::vector<size_t>( size_t )> getRow;
    getRow = [_leftDOF, _rightDOF]( size_t row ) {
        auto id = _leftDOF->getElementID( row );
        return _rightDOF->getRowDOFs( id );
    };

    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
        leftDOF, rightDOF, comm, inVar, outVar, backend, getRow );

    // Create the matrix
    auto data = std::make_shared<AMP::LinearAlgebra::CSRMatrixData<Config>>( params );
    matrix    = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Config>>( data );
    // Initialize the matrix
    matrix->zero();
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    ///// END: Temporary before updating create matrix

    if ( matrix ) {
        ut->passes( test_name + ": Able to create a square matrix" );
    } else {
        ut->failure( test_name + ": Unable to create a square matrix" );
    }

    x = matrix->createInputVector();
    y = matrix->createOutputVector();
}

template<class Config>
void checkEqualEntries( AMP::UnitTest *ut,
                        const std::string &test_name,
                        const std::string &task,
                        std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                        std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> &X,
                        std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> &Y,
                        double tol )
{
    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols_X, cols_Y;
        std::vector<double> vals_X, vals_Y;
        X->getRowByGlobalID( i, cols_X, vals_X );
        Y->getRowByGlobalID( i, cols_Y, vals_Y );
        for ( size_t j = 0; j != cols_X.size(); j++ ) {
            if ( std::abs( vals_X[j] - vals_Y[j] ) > tol ) {
                ut->failure( test_name + ": Fails to " + task +
                             AMP::Utilities::stringf( ". Difference of %e found between entries",
                                                      std::abs( vals_X[j] - vals_Y[j] ) ) );
                return;
            }
        }
    }
    ut->passes( test_name + ": Able to " + task );
}

template<AMP::LinearAlgebra::alloc Allocator>
void testCopyCast( AMP::UnitTest *ut,
                   const std::string &test_name,
                   AMP::Utilities::Backend backend,
                   std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using AMP::LinearAlgebra::index;
    using AMP::LinearAlgebra::scalar;
    using ConfigD = AMP::LinearAlgebra::CSRConfig<Allocator, index::i32, index::i64, scalar::f64>;
    using ConfigF = AMP::LinearAlgebra::CSRConfig<Allocator, index::i32, index::i64, scalar::f32>;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<ConfigD>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<ConfigF>> B = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<ConfigD>> C = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<ConfigF>> D = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x             = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y             = nullptr;
    createMatrixAndVectors<ConfigD>( ut, test_name, backend, dofManager, A, x, y );
    createMatrixAndVectors<ConfigF>( ut, test_name, backend, dofManager, B, x, y );
    createMatrixAndVectors<ConfigD>( ut, test_name, backend, dofManager, C, x, y );
    createMatrixAndVectors<ConfigF>( ut, test_name, backend, dofManager, D, x, y );

    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    B->copyCast( A );
    C->copyCast( B );
    checkEqualEntries<ConfigD>( ut,
                                test_name,
                                "copyCast double->float->double",
                                dofManager,
                                A,
                                C,
                                std::numeric_limits<float>::epsilon() );

    C->copyCast( A );
    checkEqualEntries<ConfigD>(
        ut, test_name, "copy doubles", dofManager, A, C, std::numeric_limits<double>::epsilon() );

    D->copyCast( B );
    checkEqualEntries<ConfigF>(
        ut, test_name, "copy floats", dofManager, B, D, std::numeric_limits<float>::epsilon() );
}

void matDeviceOperationsTest( AMP::UnitTest *ut, std::string input_file )
{
    std::string log_file = "output_testMatCopyCast";
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

    // clang-format off
    using AMP::LinearAlgebra::alloc;
    testCopyCast<alloc::host>( ut, "Serial Host", AMP::Utilities::Backend::Serial, scalarDOFs );
#ifdef USE_DEVICE
    testCopyCast<alloc::managed>( ut, "Hip_Cuda Managed", AMP::Utilities::Backend::Hip_Cuda, scalarDOFs );
#endif
#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )
    testCopyCast<alloc::host>( ut, "Kokkos Host", AMP::Utilities::Backend::Kokkos, scalarDOFs );
    #ifdef USE_DEVICE
    testCopyCast<alloc::managed>( ut, "Kokkos Managed", AMP::Utilities::Backend::Kokkos, scalarDOFs );
    #endif
#endif
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
    for ( auto &file : files )
        matDeviceOperationsTest( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testMatVecPerf_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 )
       << std::setfill( '0' );
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
