#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRPolicy.h"
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

#if defined( AMP_USE_HYPRE )
    #include "AMP/matrices/data/hypre/HypreCSRPolicy.h"
#endif

#include "ProfilerApp.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// This test is adapted from testMatVec.cpp and is set up to give some basic
// profiling information regarding matvec products with different matrix
// classes


template<typename Policy, class Allocator>
void createMatrixAndVectors(
    AMP::UnitTest *ut,
    std::string test_name,
    AMP::Utilities::Backend backend,
    std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> &matrix,
    std::shared_ptr<AMP::LinearAlgebra::Vector> &x,
    std::shared_ptr<AMP::LinearAlgebra::Vector> &y )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    // clang-format off
#ifdef USE_DEVICE
    // using  AMP::ManagedAllocator<void>;
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
        leftDOF, rightDOF, comm, inVar, outVar, getRow );

    params->d_backend = backend;

    // Create the matrix
    auto data = std::make_shared<AMP::LinearAlgebra::CSRMatrixData<Policy, Allocator>>( params );
    matrix    = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>>( data );
    // Initialize the matrix
    matrix->zero();
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    ///// END: Temporary before updating create matrix

    if ( matrix ) {
        ut->passes( test_name + ": Able to create a square matrix" );
    } else {
        ut->failure( test_name + ": Unable to create a square matrix" );
    }

    x = matrix->getRightVector();
    y = matrix->getLeftVector();
}

template<class Allocator>
void testCopyCast( AMP::UnitTest *ut,
                   std::string test_name,
                   AMP::Utilities::Backend backend,
                   std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using PolicyD = AMP::LinearAlgebra::CSRPolicy<size_t, int, double>;
    using PolicyF = AMP::LinearAlgebra::CSRPolicy<size_t, int, float>;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<PolicyD, Allocator>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<PolicyF, Allocator>> B = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<PolicyD, Allocator>> C = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<PolicyF, Allocator>> D = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                        = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                        = nullptr;
    createMatrixAndVectors<PolicyD, Allocator>( ut, test_name, backend, dofManager, A, x, y );
    createMatrixAndVectors<PolicyF, Allocator>( ut, test_name, backend, dofManager, B, x, y );
    createMatrixAndVectors<PolicyD, Allocator>( ut, test_name, backend, dofManager, C, x, y );
    createMatrixAndVectors<PolicyF, Allocator>( ut, test_name, backend, dofManager, D, x, y );

    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    B->copyCast( A );
    C->copyCast( B );

    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols_A, cols_C;
        std::vector<double> vals_A, vals_C;
        A->getRowByGlobalID( i, cols_A, vals_A );
        C->getRowByGlobalID( i, cols_C, vals_C );
        for ( size_t j = 0; j != cols_A.size(); j++ ) {
            if ( std::abs( vals_A[j] - vals_C[j] ) > std::numeric_limits<float>::epsilon() ) {
                ut->failure( test_name + ": Fails to copyCast double->float->double" );
                return;
            }
        }
    }
    ut->passes( test_name + ": Able to copyCast double->float->double" );

    C->copyCast( A );
    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols_A, cols_C;
        std::vector<double> vals_A, vals_C;
        A->getRowByGlobalID( i, cols_A, vals_A );
        C->getRowByGlobalID( i, cols_C, vals_C );
        for ( size_t j = 0; j != cols_A.size(); j++ ) {
            if ( std::abs( vals_A[j] - vals_C[j] ) > std::numeric_limits<double>::epsilon() ) {
                ut->failure( test_name + ": Fails to copy doubles" );
                return;
            }
        }
    }
    ut->passes( test_name + ": Able to copy doubles" );

    D->copyCast( B );
    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols_B, cols_D;
        // getRowByGlobalID always returns doubles
        std::vector<double> vals_B, vals_D;
        B->getRowByGlobalID( i, cols_B, vals_B );
        D->getRowByGlobalID( i, cols_D, vals_D );
        for ( size_t j = 0; j != cols_B.size(); j++ ) {
            if ( std::abs( vals_B[j] - vals_D[j] ) > std::numeric_limits<float>::epsilon() ) {
                ut->failure( test_name + ": Fails to copy floats" );
                return;
            }
        }
    }
    ut->passes( test_name + ": Able to copy floats" );
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
    testCopyCast<AMP::HostAllocator<void>>( ut, "Serial Host", AMP::Utilities::Backend::serial, scalarDOFs );
#ifdef USE_DEVICE
    testCopyCast<AMP::ManagedAllocator<void>>( ut, "Hip_Cuda Managed", AMP::Utilities::Backend::hip_cuda, scalarDOFs );
#endif
#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )
    testCopyCast<AMP::HostAllocator<void>>( ut, "Kokkos Host", AMP::Utilities::Backend::kokkos, scalarDOFs );
    #ifdef USE_DEVICE
    testCopyCast<AMP::ManagedAllocator<void>>( ut, "Kokkos Managed", AMP::Utilities::Backend::kokkos, scalarDOFs );
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
