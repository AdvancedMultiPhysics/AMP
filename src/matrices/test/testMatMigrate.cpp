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

// This test is adapted from testMatVecPerf.cpp
// In this case the matrix and vectors are created on host
// then migrated to different spaces before doing the product
template<typename Config>
void test_space_n_precision_migration( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> inVec,
                                       std::shared_ptr<AMP::LinearAlgebra::Vector> outVec,
                                       AMP::UnitTest *ut )
{
    auto mat_migrate = AMP::LinearAlgebra::createMatrix<Config>( matrix );
    auto mem_loc     = AMP::Utilities::getAllocatorMemoryType<typename Config::allocator_type>();
    auto x = AMP::LinearAlgebra::createVector<typename Config::scalar_t>( inVec, mem_loc );
    auto y = AMP::LinearAlgebra::createVector<typename Config::scalar_t>( outVec, mem_loc );
    x->copy( *inVec );
    y->copy( *outVec );
    x->makeConsistent();
    y->makeConsistent();
    mat_migrate->mult( x, y );
    auto yNorm = static_cast<typename Config::scalar_t>( y->L1Norm() );

    if ( yNorm == static_cast<typename Config::scalar_t>( mat_migrate->numGlobalRows() ) ) {
        ut->passes( "Passes 1 norm test with pseudo Laplacian" );
    } else {
        AMP::pout << "1 Norm " << yNorm << ", number of rows " << mat_migrate->numGlobalRows()
                  << std::endl;
        std::string space_name( AMP::Utilities::getString( mem_loc ) );
        ut->failure( "Migrate to " + space_name + ": Fails 1 norm test with pseudo Laplacian" );
    }
}

template<typename ConfigIn, typename ConfigTo>
void test_accuracy_loss( std::shared_ptr<AMP::LinearAlgebra::Matrix> &matrix,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                         AMP::UnitTest *ut )
{
    auto mat_migrate      = AMP::LinearAlgebra::createMatrix<ConfigTo>( matrix );
    auto mat_migrate_back = AMP::LinearAlgebra::createMatrix<ConfigIn>( mat_migrate );

    auto X = std::dynamic_pointer_cast<AMP::LinearAlgebra::CSRMatrix<ConfigIn>>( matrix );
    auto Y = std::dynamic_pointer_cast<AMP::LinearAlgebra::CSRMatrix<ConfigIn>>( mat_migrate_back );

    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols_X, cols_Y;
        std::vector<double> vals_X, vals_Y;
        X->getRowByGlobalID( i, cols_X, vals_X );
        Y->getRowByGlobalID( i, cols_Y, vals_Y );
        for ( size_t j = 0; j != cols_X.size(); j++ ) {
            if ( std::abs( vals_X[j] - vals_Y[j] ) >
                 std::numeric_limits<typename ConfigTo::scalar_t>::epsilon() ) {
                ut->failure( "Failed. Precision loss higher than expected. " +
                             AMP::Utilities::stringf( "Difference of %e found between entries.",
                                                      std::abs( vals_X[j] - vals_Y[j] ) ) );
                return;
            }
        }
    }
    ut->passes( "Able to migrate with precision change" );
}

size_t matVecTestWithDOFs( AMP::UnitTest *ut,
                           const std::string &type,
                           std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );

    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    auto inVec  = AMP::LinearAlgebra::createVector( dofManager, inVar );
    auto outVec = AMP::LinearAlgebra::createVector( dofManager, outVar );

    inVec->setToScalar( 1.0 );
    inVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    outVec->zero();

    // Create the matrix
    auto matrix = AMP::LinearAlgebra::createMatrix( inVec, outVec, type );
    if ( matrix ) {
        ut->passes( "Able to create a square matrix" );
    } else {
        ut->failure( "Unable to create a square matrix" );
    }

    fillWithPseudoLaplacian( matrix, dofManager );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    size_t nGlobalRows = matrix->numGlobalRows();
    size_t nLocalRows  = matrix->numLocalRows();
    AMP::pout << type << " Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

#ifdef AMP_USE_HYPRE
    using scalar_t = typename AMP::LinearAlgebra::scalar_info<AMP::LinearAlgebra::hypre_real>::type;
#else
    using scalar_t = double;
#endif

    // migrate to space and test
    auto test_space_migration = [=]( AMP::Utilities::MemoryType mem_loc ) -> void {
        auto mat_migrate = AMP::LinearAlgebra::createMatrix( matrix, mem_loc );
        auto x           = AMP::LinearAlgebra::createVector( inVec, mem_loc );
        auto y           = AMP::LinearAlgebra::createVector( outVec, mem_loc );
        x->copy( *inVec );
        y->copy( *outVec );
        x->makeConsistent();
        y->makeConsistent();
        mat_migrate->mult( x, y );
        auto yNorm = static_cast<scalar_t>( y->L1Norm() );

        if ( yNorm == static_cast<scalar_t>( mat_migrate->numGlobalRows() ) ) {
            ut->passes( "Passes 1 norm test with pseudo Laplacian" );
        } else {
            AMP::pout << "1 Norm " << yNorm << ", number of rows " << mat_migrate->numGlobalRows()
                      << std::endl;
            std::string space_name( AMP::Utilities::getString( mem_loc ) );
            ut->failure( "Migrate to " + space_name + ": Fails 1 norm test with pseudo Laplacian" );
        }
    };

    /*
     TESTING MEMORY MIGRATION
    */
    test_space_migration( AMP::Utilities::MemoryType::host );
#ifdef AMP_USE_DEVICE
    test_space_migration( AMP::Utilities::MemoryType::managed );
    test_space_migration( AMP::Utilities::MemoryType::device );
#endif

    using AMP::LinearAlgebra::alloc;
    using AMP::LinearAlgebra::index;
    using AMP::LinearAlgebra::scalar;

    /*
    TESTING MEMORY MIGRATION AND SCALAR_T PRECISION CHANGE
    */
    using Config_hf =
        AMP::LinearAlgebra::CSRConfig<alloc::host, index::i32, index::i64, scalar::f32>;
    test_space_n_precision_migration<Config_hf>( matrix, inVec, outVec, ut );
#ifdef AMP_USE_HYPRE
    using Config_H_hf =
        AMP::LinearAlgebra::CSRConfig<alloc::host, index::i32, index::ill, scalar::f32>;
    test_space_n_precision_migration<Config_H_hf>( matrix, inVec, outVec, ut );
#endif

// actually migrate if device is available
#ifdef AMP_USE_DEVICE
    using Config_mf =
        AMP::LinearAlgebra::CSRConfig<alloc::managed, index::i32, index::i64, scalar::f32>;
    test_space_n_precision_migration<Config_mf>( matrix, inVec, outVec, ut );

    using Config_df =
        AMP::LinearAlgebra::CSRConfig<alloc::device, index::i32, index::i64, scalar::f32>;
    test_space_n_precision_migration<Config_df>( matrix, inVec, outVec, ut );

    #ifdef AMP_USE_HYPRE
    using Config_H_mf =
        AMP::LinearAlgebra::CSRConfig<alloc::managed, index::i32, index::ill, scalar::f32>;
    test_space_n_precision_migration<Config_H_mf>( matrix, inVec, outVec, ut );

    using Config_H_df =
        AMP::LinearAlgebra::CSRConfig<alloc::device, index::i32, index::ill, scalar::f32>;
    test_space_n_precision_migration<Config_H_df>( matrix, inVec, outVec, ut );
    #endif
#endif

    /*
     TESTING ACCURACY LOSS ON SCALAR_T PRECISION CHANGE
    */
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_hf>(
        matrix, dofManager, ut );
#ifdef AMP_USE_HYPRE
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_H_hf>(
        matrix, dofManager, ut );
#endif
#ifdef AMP_USE_DEVICE
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_mf>(
        matrix, dofManager, ut );
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_df>(
        matrix, dofManager, ut );
    #ifdef AMP_USE_HYPRE
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_H_mf>(
        matrix, dofManager, ut );
    test_accuracy_loss<AMP::LinearAlgebra::DefaultHostCSRConfig, Config_H_df>(
        matrix, dofManager, ut );
    #endif
#endif
    return nGlobalRows;
}

size_t matVecTest( AMP::UnitTest *ut, const std::string &input_file )
{
    std::string log_file = "output_testMatMigrate";
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

    size_t nGlobal = matVecTestWithDOFs( ut, "CSRMatrix", scalarDOFs );

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
    ss << "testMatMigrate_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 ) << std::setfill( '0' )
       << nGlobal;
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
