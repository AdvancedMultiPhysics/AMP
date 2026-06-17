#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/MatrixParameters.h"
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

template<typename Config>
void createMatrixAndVectors( AMP::UnitTest *ut,
                             AMP::Utilities::Backend backend,
                             std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                             std::shared_ptr<AMP::LinearAlgebra::Matrix> &matrix,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &x,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &y )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    matrix =
        pseudoLaplacianFromDOFs( "CSRMatrix", dofManager, backend, Config::mem_loc, inVar, outVar );

    if ( matrix ) {
        ut->passes( " Able to create a square matrix" );
    } else {
        ut->failure( " Unable to create a square matrix" );
    }

    x = matrix->createInputVector();
    y = matrix->createOutputVector();
}

template<class Config>
void testMatvecWithDOFs( AMP::UnitTest *ut,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                         AMP::Utilities::Backend backend )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x      = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y      = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, matrix, x, y );

    size_t nGlobalRows = matrix->numGlobalRows();
    size_t nLocalRows  = matrix->numLocalRows();
    AMP::pout << "CSRMatrix Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    matrix->mult( x, y );

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );

    if ( yNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
        ut->passes( "CSRMatrix: Passes 1 norm test with pseudo Laplacian" );
    } else {
        AMP::pout << "1 Norm " << yNorm << ", number of rows " << matrix->numGlobalRows()
                  << std::endl;
        ut->failure( "CSRMatrix: Fails 1 norm test with pseudo Laplacian" );
    }
}


template<class Config>
void testAXPY( AMP::UnitTest *ut,
               std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
               AMP::Utilities::Backend backend )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> X  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> rX = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> lX = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, X, rX, lX );
    std::shared_ptr<AMP::LinearAlgebra::Matrix> Y  = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> rY = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> lY = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, Y, rY, lY );

    // X = Y = pL
    // X = -2Y + X = -pL
    scalar_t alpha = -2.;
    X->axpy( alpha, Y );

    rX->setToScalar( 1.0 );
    rX->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    lX->zero();
    X->mult( rX, lX );

    rY->copyVector( rX );
    rY->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    lY->zero();
    Y->mult( rY, lY );

    // Check pL * one + (-pL*one) = 0
    auto z = X->createOutputVector();
    z->zero();
    z->add( *lX, *lY );

    auto norm = static_cast<double>( z->L1Norm() );
    if ( norm < std::numeric_limits<scalar_t>::epsilon() )
        ut->passes( "CSRMatrix: AXPY succeeded" );
    else {
        AMP::pout << "L1 norm of difference is  " << norm << std::endl;
        ut->failure( "CSRMatrix: AXPY failed" );
    }
}

template<typename Config>
void testScale( AMP::UnitTest *ut,
                std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                AMP::Utilities::Backend backend )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, A, x, y );

    size_t nGlobalRows = A->numGlobalRows();

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    A->scale( 2.5 );
    A->mult( x, y );

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( ( yNorm - 2.5 * nGlobalRows ) < std::numeric_limits<scalar_t>::epsilon() ) {
        ut->passes( "CSRMatrix: Able to scale matrix" );
    } else {
        ut->failure( "CSRMatrix: Fails matrix scaling" );
    }
}


template<typename Config>
void testSetScalar( AMP::UnitTest *ut,
                    std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                    AMP::Utilities::Backend backend )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, A, x, y );
    auto matData =
        std::dynamic_pointer_cast<AMP::LinearAlgebra::CSRMatrixData<Config>>( A->getMatrixData() );
    const auto rank_nnz  = static_cast<scalar_t>( matData->numberOfNonZeros() );
    const auto total_nnz = AMP::AMPManager::getCommWorld().sumReduce( rank_nnz );

    // set to all ones and multiply, should give L1 norm matching total_nnz
    A->setScalar( 1. );
    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();
    A->mult( x, y );
    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( std::fabs( yNorm - total_nnz ) >
         std::numeric_limits<scalar_t>::epsilon() ) { // should generally be exact
        ut->failure( "CSRMatrix: Fails to set matrix to scalar" );
        return;
    }
    ut->passes( "CSRMatrix: Able to set matrix to scalar" );

    A->zero();
    y->zero();
    A->mult( x, y );
    yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm < std::numeric_limits<scalar_t>::epsilon() ) {
        ut->passes( "CSRMatrix: Able to set matrix to 0" );
    } else {
        ut->failure( "CSRMatrix: Fails to set matrix to 0" );
    }
}

template<typename Config>
void testGetSetDiagonal( AMP::UnitTest *ut,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                         AMP::Utilities::Backend backend )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::Matrix> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y = nullptr;
    createMatrixAndVectors<Config>( ut, backend, dofManager, A, x, y );

    x->setToScalar( 1.0 );
    A->setDiagonal( x );
    A->extractDiagonal( y );
    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm == static_cast<scalar_t>( A->numGlobalRows() ) ) {
        ut->passes( "CSRMatrix: Able to set/get diagonal" );
    } else {
        ut->failure( "CSRMatrix: Fails to set/get diagonal" );
    }

    A->setIdentity();
    y->zero();
    A->mult( x, y );
    yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm == static_cast<scalar_t>( A->numGlobalRows() ) ) {
        ut->passes( "CSRMatrix: Able to set to Identity" );
    } else {
        ut->failure( "CSRMatrix: Fails to set to Identity" );
    }
}

void matDeviceOperationsTest( AMP::UnitTest *ut, const std::string &input_file )
{
    // clang-format off
#ifdef AMP_USE_DEVICE
    using Config = AMP::LinearAlgebra::DefaultCSRConfig<AMP::LinearAlgebra::alloc::device>;
    const auto backend = AMP::Utilities::Backend::Hip_Cuda;
#else
    using Config = AMP::LinearAlgebra::DefaultCSRConfig;
    const auto backend = AMP::Utilities::Backend::Serial;
#endif

    // clang-format on

    std::string log_file = "output_testMatOpDev";
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
    testMatvecWithDOFs<Config>( ut, scalarDOFs, backend );
    testAXPY<Config>( ut, scalarDOFs, backend );
    testScale<Config>( ut, scalarDOFs, backend );
    testSetScalar<Config>( ut, scalarDOFs, backend );
    testGetSetDiagonal<Config>( ut, scalarDOFs, backend );
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
    ss << "testMatOpDev_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
