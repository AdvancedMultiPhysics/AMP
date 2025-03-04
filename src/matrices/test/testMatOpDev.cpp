#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRPolicy.h"
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
    std::string type,
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

    // Create the matrix
    // auto matrix = AMP::LinearAlgebra::createMatrix( inVec, outVec, type );

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

    // clang-format off
#ifdef USE_DEVICE
    params->d_backend = AMP::Utilities::Backend::hip_cuda;
#else
    params->d_backend = AMP::Utilities::Backend::serial;
#endif
    // clang-format on

    // Create the matrix
    auto data = std::make_shared<AMP::LinearAlgebra::CSRMatrixData<Policy, Allocator>>( params );
    matrix    = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>>( data );
    // Initialize the matrix
    matrix->zero();
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    ///// END: Temporary before updating create matrix

    if ( matrix ) {
        ut->passes( type + ": Able to create a square matrix" );
    } else {
        ut->failure( type + ": Unable to create a square matrix" );
    }

    x = matrix->getRightVector();
    y = matrix->getLeftVector();
}

template<typename Policy, class Allocator>
void testGetSetValues( AMP::UnitTest *ut,
                       std::string type,
                       std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> matrix = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                            = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                            = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, matrix, x, y );

    fillWithPseudoLaplacian( matrix, dofManager );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        std::vector<size_t> cols;
        std::vector<double> vals;
        matrix->getRowByGlobalID( i, cols, vals );
        for ( size_t j = 0; j != cols.size(); j++ ) {
            double ans   = ( i == cols[j] ) ? cols.size() : -1.;
            double value = matrix->getValueByGlobalID( i, cols[j] );
            if ( vals[j] != ans || value != vals[j] ) {
                ut->failure( "bad value in matrix " + matrix->type() );
                return;
            }
        }
    }
    ut->passes( type + ": Able to get and set" );
}

template<typename Policy, class Allocator>
void testMatvecWithDOFs( AMP::UnitTest *ut,
                         std::string type,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager,
                         bool testTranspose )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> matrix = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                            = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                            = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, matrix, x, y );

    fillWithPseudoLaplacian( matrix, dofManager );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    size_t nGlobalRows = matrix->numGlobalRows();
    size_t nLocalRows  = matrix->numLocalRows();
    AMP::pout << type << " Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    matrix->mult( x, y );

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );

    if ( yNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
        ut->passes( type + ": Passes 1 norm test with pseudo Laplacian" );
    } else {
        AMP::pout << "1 Norm " << yNorm << ", number of rows " << matrix->numGlobalRows()
                  << std::endl;
        ut->failure( type + ": Fails 1 norm test with pseudo Laplacian" );
    }
    if ( testTranspose ) {
        // Repeat test with transpose multiply (Laplacian is symmetric)
        y->setToScalar( 1.0 );
        y->makeConsistent();
        x->zero();
        matrix->multTranspose( y, x );

        auto xNorm = static_cast<scalar_t>( x->L1Norm() );

        if ( xNorm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
            ut->passes( type + ": Passes 1 norm test with pseudo Laplacian transpose" );
        } else {
            AMP::pout << "Transpose 1 Norm " << xNorm << ", number of rows "
                      << matrix->numGlobalRows() << std::endl;
            ut->failure( type + ": Fails 1 norm test with pseudo Laplacian transpose" );
        }
    }
}


template<typename Policy, class Allocator>
void testAXPY( AMP::UnitTest *ut,
               std::string type,
               std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> X = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> xX                      = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> yX                      = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, X, xX, yX );
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> Y = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> xY                      = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> yY                      = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, Y, xY, yY );

    fillWithPseudoLaplacian( X, dofManager );
    X->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    fillWithPseudoLaplacian( Y, dofManager );
    Y->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    // X = Y = pL
    // X = -2Y + X = -pL
    double alpha = -2.;
    X->axpy( alpha, Y );

    xX->setToScalar( 1.0 );
    xX->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    yX->zero();
    xY->copyVector( xX );
    yY->zero();
    X->mult( xX, yX );
    Y->mult( xY, yY );
    // Check pL * one + (-pL*one) = 0
    auto z = X->getLeftVector();
    z->zero();
    z->add( *yX, *yY );
    if ( z->L1Norm() < std::numeric_limits<scalar_t>::epsilon() )
        ut->passes( type + ": AXPY succeeded" );
    else
        ut->failure( type + ": AXPY failed" );
}

template<typename Policy, class Allocator>
void testScale( AMP::UnitTest *ut,
                std::string type,
                std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                       = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                       = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, A, x, y );

    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    size_t nGlobalRows = A->numGlobalRows();

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    A->scale( 2.5 );
    A->mult( x, y );

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( ( yNorm - 2.5 * nGlobalRows ) < std::numeric_limits<scalar_t>::epsilon() ) {
        ut->passes( type + ": Able to scale matrix" );
    } else {
        ut->failure( type + ": Fails matrix scaling" );
    }
}


template<typename Policy, class Allocator>
void testSetScalar( AMP::UnitTest *ut,
                    std::string type,
                    std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                       = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                       = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, A, x, y );

    A->setScalar( 1. );

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y->zero();

    A->mult( x, y );

    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        auto cols        = A->getColumnIDs( i );
        const auto ncols = cols.size();
        if ( ncols != y->getValueByGlobalID( i ) ) {
            ut->failure( type + ": Fails to set matrix to scalar" );
            return;
        }
    }
    ut->passes( type + ": Able to set matrix to scalar" );

    A->zero();
    y->zero();
    A->mult( x, y );

    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm < std::numeric_limits<scalar_t>::epsilon() ) {
        ut->passes( type + ": Able to set matrix to 0" );
    } else {
        ut->failure( type + ": Fails to set matrix to 0" );
    }
}

template<typename Policy, class Allocator>
void testGetSetDiagonal( AMP::UnitTest *ut,
                         std::string type,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                       = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                       = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, A, x, y );

    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    x->setToScalar( 1.0 );
    A->setDiagonal( x );
    A->extractDiagonal( y );
    auto yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm == static_cast<scalar_t>( A->numGlobalRows() ) ) {
        ut->passes( type + ": Able to set/get diagonal" );
    } else {
        ut->failure( type + ": Fails to set/get diagonal" );
    }

    A->setIdentity();
    y->zero();
    A->mult( x, y );
    yNorm = static_cast<scalar_t>( y->L1Norm() );
    if ( yNorm == static_cast<scalar_t>( A->numGlobalRows() ) ) {
        ut->passes( type + ": Able to set to Identity" );
    } else {
        ut->failure( type + ": Fails to set to Identity" );
    }
}

template<typename Policy, class Allocator>
void testLinfNorm( AMP::UnitTest *ut,
                   std::string type,
                   std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Policy::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Policy, Allocator>> A = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> x                       = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> y                       = nullptr;
    createMatrixAndVectors<Policy, Allocator>( ut, type, dofManager, A, x, y );

    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    size_t lmax = 0;
    for ( size_t i = dofManager->beginDOF(); i != dofManager->endDOF(); i++ ) {
        auto cols = A->getColumnIDs( i );
        lmax      = std::max( lmax, cols.size() * 2 - 1 );
    }
    auto comm  = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto ANorm = static_cast<scalar_t>( A->LinfNorm() );
    auto rNorm = static_cast<scalar_t>( comm.maxReduce<scalar_t>( lmax ) );
    if ( ANorm == rNorm ) {
        ut->passes( type + ": Able to compute Linf" );
    } else {
        ut->failure( type + ": Fails to compute Linf" );
    }
}

void matDeviceOperationsTest( AMP::UnitTest *ut, std::string input_file )
{
    // clang-format off
#ifdef USE_DEVICE
    using Allocator      = AMP::ManagedAllocator<void>;
    bool testTransposeOp = false;
#else
    using Allocator      = AMP::HostAllocator<void>;
    bool testTransposeOp = true;
#endif

#if defined( AMP_USE_HYPRE )
    using Policy = AMP::LinearAlgebra::HypreCSRPolicy;
#else
    using Policy = AMP::LinearAlgebra::CSRPolicy<size_t, int, double>;
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
    testGetSetValues<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
    testMatvecWithDOFs<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs, testTransposeOp );
    testAXPY<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
    testScale<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
    testSetScalar<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
    testGetSetDiagonal<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
    testLinfNorm<Policy, Allocator>( ut, "CSRMatrix", scalarDOFs );
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
