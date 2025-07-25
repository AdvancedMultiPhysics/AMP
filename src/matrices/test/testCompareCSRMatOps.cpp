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
                             std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> &matrix,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &x,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> &y )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );

    // using  AMP::ManagedAllocator<void>;
    auto inVec = AMP::LinearAlgebra::createVector(
        dofManager, inVar, true, AMP::Utilities::MemoryType::managed );
    auto outVec = AMP::LinearAlgebra::createVector(
        dofManager, outVar, true, AMP::Utilities::MemoryType::managed );

    // Create the matrix

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
        ut->passes( " Able to create a square matrix" );
    } else {
        ut->failure( " Unable to create a square matrix" );
    }

    x = matrix->getRightVector();
    y = matrix->getLeftVector();
}

template<typename Config>
void testMatvecWithDOFs( AMP::UnitTest *ut,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    using scalar_t = typename Config::scalar_t;

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> dev_mat = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> dev_x              = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> dev_y              = nullptr;
    createMatrixAndVectors<Config>(
        ut, AMP::Utilities::Backend::hip_cuda, dofManager, dev_mat, dev_x, dev_y );

    fillWithPseudoLaplacian( dev_mat, dofManager );
    dev_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    dev_x->setRandomValues();
    // this shouldn't be necessary, but evidently is!
    dev_x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    dev_y->zero();

    dev_mat->mult( dev_x, dev_y );

    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> kks_mat = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> kks_x              = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> kks_y              = nullptr;
    createMatrixAndVectors<Config>(
        ut, AMP::Utilities::Backend::kokkos, dofManager, kks_mat, kks_x, kks_y );

    fillWithPseudoLaplacian( kks_mat, dofManager );
    kks_mat->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    kks_x->copyVector( dev_x );
    kks_x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    kks_y->zero();

    kks_mat->mult( kks_x, kks_y );

    const auto tolerance = 2 * std::numeric_limits<scalar_t>::epsilon();
    {
        auto sum = kks_mat->getLeftVector();
        sum->axpy( -1.0, *kks_y, *dev_y );
        auto norm = static_cast<scalar_t>( sum->L1Norm() / kks_y->L1Norm() );
        if ( norm < tolerance ) {
            ut->passes( "Matvec succeed when compared using kks op" );
        } else {
            ut->failure( AMP::Utilities::stringf(
                "Matvec fails: Relative norm of the difference is %e and tolerance is %e",
                norm,
                tolerance ) );
        }
    }
    {
        auto sum = dev_mat->getLeftVector();
        sum->axpy( -1.0, *kks_y, *dev_y );
        auto norm = static_cast<scalar_t>( sum->L1Norm() / dev_y->L1Norm() );
        if ( norm < tolerance ) {
            ut->passes( "Matvec succeed when compared using dev op" );
        } else {
            ut->failure( AMP::Utilities::stringf(
                "Matvec fails: Relative norm of the difference is %e and tolerance is %e",
                norm,
                tolerance ) );
        }
    }
}

void compareCSRMatOps( AMP::UnitTest *ut, std::string input_file )
{
    constexpr auto allocator = AMP::LinearAlgebra::alloc::managed;
    using Config             = AMP::LinearAlgebra::DefaultCSRConfig<allocator>;

    std::string log_file = "output_testCompareCSRMatOps";
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
    testMatvecWithDOFs<Config>( ut, scalarDOFs );
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
        compareCSRMatOps( &ut, file );

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
