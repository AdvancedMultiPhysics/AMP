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

size_t matVecTestWithDOFs( AMP::UnitTest *ut,
                           std::string type,
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
    auto test_space = [=]( AMP::Utilities::MemoryType mem_loc ) -> void {
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

    // testing on host is really a clone
    test_space( AMP::Utilities::MemoryType::host );

    // actually migrate if device is available
#ifdef AMP_USE_DEVICE
    test_space( AMP::Utilities::MemoryType::managed );
    test_space( AMP::Utilities::MemoryType::device );
#endif


    return nGlobalRows;
}

size_t matVecTest( AMP::UnitTest *ut, std::string input_file )
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
