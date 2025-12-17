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
#include "AMP/solvers/amg/MIS2Aggregator.h"
#include "AMP/solvers/amg/SimpleAggregator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

// This test is adapted from testMatVecPerf.cpp (and hence testMatVec.cpp)
// This version forms simple aggregates of nodes of a matrix (A) into another
// matrix (P) then tests that A*(P*x)=(A*P)*x. P is transposed and (Pt*A*P)*x
// is also tested

size_t matMultTestWithDOFs( AMP::UnitTest *ut,
                            std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
#ifdef AMP_USE_DEVICE
    auto inVec = AMP::LinearAlgebra::createVector(
        dofManager, inVar, true, AMP::Utilities::MemoryType::managed );
    auto outVec = AMP::LinearAlgebra::createVector(
        dofManager, outVar, true, AMP::Utilities::MemoryType::managed );
    return 0;
#else
    auto inVec  = AMP::LinearAlgebra::createVector( dofManager, inVar );
    auto outVec = AMP::LinearAlgebra::createVector( dofManager, outVar );
#endif

    // Create the matrix
    auto A = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix" );
    fillWithPseudoLaplacian( A, dofManager );
    A->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    size_t nGlobalRows = A->numGlobalRows();
    size_t nLocalRows  = A->numLocalRows();
    AMP::pout << "CSRMatrix Global rows: " << nGlobalRows << " Local rows: " << nLocalRows
              << std::endl;

    // Create aggregate matrix
    AMP::Solver::AMG::CoarsenSettings crs_settings{ 0.4f, "classical_min" };
    auto agg = std::make_shared<AMP::Solver::AMG::MIS2Aggregator>( crs_settings );
    auto P   = agg->getAggregateMatrix( A );

    // perform A*P SpGEMM
    auto AP = AMP::LinearAlgebra::Matrix::matMatMult( A, P );

    // vectors of ones to apply operators to
    auto xa  = A->createInputVector();
    auto xp  = P->createInputVector();
    auto xap = AP->createInputVector();
    xa->setToScalar( 1.0 );
    xp->setToScalar( 1.0 );
    xap->setToScalar( 1.0 );
    xa->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    xp->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    xap->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // output vectors
    auto ya  = A->createOutputVector();
    auto yp  = AP->createOutputVector();
    auto yap = AP->createOutputVector();
    ya->zero();
    yp->zero();
    yap->zero();

    // vector of ones is in col-span of P so A * xa should be same as AP * xap
    A->mult( xa, ya );
    P->mult( xp, yp );
    AP->mult( xap, yap );
    const auto l1ya    = static_cast<double>( ya->L1Norm() );
    const auto l1yp    = static_cast<double>( yp->L1Norm() );
    const auto l1yap   = static_cast<double>( yap->L1Norm() );
    const auto nrows_d = static_cast<double>( nGlobalRows );

    if ( AMP::Utilities::approx_equal( l1ya, l1yap ) &&
         AMP::Utilities::approx_equal( l1yp, nrows_d ) ) {
        ut->passes( "matMatMult A*P CSRMatrix" );
    } else {
        AMP::pout << "matMatMult A*P CSRMatrix fails with l1ya = " << l1ya << ", l1yp = " << l1yp
                  << ", l1yap = " << l1yap << std::endl;
        ut->failure( "matMatMult A*P CSRMatrix" );
    }

    // Get transpose of aggregate matrix and produce coarsened matrix
    auto Pt    = P->transpose();
    auto PtAP  = AMP::LinearAlgebra::Matrix::matMatMult( Pt, AP );
    auto xptap = PtAP->createInputVector();
    auto yptap = PtAP->createOutputVector();
    xptap->setToScalar( 1.0 );
    xptap->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    yptap->zero();

    // Test coarsened matrix
    PtAP->mult( xptap, yptap );
    const auto l1yptap = static_cast<double>( yptap->L1Norm() );
    if ( AMP::Utilities::approx_equal( l1yptap, nrows_d ) ) {
        ut->passes( "matMatMult Pt*A*P CSRMatrix" );
    } else {
        AMP::pout << "matMatMult Pt*A*P CSRMatrix fails with l1yptap = " << l1yptap
                  << ", expected = " << nrows_d << std::endl;
        ut->failure( "matMatMult Pt*A*P CSRMatrix" );
    }

    return nGlobalRows;
}

size_t matMultTest( AMP::UnitTest *ut, const std::string &input_file )
{
    std::string log_file = "output_testMatMultCoarsen";
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

    return matMultTestWithDOFs( ut, scalarDOFs );
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
        // Same input files as testMatVecPerf
        files.emplace_back( "input_testMatVecPerf-1" );
    }

    size_t nGlobal = 0;
    for ( auto &file : files )
        nGlobal = matMultTest( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testMatMultCoarsen_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 ) << std::setfill( '0' )
       << nGlobal;
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
