#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRMatrixParameters.h"
#include "AMP/matrices/CSRPolicy.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/testHelpers/MatrixDataTransforms.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/libmesh/ReadTestMesh.h"
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

#include <iostream>
#include <string>

void matVecTestWithDOFs( AMP::UnitTest *ut,
                         std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );
    // Create the vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = std::make_shared<AMP::LinearAlgebra::Variable>( "outputVar" );
    auto inVec  = AMP::LinearAlgebra::createVector( dofManager, inVar );
    auto outVec = AMP::LinearAlgebra::createVector( dofManager, outVar );

    std::string type;
#if defined( AMP_USE_TRILINOS )
    type = "ManagedEpetraMatrix";
#elif defined( AMP_USE_PETSC )
    type         = "NativePetscMatrix";
#else
    AMP_ERROR( "This test requires either Trilinos or Petsc matrices to be enabled" );
#endif

    // Create the matrix
    auto matrix = AMP::LinearAlgebra::createMatrix( inVec, outVec, type );
    if ( matrix ) {
        ut->passes( "Able to create a square matrix" );
    } else {
        ut->failure( "Unable to create a square matrix" );
    }

    fillWithPseudoLaplacian( matrix, dofManager );

    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    auto nGlobalRows1 = matrix->numGlobalRows();
    auto nLocalRows1  = matrix->numLocalRows();

#if defined( AMP_USE_HYPRE )
    using Policy = AMP::LinearAlgebra::HypreCSRPolicy;
#else
    using Policy = AMP::LinearAlgebra::CSRPolicy<size_t, int, double>;
#endif
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    gidx_t firstRow, endRow;
    lidx_t nnz_pad;
    std::vector<lidx_t> nnz_d, nnz_od;
    std::vector<lidx_t> rowstart_d, rowstart_od;
    std::vector<gidx_t> cols_d, cols_od;
    std::vector<lidx_t> cols_loc_d, cols_loc_od;
    std::vector<scalar_t> coeffs_d, coeffs_od;

    AMP::LinearAlgebra::transformDofToCSR<Policy>( matrix,
                                                   firstRow,
                                                   endRow,
                                                   nnz_d,
						   rowstart_d,
                                                   cols_d,
                                                   cols_loc_d,
                                                   coeffs_d,
                                                   nnz_od,
						   rowstart_od,
                                                   cols_od,
                                                   cols_loc_od,
                                                   coeffs_od,
						   nnz_pad );

    AMP::LinearAlgebra::CSRMatrixParameters<Policy>::CSRSerialMatrixParameters pars_d
      { nnz_d.data(),
	rowstart_d.data(),
	cols_d.data(),
	cols_loc_d.data(),
	coeffs_d.data() };

    AMP::LinearAlgebra::CSRMatrixParameters<Policy>::CSRSerialMatrixParameters pars_od
      { nnz_od.data(),
	rowstart_od.data(),
	cols_od.data(),
	cols_loc_od.data(),
	coeffs_od.data() };

    auto csrParams =
        std::make_shared<AMP::LinearAlgebra::CSRMatrixParameters<Policy>>( firstRow,
                                                                           endRow,
									   pars_d,
									   pars_od,
									   nnz_pad,
                                                                           comm );

    auto csrMatrix = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Policy>>( csrParams );
    AMP_ASSERT( csrMatrix );

    auto nGlobalRows2 = csrMatrix->numGlobalRows();
    auto nLocalRows2  = csrMatrix->numLocalRows();

    if ( nGlobalRows1 == nGlobalRows2 && nLocalRows1 == nLocalRows2 ) {
        ut->passes( "Number of local and global rows match for default and CSR matrices" );
    } else {
        ut->failure( "Number of local and global rows don't match for default and CSR matrices" );
    }

    auto x  = matrix->getRightVector();
    auto y1 = matrix->getRightVector();
    auto y2 = matrix->getRightVector();

    x->setToScalar( 1.0 );
    // this shouldn't be necessary, but evidently is!
    x->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    y1->zero();
    y2->zero();

    matrix->mult( x, y1 );

    auto y1Norm = static_cast<scalar_t>( y1->L1Norm() );

    if ( y1Norm == static_cast<scalar_t>( matrix->numGlobalRows() ) ) {
        ut->passes( "Passes 1 norm test with pseudo Laplacian with default matvec" );
    } else {
        AMP::pout << "1 Norm " << y1Norm << ", number of rows " << matrix->numGlobalRows()
                  << std::endl;
        ut->failure( "Fails 1 norm test with pseudo Laplacian with default matvec" );
    }

    csrMatrix->mult( x, y2 );
    y2->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    auto y2Norm = static_cast<scalar_t>( y2->L1Norm() );

    if ( y2Norm == static_cast<scalar_t>( csrMatrix->numGlobalRows() ) ) {
        ut->passes( "Passes 1 norm test with pseudo Laplacian with CSR matvec" );
    } else {
        AMP::pout << "CSR matvec: 1 Norm " << y2Norm << ", number of rows "
                  << csrMatrix->numGlobalRows() << std::endl;
        ut->failure( "Fails 1 norm test with pseudo Laplacian with CSR matvec" );
    }

    y1->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    y1->subtract( *y1, *y2 );

    auto maxNorm = static_cast<scalar_t>( y1->maxNorm() );
    auto l2Norm  = static_cast<scalar_t>( y1->L2Norm() );

    if ( maxNorm < 1.0e-14 && l2Norm < 1.0e-14 ) {
        ut->passes( "Matvec with converted CSR matches default matvec" );
    } else {
        AMP::pout << "maxNorm " << maxNorm << ", l2 norm " << l2Norm << std::endl;
        ut->failure( "Matvec with converted CSR matches fails to default matvec" );
    }

    auto csrMatrix2 = AMP::LinearAlgebra::createMatrix( inVec, outVec, "CSRMatrix" );
    fillWithPseudoLaplacian( csrMatrix2, dofManager );
    csrMatrix2->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    y1->zero();
    y2->zero();
    matrix->mult( x, y1 );
    csrMatrix2->mult( x, y2 );
    y1->subtract( *y1, *y2 );

    maxNorm = static_cast<scalar_t>( y1->maxNorm() );
    l2Norm  = static_cast<scalar_t>( y1->L2Norm() );

    if ( maxNorm < 1.0e-14 && l2Norm < 1.0e-14 ) {
        ut->passes( "Matvec with CSR matches default matvec" );
    } else {
        AMP::pout << "maxNorm " << maxNorm << ", l2 norm " << l2Norm << std::endl;
        ut->failure( "Matvec with CSR matches fails to default matvec" );
    }
}

void matVecTest( AMP::UnitTest *ut, std::string input_file )
{

    std::string log_file = "output_testMatVec";
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
    matVecTestWithDOFs( ut, scalarDOFs );

    auto vectorDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );
    matVecTestWithDOFs( ut, vectorDOFs );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> files;

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {

        files.emplace_back( "input_testMatVec-1" );
        files.emplace_back( "input_testMatVec-2" );
    }

    for ( auto &file : files )
        matVecTest( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
