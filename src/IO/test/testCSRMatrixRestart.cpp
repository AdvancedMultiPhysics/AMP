#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/testHelpers/MatrixTests.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>

#define PASS_FAIL( test, MSG )  \
    do {                        \
        if ( test )             \
            ut->passes( MSG );  \
        else                    \
            ut->failure( MSG ); \
    } while ( 0 )

template<typename entries_t>
bool equalRawArrays( const size_t n1, const size_t n2, entries_t const *a1, entries_t const *a2 )
{
    if ( a1 == nullptr && a2 == nullptr )
        return true;
    if ( ( a1 == nullptr && a2 != nullptr ) || ( a1 != nullptr && a2 == nullptr ) ) {
        AMP::pout << "One of the pointers is null " << std::endl;
        return false;
    }
    if ( n1 != n2 ) {
        AMP::pout << "Lengths don't match " << std::endl;
        return false;
    }
    bool pass = true;
    for ( size_t i = 0u; i < n1; ++i ) {
        pass = pass && ( a1[i] == a2[i] );
        if ( !pass )
            AMP::pout << "a1[" << i << "]" << "a2[" << i << "]" << std::endl;
    }
    if ( !pass )
        AMP::pout << "Values don't match " << std::endl;

    return pass;
}

template<typename Config>
void compareLocalMatrixData(
    AMP::UnitTest *ut,
    std::shared_ptr<const AMP::LinearAlgebra::CSRLocalMatrixData<Config>> d1,
    std::shared_ptr<const AMP::LinearAlgebra::CSRLocalMatrixData<Config>> d2 )
{
    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    PASS_FAIL( d1->getMemoryLocation() == d2->getMemoryLocation(), "getMemoryLocation" );
    PASS_FAIL( d1->isDiag() == d2->isDiag(), "isDiag" );
    PASS_FAIL( d1->isEmpty() == d2->isEmpty(), "isEmpty" );
    PASS_FAIL( d1->numberOfNonZeros() == d2->numberOfNonZeros(), "numberOfNonZeros" );
    PASS_FAIL( d1->numLocalRows() == d2->numLocalRows(), "numLocalRows" );
    PASS_FAIL( d1->numLocalColumns() == d2->numLocalColumns(), "numLocalColumns" );
    PASS_FAIL( d1->numUniqueColumns() == d2->numUniqueColumns(), "numUniqueColumns" );
    PASS_FAIL( d1->beginRow() == d2->beginRow(), "beginRow" );
    PASS_FAIL( d1->endRow() == d2->endRow(), "endRow" );
    PASS_FAIL( d1->beginCol() == d2->beginCol(), "beginCol" );
    PASS_FAIL( d1->endCol() == d2->endCol(), "endCol" );

    auto n1       = d1->numUniqueColumns();
    auto n2       = d2->numUniqueColumns();
    const auto c1 = d1->getColumnMap();
    const auto c2 = d2->getColumnMap();
    PASS_FAIL( equalRawArrays<gidx_t>( n1, n2, c1, c2 ), "getColumnMap" );

    const auto &[rs1, cols1, cols_loc1, coeffs1] = d1->getDataFields();
    const auto &[rs2, cols2, cols_loc2, coeffs2] = d2->getDataFields();

    n1 = d1->numberOfNonZeros();
    n2 = d2->numberOfNonZeros();
    PASS_FAIL( equalRawArrays<gidx_t>( n1, n2, cols1, cols2 ), "d_cols" );
    PASS_FAIL( equalRawArrays<lidx_t>( n1, n2, cols_loc1, cols_loc2 ), "d_cols_loc" );
    PASS_FAIL( equalRawArrays<scalar_t>( n1, n2, coeffs1, coeffs2 ), "d_coeffs" );

    n1 = d1->numLocalRows();
    n2 = d2->numLocalRows();
    PASS_FAIL( equalRawArrays<lidx_t>( n1, n2, rs1, rs2 ), "d_row_starts" );
}

template<typename Config>
void compareMatrices( AMP::UnitTest *ut,
                      std::shared_ptr<const AMP::LinearAlgebra::CSRMatrix<Config>> m1,
                      std::shared_ptr<const AMP::LinearAlgebra::CSRMatrix<Config>> m2 )
{
    AMP_ASSERT( m1 && m2 );
    auto d1 = std::dynamic_pointer_cast<const AMP::LinearAlgebra::CSRMatrixData<Config>>(
        m1->getMatrixData() );
    auto d2 = std::dynamic_pointer_cast<const AMP::LinearAlgebra::CSRMatrixData<Config>>(
        m2->getMatrixData() );
    PASS_FAIL( d1->numLocalRows() == d2->numLocalRows(), "numLocalRows" );
    PASS_FAIL( d1->numLocalColumns() == d2->numLocalColumns(), "numLocalColumns" );
    PASS_FAIL( d1->numGlobalRows() == d2->numGlobalRows(), "numGlobalRows" );
    PASS_FAIL( d1->numGlobalColumns() == d2->numGlobalColumns(), "numGlobalColumns" );
    PASS_FAIL( d1->beginRow() == d2->beginRow(), "beginRow" );
    PASS_FAIL( d1->endRow() == d2->endRow(), "endRow" );
    PASS_FAIL( d1->beginCol() == d2->beginCol(), "beginCol" );
    PASS_FAIL( d1->endCol() == d2->endCol(), "endCol" );
    PASS_FAIL( d1->getBackend() == d2->getBackend(), "getBackend" );
    PASS_FAIL( d1->getCoeffType() == d2->getCoeffType(), "getCoeffType" );
    PASS_FAIL( d1->isSquare() == d2->isSquare(), "isSquare" );
    PASS_FAIL( d1->isEmpty() == d2->isEmpty(), "isEmpty" );
    PASS_FAIL( d1->numberOfNonZeros() == d2->numberOfNonZeros(), "numberOfNonZeros" );
    PASS_FAIL( d1->numberOfNonZerosDiag() == d2->numberOfNonZerosDiag(), "numberOfNonZerosDiag" );
    PASS_FAIL( d1->numberOfNonZerosOffDiag() == d2->numberOfNonZerosOffDiag(),
               "numberOfNonZerosOffDiag" );
    PASS_FAIL( d1->hasOffDiag() == d2->hasOffDiag(), "hasOffDiag" );
    PASS_FAIL( d1->getMemoryLocation() == d2->getMemoryLocation(), "getMemoryLocation" );

    compareLocalMatrixData<Config>( ut, d1->getDiagMatrix(), d2->getDiagMatrix() );
    compareLocalMatrixData<Config>( ut, d1->getOffdMatrix(), d2->getOffdMatrix() );
}

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
        dofManager, inVar, true, AMP::Utilities::MemoryType::host );
    auto outVec = AMP::LinearAlgebra::createVector(
        dofManager, outVar, true, AMP::Utilities::MemoryType::host );

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

    x = matrix->createInputVector();
    y = matrix->createOutputVector();
}

template<typename Config>
void testCSRMatrixRestartWithDOFs( AMP::UnitTest *ut,
                                   std::shared_ptr<AMP::Discretization::DOFManager> &dofManager )
{
    std::shared_ptr<AMP::LinearAlgebra::CSRMatrix<Config>> matrix = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> uVec              = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> yVec              = nullptr;
    createMatrixAndVectors<Config>(
        ut, AMP::Utilities::Backend::Serial, dofManager, matrix, uVec, yVec );

    fillWithPseudoLaplacian( matrix, dofManager );
    matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    uVec->setRandomValues();
    // this shouldn't be necessary, but evidently is!
    uVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    yVec->zero();

    matrix->mult( uVec, yVec );

    auto norm1 = yVec->L1Norm();

    AMP::IO::RestartManager writer;
    writer.registerData( matrix, "matrix" );

    // Write the restart data
    writer.write( "testCSRMatrixRestart" );

    // Read and check the restart data
    AMP::IO::RestartManager reader( "testCSRMatrixRestart" );
    auto restartedMatrix = reader.getData<AMP::LinearAlgebra::CSRMatrix<Config>>( "matrix" );
    compareMatrices<Config>( ut, matrix, restartedMatrix );
}

void compareCSRMatrixRestart( AMP::UnitTest *ut, const std::string &input_file )
{
    constexpr auto allocator = AMP::LinearAlgebra::alloc::host;
    using Config             = AMP::LinearAlgebra::DefaultCSRConfig<allocator>;

    std::string log_file = "output_testCompareCSRMatrixRestart";
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
    testCSRMatrixRestartWithDOFs<Config>( ut, scalarDOFs );
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

        files.emplace_back( "input_testMatIO-1" );
    }
    for ( auto &file : files )
        compareCSRMatrixRestart( &ut, file );

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testCSRMatrixRestart_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize() << "_n" << std::setw( 9 )
       << std::setfill( '0' );
    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
