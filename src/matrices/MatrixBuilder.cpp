#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/DenseSerialMatrix.h"
#include "AMP/matrices/GetRowHelper.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/data/DenseSerialMatrixData.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#ifdef AMP_USE_PETSC
    #include "AMP/matrices/petsc/NativePetscMatrix.h"
    #include "AMP/vectors/petsc/PetscHelpers.h"
#endif
#ifdef AMP_USE_TRILINOS
    #include "AMP/matrices/trilinos/EpetraMatrixData.h"
    #include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"
#endif

#ifdef AMP_USE_HYPRE
    #include "HYPRE.h"
    #include "HYPRE_IJ_mv.h"
    #include "HYPRE_utilities.h"
#endif

#include <functional>

namespace AMP::LinearAlgebra {


/********************************************************
 * Set the default matrix type                           *
 * Definition set by CMake variable DEFAULT_MATRIX       *
 ********************************************************/
std::string getDefaultMatrixType() { return DEFAULT_MATRIX; }


/********************************************************
 * Check if we have a spare matrix available             *
 ********************************************************/
#if defined( AMP_USE_TRILINOS ) || defined( AMP_USE_PETSC )
bool haveSparseMatrix() { return true; }
#else
bool haveSparseMatrix() { return false; }
#endif


/********************************************************
 * Build a ManagedEpetraMatrix                           *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Matrix>
createManagedMatrix( [[maybe_unused]] AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                     [[maybe_unused]] AMP::LinearAlgebra::Vector::shared_ptr rightVec,
                     [[maybe_unused]] const std::function<std::vector<size_t>( size_t )> &getRow,
                     [[maybe_unused]] const std::string &type )
{
    if ( type == "ManagedEpetraMatrix" ) {
#if defined( AMP_USE_TRILINOS )
        // Get the DOFs
        auto leftDOF  = leftVec->getDOFManager();
        auto rightDOF = rightVec->getDOFManager();
        if ( leftDOF->getComm().compare( rightDOF->getComm() ) == 0 )
            AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested" );
        AMP_MPI comm = leftDOF->getComm();
        if ( comm.getSize() == 1 )
            comm = AMP_MPI( AMP_COMM_SELF );

        // Create the matrix parameters
        auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
            leftDOF,
            rightDOF,
            comm,
            leftVec->getVariable(),
            rightVec->getVariable(),
            leftVec->getCommunicationList(),
            rightVec->getCommunicationList(),
            getRow );

        // Create the matrix
        auto newMatrixData = std::make_shared<AMP::LinearAlgebra::EpetraMatrixData>( params );
        auto newMatrix = std::make_shared<AMP::LinearAlgebra::ManagedEpetraMatrix>( newMatrixData );
        newMatrix->zero();
        newMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
        return newMatrix;
#else
        AMP_ERROR( "Unable to build ManagedEpetraMatrix without Trilinos" );
#endif
    } else {
        AMP_ERROR( "Unknown ManagedMatrix type" );
        return nullptr;
    }
}

/********************************************************
 * Build a CSRMatrix                                    *
 ********************************************************/
template<typename Config>
std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix( AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                 AMP::LinearAlgebra::Vector::shared_ptr rightVec,
                 const std::function<std::vector<size_t>( size_t )> &getRow )
{
    auto memType = AMP::Utilities::getMemoryType( rightVec->getRawDataBlockAsVoid( 0 ) );
    auto accelerationBackend = AMP::Utilities::getDefaultBackend( memType );
    return createCSRMatrix<Config>( leftVec, rightVec, getRow, accelerationBackend );
}

template<typename Config>
std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix( AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                 AMP::LinearAlgebra::Vector::shared_ptr rightVec,
                 const std::function<std::vector<size_t>( size_t )> &getRow,
                 AMP::Utilities::Backend accelerationBackend )
{
    using gidx_t = typename Config::gidx_t;
    using lidx_t = typename Config::lidx_t;

    // Get the DOFs
    auto leftDOF  = leftVec->getDOFManager();
    auto rightDOF = rightVec->getDOFManager();
    if ( leftDOF->getComm().compare( rightDOF->getComm() ) == 0 )
        AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested, and needs to "
                   "be fixed" );
    AMP_MPI comm = leftDOF->getComm();
    if ( comm.getSize() == 1 )
        comm = AMP_MPI( AMP_COMM_SELF );

    std::shared_ptr<AMP::LinearAlgebra::MatrixParametersBase> params;

    if ( getRow ) {
        // explicit getRow function passed in, wrap into MatrixParameters
        params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
            leftDOF,
            rightDOF,
            comm,
            leftVec->getVariable(),
            rightVec->getVariable(),
            leftVec->getCommunicationList(),
            rightVec->getCommunicationList(),
            accelerationBackend,
            getRow );
    } else {
        // no getRow function available
        // Use GetRowHelper class to build a usable default
        auto rowHelper = std::make_shared<GetRowHelper>( leftDOF, rightDOF );

        std::function<void( const gidx_t, lidx_t &, lidx_t & )> getRowNNZ =
            [rowHelper]( const gidx_t row, lidx_t &nnz_local, lidx_t &nnz_remote ) {
                rowHelper->NNZ( row, nnz_local, nnz_remote );
            };
        std::function<void( const gidx_t, gidx_t *, gidx_t * )> getRowCols =
            [rowHelper]( const gidx_t row, gidx_t *cols_local, gidx_t *cols_remote ) {
                rowHelper->getRow( row, cols_local, cols_remote );
            };
        params = std::make_shared<AMP::LinearAlgebra::AMPCSRMatrixParameters<Config>>(
            leftDOF,
            rightDOF,
            comm,
            leftVec->getVariable(),
            rightVec->getVariable(),
            leftVec->getCommunicationList(),
            rightVec->getCommunicationList(),
            accelerationBackend,
            getRowNNZ,
            getRowCols );
    }

    // Create the matrix
    auto data      = std::make_shared<AMP::LinearAlgebra::CSRMatrixData<Config>>( params );
    auto newMatrix = std::make_shared<AMP::LinearAlgebra::CSRMatrix<Config>>( data );
    // Initialize the matrix
    newMatrix->zero();
    newMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    return newMatrix;
}

template std::shared_ptr<AMP::LinearAlgebra::Matrix> createCSRMatrix<DefaultCSRConfig<alloc::host>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow );

template std::shared_ptr<AMP::LinearAlgebra::Matrix> createCSRMatrix<DefaultCSRConfig<alloc::host>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow,
    AMP::Utilities::Backend accelerationBackend );

#ifdef USE_DEVICE
template std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix<DefaultCSRConfig<alloc::managed>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow );

template std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix<DefaultCSRConfig<alloc::managed>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow,
    AMP::Utilities::Backend accelerationBackend );

template std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix<DefaultCSRConfig<alloc::device>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow );

template std::shared_ptr<AMP::LinearAlgebra::Matrix>
createCSRMatrix<DefaultCSRConfig<alloc::device>>(
    AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    const std::function<std::vector<size_t>( size_t )> &getRow,
    AMP::Utilities::Backend accelerationBackend );
#endif

/********************************************************
 * Build a DenseSerialMatrix                             *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Matrix>
createDenseSerialMatrix( AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                         AMP::LinearAlgebra::Vector::shared_ptr rightVec )
{
    // Get the DOFs
    auto leftDOF  = leftVec->getDOFManager();
    auto rightDOF = rightVec->getDOFManager();
    if ( leftDOF->getComm().compare( rightDOF->getComm() ) == 0 )
        AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested, and needs to "
                   "be fixed" );
    AMP_MPI comm = leftDOF->getComm();
    if ( comm.getSize() == 1 )
        comm = AMP_MPI( AMP_COMM_SELF );
    else
        AMP_ERROR( "serial dense matrix does not support parallel matrices" );

    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
        leftDOF, rightDOF, comm, leftVec->getVariable(), rightVec->getVariable() );

    // Create the matrix
    auto data      = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrixData>( params );
    auto newMatrix = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrix>( data );

    // Initialize the matrix
    newMatrix->zero();
    newMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    return newMatrix;
}

/********************************************************
 * Build a NativePetscMatrix                             *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Matrix> createNativePetscMatrix(
    [[maybe_unused]] AMP::LinearAlgebra::Vector::shared_ptr leftVec,
    [[maybe_unused]] AMP::LinearAlgebra::Vector::shared_ptr rightVec,
    [[maybe_unused]] const std::function<std::vector<size_t>( size_t )> &getRow )
{
#if defined( AMP_USE_PETSC )
    // Get the DOFs
    auto leftDOF  = leftVec->getDOFManager();
    auto rightDOF = rightVec->getDOFManager();
    if ( leftDOF->getComm().compare( rightDOF->getComm() ) == 0 )
        AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested, and needs to "
                   "be fixed" );
    AMP_MPI comm = leftDOF->getComm();

    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
        leftDOF, rightDOF, comm, leftVec->getVariable(), rightVec->getVariable(), getRow );

    // Create the matrix
    auto newMatrix = std::make_shared<AMP::LinearAlgebra::NativePetscMatrix>( params );
    // Initialize the matrix
    newMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    return newMatrix;
#else
    AMP_ERROR( "Unable to build NativePetscMatrix without Petsc" );
    return nullptr;
#endif
}


/********************************************************
 * Test the matrix to ensure it is valid                 *
 ********************************************************/
static void test( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    auto leftDOF         = matrix->getLeftDOFManager();
    auto rightDOF        = matrix->getRightDOFManager();
    size_t N_local_row1  = leftDOF->numLocalDOF();
    size_t N_local_row2  = matrix->numLocalRows();
    size_t N_local_col1  = rightDOF->numLocalDOF();
    size_t N_local_col2  = matrix->numLocalColumns();
    size_t N_global_row1 = leftDOF->numGlobalDOF();
    size_t N_global_row2 = matrix->numGlobalRows();
    size_t N_global_col1 = rightDOF->numGlobalDOF();
    size_t N_global_col2 = matrix->numGlobalColumns();
    AMP_ASSERT( N_local_row1 == N_local_row2 );
    AMP_ASSERT( N_local_col1 == N_local_col2 );
    AMP_ASSERT( N_global_row1 == N_global_row2 );
    AMP_ASSERT( N_global_col1 == N_global_col2 );
    AMP_ASSERT( !matrix->getComm().isNull() );
}


/********************************************************
 * Matrix builder                                        *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Matrix>
createMatrix( AMP::LinearAlgebra::Vector::shared_ptr rightVec,
              AMP::LinearAlgebra::Vector::shared_ptr leftVec,
              std::string type,
              std::function<std::vector<size_t>( size_t )> getRow )
{

    // Find memory type associated with (right) vector
    auto memType = AMP::Utilities::getMemoryType( rightVec->getRawDataBlockAsVoid( 0 ) );
    auto accelerationBackend = AMP::Utilities::getDefaultBackend( memType );
    return createMatrix( rightVec, leftVec, accelerationBackend, type, getRow );
}

std::shared_ptr<AMP::LinearAlgebra::Matrix>
createMatrix( AMP::LinearAlgebra::Vector::shared_ptr rightVec,
              AMP::LinearAlgebra::Vector::shared_ptr leftVec,
              AMP::Utilities::Backend accelerationBackend,
              std::string type,
              std::function<std::vector<size_t>( size_t )> getRow )
{
    if ( type == "auto" )
        type = getDefaultMatrixType();
    if ( type == "NULL" )
        return nullptr; // Special case to return nullptr

    // Create the default getRow function (if not provided)
    bool useDefaultGetRow = false;
    if ( !getRow ) {
        useDefaultGetRow    = true;
        const auto leftDOF  = leftVec->getDOFManager().get();
        const auto rightDOF = rightVec->getDOFManager().get();
        getRow              = [leftDOF, rightDOF]( size_t row ) {
            auto id = leftDOF->getElementID( row );
            return rightDOF->getRowDOFs( id );
        };
    }

    // Find memory type associated with (right) vector
    auto memType = AMP::Utilities::getMemoryType( rightVec->getRawDataBlockAsVoid( 0 ) );

    // Build the matrix
    std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix;
    if ( type == "ManagedEpetraMatrix" ) {
        matrix = createManagedMatrix( leftVec, rightVec, getRow, type );
    } else if ( type == "NativePetscMatrix" ) {
        matrix = createNativePetscMatrix( leftVec, rightVec, getRow );
    } else if ( type == "CSRMatrix" ) {
        // CSRMatrixData can use GetRowHelper class internally
        // only pass in getRow function if explicitly provided
        std::function<std::vector<size_t>( size_t )> emptyGetRow;
        if ( memType <= AMP::Utilities::MemoryType::host ) {
            matrix = createCSRMatrix<DefaultCSRConfig<alloc::host>>(
                leftVec, rightVec, useDefaultGetRow ? emptyGetRow : getRow, accelerationBackend );
        } else if ( memType == AMP::Utilities::MemoryType::managed ) {
#ifdef USE_DEVICE
            matrix = createCSRMatrix<DefaultCSRConfig<alloc::managed>>(
                leftVec, rightVec, useDefaultGetRow ? emptyGetRow : getRow, accelerationBackend );
#else
            AMP_ERROR( "Creating CSRMatrix in managed memory requires HIP or CUDA support" );
#endif
        } else if ( memType == AMP::Utilities::MemoryType::device ) {
#ifdef USE_DEVICE
            matrix = createCSRMatrix<DefaultCSRConfig<alloc::device>>(
                leftVec, rightVec, useDefaultGetRow ? emptyGetRow : getRow, accelerationBackend );
#else
            AMP_ERROR( "Creating CSRMatrix in device memory requires HIP or CUDA support" );
#endif
        } else {
            AMP_ERROR( "Unknown memory space in createMatrix" );
        }
    } else if ( type == "DenseSerialMatrix" ) {
        matrix = createDenseSerialMatrix( leftVec, rightVec );
    } else {
        AMP_ERROR( "Unknown matrix type to build" );
    }
    // Check that the matrix is valid
    test( matrix );
    return matrix;
}


/********************************************************
 * Create Matrix from PETSc Mat                          *
 ********************************************************/
#if defined( AMP_USE_PETSC )
std::shared_ptr<Matrix> createMatrix( Mat M, bool deleteable )
{
    auto matrix = std::make_shared<NativePetscMatrix>( M, deleteable );
    AMP_ASSERT( !matrix->getComm().isNull() );
    return matrix;
}
#endif


} // namespace AMP::LinearAlgebra
