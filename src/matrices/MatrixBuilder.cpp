#ifdef USE_AMP_VECTORS

#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/DenseSerialMatrix.h"
#include "AMP/matrices/ManagedMatrix.h"
#include "AMP/matrices/ManagedMatrixParameters.h"
#include "AMP/utils/Utilities.h"

#ifdef USE_EXT_TRILINOS
#include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"
#endif
#ifdef USE_EXT_PETSC
#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/vectors/petsc/PetscHelpers.h"
#endif

#include <functional>

#ifdef USE_EXT_TRILINOS
#include <Epetra_CrsMatrix.h>
#endif


namespace AMP::LinearAlgebra {


/********************************************************
 * Build a ManagedPetscMatrix                             *
 ********************************************************/
AMP::LinearAlgebra::Matrix::shared_ptr
createManagedMatrix( AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                     AMP::LinearAlgebra::Vector::shared_ptr rightVec,
                     const std::function<std::vector<size_t>( size_t )> &getRow,
                     const std::string &type )
{
    // Get the DOFs
    auto leftDOF  = leftVec->getDOFManager();
    auto rightDOF = rightVec->getDOFManager();
    if ( leftDOF->getComm().compare( rightVec->getComm() ) == 0 )
        AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested" );
    AMP_MPI comm = leftDOF->getComm();
    if ( comm.getSize() == 1 )
        comm = AMP_MPI( AMP_COMM_SELF );

    // Create the matrix parameters
    auto params =
        std::make_shared<AMP::LinearAlgebra::ManagedMatrixParameters>( leftDOF, rightDOF, comm );
    params->d_CommListLeft  = leftVec->getCommunicationList();
    params->d_CommListRight = rightVec->getCommunicationList();
    params->d_VariableLeft  = leftVec->getVariable();
    params->d_VariableRight = rightVec->getVariable();

    // Add the row sizes and local columns to the matrix parameters
    std::set<size_t> columns;
    size_t row_start = leftDOF->beginDOF();
    size_t row_end   = leftDOF->endDOF();
    for ( size_t row = row_start; row < row_end; row++ ) {
        auto col = getRow( row );
        params->setEntriesInRow( row - row_start, col.size() );
        for ( auto &tmp : col )
            columns.insert( tmp );
    }
    params->addColumns( columns );

    // Create the matrix
    std::shared_ptr<AMP::LinearAlgebra::ManagedMatrix> newMatrix;
    if ( type == "ManagedEpetraMatrix" ) {
#if defined( USE_EXT_TRILINOS )
        auto mat = std::make_shared<AMP::LinearAlgebra::ManagedEpetraMatrix>( params );
        mat->setEpetraMaps( leftVec, rightVec );
        newMatrix = mat;
#else
        AMP_ERROR( "Unable to build ManagedEpetraMatrix without Trilinos" );
#endif
    } else {
        AMP_ERROR( "Unknown ManagedMatrix type" );
    }

    // Initialize the matrix
    for ( size_t row = row_start; row < row_end; row++ ) {
        auto col = getRow( row );
        newMatrix->createValuesByGlobalID( row, col );
    }
    newMatrix->fillComplete();
    newMatrix->zero();
    newMatrix->makeConsistent();
    return newMatrix;
}


/********************************************************
 * Build a DenseSerialMatrix                             *
 ********************************************************/
AMP::LinearAlgebra::Matrix::shared_ptr
createDenseSerialMatrix( AMP::LinearAlgebra::Vector::shared_ptr leftVec,
                         AMP::LinearAlgebra::Vector::shared_ptr rightVec )
{
    // Get the DOFs
    auto leftDOF  = leftVec->getDOFManager();
    auto rightDOF = rightVec->getDOFManager();
    if ( leftDOF->getComm().compare( rightVec->getComm() ) == 0 )
        AMP_ERROR( "leftDOF and rightDOF on different comm groups is NOT tested, and needs to "
                   "be fixed" );
    AMP_MPI comm = leftDOF->getComm();
    if ( comm.getSize() == 1 )
        comm = AMP_MPI( AMP_COMM_SELF );
    else
        AMP_ERROR( "serial dense matrix does not support parallel matrices" );
    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>( leftDOF, rightDOF, comm );
    params->d_VariableLeft  = leftVec->getVariable();
    params->d_VariableRight = rightVec->getVariable();
    // Create the matrix
    auto newMatrix = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrix>( params );
    // Initialize the matrix
    newMatrix->zero();
    newMatrix->makeConsistent();
    return newMatrix;
}


/********************************************************
 * Test the matrix to ensure it is valid                 *
 ********************************************************/
static void test( AMP::LinearAlgebra::Matrix::shared_ptr matrix )
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
AMP::LinearAlgebra::Matrix::shared_ptr
createMatrix( AMP::LinearAlgebra::Vector::shared_ptr rightVec,
              AMP::LinearAlgebra::Vector::shared_ptr leftVec,
              const std::string &type,
              std::function<std::vector<size_t>( size_t )> getRow )
{
    // Determine the type of matrix to build
    std::string type2 = type;
    if ( type == "auto" ) {
#if defined( USE_EXT_TRILINOS )
        type2 = "ManagedEpetraMatrix";
#else
        type2 = "DenseSerialMatrix";
#endif
    }
    // Create the default getRow function (if not provided)
    if ( !getRow ) {
        const auto leftDOF  = leftVec->getDOFManager().get();
        const auto rightDOF = rightVec->getDOFManager().get();
        getRow              = [leftDOF, rightDOF]( size_t row ) {
            auto elem = leftDOF->getElement( row );
            return rightDOF->getRowDOFs( elem );
        };
    }
    // Build the matrix
    AMP::LinearAlgebra::Matrix::shared_ptr matrix;
    if ( type2 == "ManagedEpetraMatrix" ) {
        matrix = createManagedMatrix( leftVec, rightVec, getRow, type2 );
        test( matrix );
    } else if ( type2 == "DenseSerialMatrix" ) {
        matrix = createDenseSerialMatrix( leftVec, rightVec );
        test( matrix );
    } else {
        AMP_ERROR( "Unknown matrix type to build" );
    }
    return matrix;
}


/********************************************************
 * Create Matrix from PETSc Mat                          *
 ********************************************************/
#if defined( USE_EXT_PETSC )
std::shared_ptr<Matrix> createMatrix( Mat M, bool deleteable )
{
    auto matrix = std::make_shared<NativePetscMatrix>( M, deleteable );
    AMP_ASSERT( !matrix->getComm().isNull() );
    return matrix;
}
#endif


} // namespace AMP::LinearAlgebra

#endif
