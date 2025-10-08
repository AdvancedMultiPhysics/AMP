#include "AMP/matrices/trilinos/tpetra/ManagedTpetraMatrix.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixOperations.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"

#include "ProfilerApp.h"
#include <algorithm>


namespace AMP::LinearAlgebra {


/********************************************************
 * Constructors                                          *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
ManagedTpetraMatrix<ST, LO, GO, NT>::ManagedTpetraMatrix( std::shared_ptr<MatrixParameters> params )
{
    d_matrixData = std::make_shared<TpetraMatrixData<ST, LO, GO, NT>>( params );
    d_matrixOps  = std::make_shared<TpetraMatrixOperations<ST, LO, GO, NT>>();
}

template<typename ST, typename LO, typename GO, typename NT>
ManagedTpetraMatrix<ST, LO, GO, NT>::ManagedTpetraMatrix(
    const ManagedTpetraMatrix<ST, LO, GO, NT> &rhs )
    : Matrix( rhs )
{
    const auto rhsData =
        std::dynamic_pointer_cast<const TpetraMatrixData<ST, LO, GO, NT>>( rhs.getMatrixData() );
    AMP_ASSERT( rhsData );
    d_matrixData = std::make_shared<TpetraMatrixData<ST, LO, GO, NT>>( *rhsData );
    d_matrixOps  = std::make_shared<TpetraMatrixOperations<ST, LO, GO, NT>>();
}

template<typename ST, typename LO, typename GO, typename NT>
ManagedTpetraMatrix<ST, LO, GO, NT>::ManagedTpetraMatrix(
    Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> m )
{
    d_matrixData = std::make_shared<TpetraMatrixData<ST, LO, GO, NT>>( m );
    d_matrixOps  = std::make_shared<TpetraMatrixOperations<ST, LO, GO, NT>>();
}

template<typename ST, typename LO, typename GO, typename NT>
ManagedTpetraMatrix<ST, LO, GO, NT>::ManagedTpetraMatrix( std::shared_ptr<MatrixData> data )
{
    d_matrixData = data;
    d_matrixOps  = std::make_shared<TpetraMatrixOperations<ST, LO, GO, NT>>();
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Matrix> ManagedTpetraMatrix<ST, LO, GO, NT>::clone() const
{
    return std::make_shared<ManagedTpetraMatrix<ST, LO, GO, NT>>( d_matrixData->cloneMatrixData() );
}

template<typename ST, typename LO, typename GO, typename NT>
Tpetra::CrsMatrix<ST, LO, GO, NT> &ManagedTpetraMatrix<ST, LO, GO, NT>::getTpetra_CrsMatrix()
{
    auto data = std::dynamic_pointer_cast<TpetraMatrixData<ST, LO, GO, NT>>( d_matrixData );
    AMP_ASSERT( data );
    return data->getTpetra_CrsMatrix();
}

template<typename ST, typename LO, typename GO, typename NT>
const Tpetra::CrsMatrix<ST, LO, GO, NT> &
ManagedTpetraMatrix<ST, LO, GO, NT>::getTpetra_CrsMatrix() const
{
    const auto data =
        std::dynamic_pointer_cast<const TpetraMatrixData<ST, LO, GO, NT>>( d_matrixData );
    AMP_ASSERT( data );
    return data->getTpetra_CrsMatrix();
}

template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Matrix> ManagedTpetraMatrix<ST, LO, GO, NT>::transpose() const
{
    return std::make_shared<ManagedTpetraMatrix<ST, LO, GO, NT>>( d_matrixData->transpose() );
}

/********************************************************
 * Get the left/right Vector/DOFManager                  *
 ********************************************************/
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector> ManagedTpetraMatrix<ST, LO, GO, NT>::createInputVector() const
{
    const auto data =
        std::dynamic_pointer_cast<const TpetraMatrixData<ST, LO, GO, NT>>( d_matrixData );
    AMP_ASSERT( data );
    return data->createInputVector();
}
template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector> ManagedTpetraMatrix<ST, LO, GO, NT>::createOutputVector() const
{
    const auto data =
        std::dynamic_pointer_cast<const TpetraMatrixData<ST, LO, GO, NT>>( d_matrixData );
    AMP_ASSERT( data );
    return data->createOutputVector();
}


template<typename ST, typename LO, typename GO, typename NT>
std::shared_ptr<Vector>
ManagedTpetraMatrix<ST, LO, GO, NT>::extractDiagonal( std::shared_ptr<Vector> vec ) const
{
    if ( !vec )
        vec = createInputVector();
    d_matrixOps->extractDiagonal( *d_matrixData, vec );
    return vec;
}

template<typename ST, typename LO, typename GO, typename NT>
void ManagedTpetraMatrix<ST, LO, GO, NT>::multiply( shared_ptr other_op,
                                                    std::shared_ptr<Matrix> &result )
{
    if ( this->numGlobalColumns() != other_op->numGlobalRows() )
        AMP_ERROR( "Inner matrix dimensions must agree" );
    if ( !std::dynamic_pointer_cast<ManagedTpetraMatrix<ST, LO, GO, NT>>( other_op ) )
        AMP_ERROR( "Incompatible matrix types" );
    AMP_ASSERT( other_op->numGlobalRows() == numGlobalColumns() );
    auto leftVec  = this->createOutputVector();
    auto rightVec = other_op->createInputVector();

    auto memp = std::make_shared<MatrixParameters>( leftVec->getDOFManager(),
                                                    rightVec->getDOFManager(),
                                                    leftVec->getComm(),
                                                    d_matrixData->getLeftVariable(),
                                                    other_op->getMatrixData()->getRightVariable(),
                                                    leftVec->getCommunicationList(),
                                                    rightVec->getCommunicationList() );

    std::shared_ptr<Matrix> newMatrix = std::make_shared<ManagedTpetraMatrix>( memp );
    result.swap( newMatrix );
    PROFILE( "Tpetra::MatrixMultiply" );
    d_matrixOps->matMatMult( d_matrixData, other_op->getMatrixData(), result->getMatrixData() );
}

} // namespace AMP::LinearAlgebra
