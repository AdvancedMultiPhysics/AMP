#include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"
#include "AMP/matrices/trilinos/EpetraMatrixData.h"
#include "AMP/matrices/trilinos/EpetraMatrixOperations.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

#include "ProfilerApp.h"
#include <algorithm>

DISABLE_WARNINGS
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>
#include <Epetra_FECrsMatrix.h>
#ifdef AMP_USE_MPI
    #include <Epetra_MpiComm.h>
#else
    #include <Epetra_SerialComm.h>
#endif
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/********************************************************
 * Constructors                                          *
 ********************************************************/
ManagedEpetraMatrix::ManagedEpetraMatrix( std::shared_ptr<MatrixParameters> params )
{
    d_matrixData = std::make_shared<EpetraMatrixData>( params );
    d_matrixOps  = std::make_shared<EpetraMatrixOperations>();
}

ManagedEpetraMatrix::ManagedEpetraMatrix( const ManagedEpetraMatrix &rhs ) : Matrix( rhs )
{
    const auto rhsData = std::dynamic_pointer_cast<const EpetraMatrixData>( rhs.getMatrixData() );
    AMP_ASSERT( rhsData );
    d_matrixData = std::make_shared<EpetraMatrixData>( *rhsData );
    d_matrixOps  = std::make_shared<EpetraMatrixOperations>();
}

ManagedEpetraMatrix::ManagedEpetraMatrix( Epetra_CrsMatrix *m, bool dele )
{
    d_matrixData = std::make_shared<EpetraMatrixData>( m, dele );
    d_matrixOps  = std::make_shared<EpetraMatrixOperations>();
}

ManagedEpetraMatrix::ManagedEpetraMatrix( std::shared_ptr<MatrixData> data )
{
    d_matrixData = data;
    d_matrixOps  = std::make_shared<EpetraMatrixOperations>();
}

std::shared_ptr<Matrix> ManagedEpetraMatrix::clone() const
{
    return std::make_shared<ManagedEpetraMatrix>( d_matrixData->cloneMatrixData() );
}

Epetra_CrsMatrix &ManagedEpetraMatrix::getEpetra_CrsMatrix()
{
    auto data = std::dynamic_pointer_cast<EpetraMatrixData>( d_matrixData );
    AMP_ASSERT( data );
    return data->getEpetra_CrsMatrix();
}

const Epetra_CrsMatrix &ManagedEpetraMatrix::getEpetra_CrsMatrix() const
{
    const auto data = std::dynamic_pointer_cast<const EpetraMatrixData>( d_matrixData );
    AMP_ASSERT( data );
    return data->getEpetra_CrsMatrix();
}

std::shared_ptr<Matrix> ManagedEpetraMatrix::transpose() const
{
    return std::make_shared<ManagedEpetraMatrix>( d_matrixData->transpose() );
}

/********************************************************
 * Get the left/right Vector/DOFManager                  *
 ********************************************************/
std::shared_ptr<Vector> ManagedEpetraMatrix::getInputVector() const
{
    const auto data = std::dynamic_pointer_cast<const EpetraMatrixData>( d_matrixData );
    AMP_ASSERT( data );
    return data->getInputVector();
}
std::shared_ptr<Vector> ManagedEpetraMatrix::getOutputVector() const
{
    const auto data = std::dynamic_pointer_cast<const EpetraMatrixData>( d_matrixData );
    AMP_ASSERT( data );
    return data->getOutputVector();
}


std::shared_ptr<Vector> ManagedEpetraMatrix::extractDiagonal( std::shared_ptr<Vector> vec ) const
{
    if ( !vec )
        vec = getInputVector();
    d_matrixOps->extractDiagonal( *d_matrixData, vec );
    return vec;
}

void ManagedEpetraMatrix::multiply( shared_ptr other_op, std::shared_ptr<Matrix> &result )
{
    if ( this->numGlobalColumns() != other_op->numGlobalRows() )
        AMP_ERROR( "Inner matrix dimensions must agree" );
    if ( !std::dynamic_pointer_cast<ManagedEpetraMatrix>( other_op ) )
        AMP_ERROR( "Incompatible matrix types" );
    AMP_ASSERT( other_op->numGlobalRows() == numGlobalColumns() );
#ifdef AMP_USE_MPI
    AMP_MPI::Comm epetraComm =
        ( dynamic_cast<const Epetra_MpiComm *>( &getEpetra_CrsMatrix().RowMap().Comm() ) )->Comm();
#else
    AMP_MPI::Comm epetraComm = AMP_COMM_SELF;
#endif
    auto leftVec  = this->getOutputVector();
    auto rightVec = other_op->getInputVector();

    auto memp = std::make_shared<MatrixParameters>( leftVec->getDOFManager(),
                                                    rightVec->getDOFManager(),
                                                    AMP_MPI( epetraComm ),
                                                    d_matrixData->getLeftVariable(),
                                                    other_op->getMatrixData()->getRightVariable(),
                                                    leftVec->getCommunicationList(),
                                                    rightVec->getCommunicationList() );

    std::shared_ptr<Matrix> newMatrix = std::make_shared<ManagedEpetraMatrix>( memp );
    result.swap( newMatrix );
    PROFILE( "Epetra::MatrixMultiply" );
    d_matrixOps->matMatMult( d_matrixData, other_op->getMatrixData(), result->getMatrixData() );
}

} // namespace AMP::LinearAlgebra
