#include "AMP/matrices/DenseSerialMatrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/DenseSerialMatrixData.h"
#include "AMP/matrices/operations/DenseSerialMatrixOperations.h"
#include "AMP/vectors/VectorBuilder.h"
#include <cstdio>
#include <cstring>

#include <numeric>

namespace AMP::LinearAlgebra {


/********************************************************
 * Constructor/Destructor                                *
 ********************************************************/
DenseSerialMatrix::DenseSerialMatrix( std::shared_ptr<MatrixParametersBase> params )
    : Matrix( params )
{
    d_matrixOps = std::make_shared<DenseSerialMatrixOperations>();
}

DenseSerialMatrix::DenseSerialMatrix( std::shared_ptr<MatrixData> data ) : Matrix( data )
{
    d_matrixOps = std::make_shared<DenseSerialMatrixOperations>();
}

DenseSerialMatrix::~DenseSerialMatrix() {}

/********************************************************
 * Copy/transpose the matrix                             *
 ********************************************************/
std::shared_ptr<Matrix> DenseSerialMatrix::clone() const
{
    return std::make_shared<DenseSerialMatrix>( d_matrixData->cloneMatrixData() );
}

std::shared_ptr<Matrix> DenseSerialMatrix::transpose() const
{
    auto data = d_matrixData->transpose();
    return std::make_shared<DenseSerialMatrix>( data );
}

/********************************************************
 * Multiply two matricies                                *
 * result = this * other_op                              *
 * C(N,M) = A(N,K)*B(K,M)
 ********************************************************/
void DenseSerialMatrix::multiply( std::shared_ptr<Matrix> other_op,
                                  std::shared_ptr<Matrix> &result )
{
    // pull out matrix data objects and ensure they are of correct type
    auto thisData =
        std::dynamic_pointer_cast<AMP::LinearAlgebra::DenseSerialMatrixData>( d_matrixData );
    auto otherData = std::dynamic_pointer_cast<AMP::LinearAlgebra::DenseSerialMatrixData>(
        other_op->getMatrixData() );
    AMP_DEBUG_INSIST( thisData && otherData,
                      "DenseSerialMatrix::multiply received invalid MatrixData types" );

    // Build matrix parameters object for result from this op and the other op
    auto params =
        std::make_shared<AMP::LinearAlgebra::MatrixParameters>( getLeftDOFManager(),
                                                                other_op->getRightDOFManager(),
                                                                getComm(),
                                                                thisData->getLeftVariable(),
                                                                otherData->getRightVariable() );

    // Create the matrix
    auto newData = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrixData>( params );
    std::shared_ptr<Matrix> newMatrix =
        std::make_shared<AMP::LinearAlgebra::DenseSerialMatrix>( newData );
    AMP_ASSERT( newMatrix );
    result.swap( newMatrix );

    d_matrixOps->matMatMult( getMatrixData(), other_op->getMatrixData(), newData );
}


/********************************************************
 * Get/Set the diagonal                                  *
 ********************************************************/
Vector::shared_ptr DenseSerialMatrix::extractDiagonal( Vector::shared_ptr buf ) const
{
    Vector::shared_ptr out = buf;
    if ( !buf )
        out = this->getInputVector();

    d_matrixOps->extractDiagonal( *getMatrixData(), out );

    return out;
}

/********************************************************
 * Get the left/right vectors and DOFManagers            *
 ********************************************************/
Vector::shared_ptr DenseSerialMatrix::getInputVector() const
{
    auto var = std::dynamic_pointer_cast<DenseSerialMatrixData>( d_matrixData )->getRightVariable();
    return createVector( getRightDOFManager(), var );
}
Vector::shared_ptr DenseSerialMatrix::getOutputVector() const
{
    auto var = std::dynamic_pointer_cast<DenseSerialMatrixData>( d_matrixData )->getLeftVariable();
    return createVector( getLeftDOFManager(), var );
}


} // namespace AMP::LinearAlgebra
