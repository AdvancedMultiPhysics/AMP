#include "AMP/matrices/operations/DenseSerialMatrixOperations.h"
#include "AMP/matrices/data/DenseSerialMatrixData.h"

namespace AMP::LinearAlgebra {


static DenseSerialMatrixData const *getDenseSerialMatrixData( MatrixData const &A )
{
    auto ptr = dynamic_cast<DenseSerialMatrixData const *>( &A );
    AMP_INSIST( ptr, "dynamic cast from const MatrixData to const DenseSerialMatrixData failed" );
    return ptr;
}

static DenseSerialMatrixData *getDenseSerialMatrixData( MatrixData &A )
{
    auto ptr = dynamic_cast<DenseSerialMatrixData *>( &A );
    AMP_INSIST( ptr, "dynamic cast from const MatrixData to const DenseSerialMatrixData failed" );
    return ptr;
}

void DenseSerialMatrixOperations::mult( std::shared_ptr<const Vector> in,
                                        MatrixData const &A,
                                        std::shared_ptr<Vector> out )
{
    auto dense = getDenseSerialMatrixData( A );
    auto nrows = dense->numLocalRows();
    auto ncols = dense->numLocalColumns();
    auto data  = dense->getM();

    AMP_ASSERT( in->getGlobalSize() == ncols );
    AMP_ASSERT( out->getGlobalSize() == nrows );
    auto k = new size_t[std::max( ncols, nrows )];
    for ( size_t i = 0; i < std::max( ncols, nrows ); i++ )
        k[i] = i;
    // Get x
    auto x = new double[ncols];
    in->getValuesByGlobalID( ncols, k, x );
    // Initialize y
    auto y = new double[nrows];
    memset( y, 0, nrows * sizeof( double ) );
    // Perform y = M*x
    for ( size_t j = 0, k = 0; j < ncols; j++ ) {
        for ( size_t i = 0; i < nrows; i++, k++ )
            y[i] += data[k] * x[j];
    }
    // Save y
    out->setValuesByGlobalID( nrows, k, y );
    delete[] x;
    delete[] y;
    delete[] k;
}

void DenseSerialMatrixOperations::multTranspose( std::shared_ptr<const Vector> in,
                                                 MatrixData const &A,
                                                 std::shared_ptr<Vector> out )
{
    auto dense = getDenseSerialMatrixData( A );
    auto nrows = dense->numLocalRows();
    auto ncols = dense->numLocalColumns();
    auto data  = dense->getM();

    AMP_ASSERT( in->getGlobalSize() == nrows );
    AMP_ASSERT( out->getGlobalSize() == ncols );
    auto k = new size_t[std::max( ncols, nrows )];
    for ( size_t i = 0; i < std::max( ncols, nrows ); i++ )
        k[i] = i;
    // Get x
    auto x = new double[nrows];
    in->getValuesByGlobalID( nrows, k, x );
    // Initialize y
    auto y = new double[ncols];
    memset( y, 0, ncols * sizeof( double ) );
    // Perform y = M*x
    for ( size_t j = 0, k = 0; j < ncols; j++ ) {
        for ( size_t i = 0; i < nrows; i++, k++ )
            y[j] += data[k] * x[i];
    }
    // Save y
    out->setValuesByGlobalID( ncols, k, y );
    delete[] x;
    delete[] y;
    delete[] k;
}

void DenseSerialMatrixOperations::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    auto alpha = static_cast<double>( alpha_in );
    auto dense = getDenseSerialMatrixData( A );
    auto data  = dense->getM();
    size_t N   = dense->size();
    for ( size_t i = 0; i < N; i++ )
        data[i] *= alpha;
}

void DenseSerialMatrixOperations::matMatMult( std::shared_ptr<MatrixData> Am,
                                              std::shared_ptr<MatrixData> Bm,
                                              std::shared_ptr<MatrixData> Cm )
{
    auto Amat = getDenseSerialMatrixData( *Am );
    auto Bmat = getDenseSerialMatrixData( *Bm );
    auto Cmat = getDenseSerialMatrixData( *Cm );

    size_t N = Amat->numGlobalRows();
    size_t K = Amat->numGlobalColumns();
    AMP_ASSERT( K == Bmat->numGlobalRows() );
    size_t M = Bmat->numGlobalColumns();

    auto A = Amat->getM();
    auto B = Bmat->getM();
    auto C = Cmat->getM();
    for ( size_t m = 0; m < M; m++ ) {
        for ( size_t k = 0; k < K; k++ ) {
            double b = B[k + m * K];
            for ( size_t n = 0, ic = m*N, iA = k*N; n < N; n++, ic++, iA++ ) {
                C[ic] += A[iA] * b;
            }
        }
    }
}

void DenseSerialMatrixOperations::axpy( AMP::Scalar alpha_in, const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );
    auto alpha = static_cast<double>( alpha_in );
    auto A     = dynamic_cast<const DenseSerialMatrixData *>( &X );
    auto B     = getDenseSerialMatrixData( Y );
    auto data  = B->getM();
    if ( A ) {
        // We are dealing with two DenseSerialMatrix classes
        auto *data2 = A->getM();
        size_t N    = A->size();
        for ( size_t i = 0; i < N; i++ ) {
            data[i] += alpha * data2[i];
        }
    } else {
        // X is an unknown matrix type
        std::vector<size_t> cols;
        std::vector<double> values;
        auto nrows = B->numLocalRows();
        for ( size_t i = 0; i < nrows; i++ ) {
            X.getRowByGlobalID( static_cast<int>( i ), cols, values );
            for ( size_t j = 0; j < cols.size(); j++ )
                data[i + cols[j] * nrows] += alpha * values[j];
        }
    }
}

void DenseSerialMatrixOperations::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    auto alpha = static_cast<double>( alpha_in );
    auto dense = getDenseSerialMatrixData( A );
    auto data  = dense->getM();
    size_t N   = dense->size();
    for ( size_t i = 0; i < N; i++ )
        data[i] = alpha;
}

void DenseSerialMatrixOperations::zero( MatrixData &A )
{
    auto dense = getDenseSerialMatrixData( A );
    auto data  = dense->getM();
    memset( data, 0, dense->size() * sizeof( double ) );
}

void DenseSerialMatrixOperations::setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A )
{
    auto dense = getDenseSerialMatrixData( A );
    auto nrows = dense->numLocalRows();
    auto ncols = dense->numLocalColumns();
    auto data  = dense->getM();
    AMP_ASSERT( ncols == nrows );
    AMP_ASSERT( in->getGlobalSize() == nrows );
    auto k = new size_t[nrows];
    for ( size_t i = 0; i < nrows; i++ )
        k[i] = i;
    auto x = new double[nrows];
    in->getValuesByGlobalID( nrows, k, x );
    for ( size_t i = 0; i < nrows; i++ )
        data[i + i * nrows] = x[i];
    delete[] x;
    delete[] k;
}

void DenseSerialMatrixOperations::setIdentity( MatrixData &A )
{
    auto dense = getDenseSerialMatrixData( A );
    auto nrows = dense->numLocalRows();
    auto ncols = dense->numLocalColumns();
    auto data  = dense->getM();
    AMP_ASSERT( ncols == nrows );
    memset( data, 0, nrows * ncols * sizeof( double ) );
    for ( size_t i = 0; i < nrows; i++ )
        data[i + i * nrows] = 1.0;
}

void DenseSerialMatrixOperations::extractDiagonal( MatrixData const &A,
                                                   std::shared_ptr<Vector> buf )
{
    auto dense = getDenseSerialMatrixData( A );
    auto nrows = dense->numLocalRows();
    auto ncols = dense->numLocalColumns();
    auto data  = dense->getM();
    AMP_ASSERT( ncols == nrows );
    auto *rawVecData = buf->getRawDataBlock<double>();
    for ( size_t i = 0; i < nrows; i++ )
        rawVecData[i] = data[i + i * nrows];
}

AMP::Scalar DenseSerialMatrixOperations::LinfNorm( MatrixData const &A ) const
{
    auto dense  = getDenseSerialMatrixData( A );
    auto nrows  = dense->numLocalRows();
    auto ncols  = dense->numLocalColumns();
    auto data   = dense->getM();
    double norm = 0.0;
    for ( size_t i = 0; i < nrows; i++ ) {
        double sum = 0.0;
        for ( size_t j = 0, k = i; j < ncols; j++, k += nrows )
            sum += fabs( data[k] );
        norm = std::max( norm, sum );
    }
    return norm;
}

void DenseSerialMatrixOperations::copy( const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );
    auto A    = dynamic_cast<const DenseSerialMatrixData *>( &X );
    auto B    = getDenseSerialMatrixData( Y );
    auto data = B->getM();
    AMP_ASSERT( X.numGlobalRows() == B->numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == B->numGlobalColumns() );
    if ( A ) {
        // We are dealing with two DenseSerialMatrix classes
        auto *data2 = A->getM();
        memcpy( data, data2, A->size() * sizeof( double ) );
    } else {
        // X is an unknown matrix type
        std::vector<size_t> cols;
        std::vector<double> values;
        auto nrows = B->numLocalRows();
        for ( size_t i = 0; i < nrows; i++ ) {
            X.getRowByGlobalID( static_cast<int>( i ), cols, values );
            for ( size_t j = 0; j < cols.size(); j++ )
                data[i + cols[j] * nrows] = values[j];
        }
    }
}

void DenseSerialMatrixOperations::scale( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
void DenseSerialMatrixOperations::scaleInv( AMP::Scalar,
                                            std::shared_ptr<const Vector>,
                                            MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}


/****************************************************************
 * Get row sums                                                  *
 ****************************************************************/
void DenseSerialMatrixOperations::getRowSums( MatrixData const &A, std::shared_ptr<Vector> sum )
{
    AMP_ASSERT( sum );
    auto B    = getDenseSerialMatrixData( A );
    auto Nr   = B->numLocalRows();
    auto Nc   = B->numLocalColumns();
    auto data = B->getM();
    for ( size_t i = 0; i < Nr; i++ ) {
        double s = 0.0;
        for ( size_t j = 0, k = i; j < Nc; j++, k += Nr )
            s += data[k];
        sum->setValueByGlobalID( i, s );
    }
}
void DenseSerialMatrixOperations::getRowSumsAbsolute( MatrixData const &A,
                                                      std::shared_ptr<Vector> sum,
                                                      const bool removeZeros )
{
    AMP_ASSERT( !removeZeros );
    AMP_ASSERT( sum );
    auto B    = getDenseSerialMatrixData( A );
    auto Nr   = B->numLocalRows();
    auto Nc   = B->numLocalColumns();
    auto data = B->getM();
    for ( size_t i = 0; i < Nr; i++ ) {
        double s = 0.0;
        for ( size_t j = 0, k = i; j < Nc; j++, k += Nr )
            s += fabs( data[k] );
        sum->setValueByGlobalID( i, s );
    }
}


} // namespace AMP::LinearAlgebra
