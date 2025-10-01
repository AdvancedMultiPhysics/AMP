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
    const auto m1Data     = getDenseSerialMatrixData( A );
    const auto nrows      = m1Data->d_rows;
    const auto ncols      = m1Data->d_cols;
    const auto *m1RawData = m1Data->d_M;

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
    for ( size_t j = 0; j < ncols; j++ ) {
        for ( size_t i = 0; i < nrows; i++ )
            y[i] += m1RawData[i + j * nrows] * x[j];
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
    const auto m1Data     = getDenseSerialMatrixData( A );
    const auto nrows      = m1Data->d_rows;
    const auto ncols      = m1Data->d_cols;
    const auto *m1RawData = m1Data->d_M;

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
    for ( size_t j = 0; j < ncols; j++ ) {
        for ( size_t i = 0; i < nrows; i++ )
            y[j] += m1RawData[i + j * nrows] * x[i];
    }
    // Save y
    out->setValuesByGlobalID( ncols, k, y );
    delete[] x;
    delete[] y;
    delete[] k;
}

void DenseSerialMatrixOperations::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    const auto alpha = static_cast<double>( alpha_in );
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    for ( size_t i = 0; i < nrows * ncols; i++ )
        m1RawData[i] *= alpha;
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

    auto *C = Cmat->d_M;
    // Perform the multiplication
    if ( dynamic_cast<DenseSerialMatrixData const *>( Bmat ) == nullptr ) {
        // X is an unknown matrix type
        AMP_ERROR( "Not programmed yet" );
    } else {
        // We are dealing with all DenseSerialMatrix classes
        const double *A = Amat->d_M;
        const double *B = Bmat->d_M;
        for ( size_t m = 0; m < M; m++ ) {
            for ( size_t k = 0; k < K; k++ ) {
                double b = B[k + m * K];
                for ( size_t n = 0; n < N; n++ ) {
                    C[n + m * N] += A[n + k * N] * b;
                }
            }
        }
    }
}

void DenseSerialMatrixOperations::axpy( AMP::Scalar alpha_in, const MatrixData &X, MatrixData &Y )
{
    const auto alpha = static_cast<double>( alpha_in );
    auto m1Data      = getDenseSerialMatrixData( Y );
    auto *m1RawData  = m1Data->d_M;
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;

    AMP_ASSERT( X.numGlobalRows() == m1Data->numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == m1Data->numGlobalColumns() );
    if ( dynamic_cast<DenseSerialMatrixData const *>( &X ) == nullptr ) {
        // X is an unknown matrix type
        std::vector<size_t> cols;
        std::vector<double> values;
        for ( size_t i = 0; i < nrows; i++ ) {
            X.getRowByGlobalID( static_cast<int>( i ), cols, values );
            for ( size_t j = 0; j < cols.size(); j++ )
                m1RawData[i + cols[j] * nrows] += alpha * values[j];
        }
    } else {
        // We are dealing with two DenseSerialMatrix classes
        auto m2Data = getDenseSerialMatrixData( X );
        AMP_ASSERT( m2Data );
        auto *m2RawData = m2Data->d_M;
        for ( size_t i = 0; i < nrows * ncols; i++ ) {
            m1RawData[i] += alpha * m2RawData[i];
        }
    }
}

void DenseSerialMatrixOperations::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    const auto alpha = static_cast<double>( alpha_in );
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    for ( size_t i = 0; i < nrows * ncols; i++ )
        m1RawData[i] = alpha;
}

void DenseSerialMatrixOperations::zero( MatrixData &A )
{
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    memset( m1RawData, 0, nrows * ncols * sizeof( double ) );
}

void DenseSerialMatrixOperations::setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A )
{
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    AMP_ASSERT( ncols == nrows );
    AMP_ASSERT( in->getGlobalSize() == nrows );
    auto k = new size_t[nrows];
    for ( size_t i = 0; i < nrows; i++ )
        k[i] = i;
    auto x = new double[nrows];
    in->getValuesByGlobalID( nrows, k, x );
    for ( size_t i = 0; i < nrows; i++ )
        m1RawData[i + i * nrows] = x[i];
    delete[] x;
    delete[] k;
}

void DenseSerialMatrixOperations::setIdentity( MatrixData &A )
{
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    AMP_ASSERT( ncols == nrows );
    memset( m1RawData, 0, nrows * ncols * sizeof( double ) );
    for ( size_t i = 0; i < nrows; i++ )
        m1RawData[i + i * nrows] = 1.0;
}

void DenseSerialMatrixOperations::extractDiagonal( MatrixData const &A,
                                                   std::shared_ptr<Vector> buf )
{
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    auto *m1RawData  = m1Data->d_M;
    auto *rawVecData = buf->getRawDataBlock<double>();

    for ( size_t i = 0; i < nrows; i++ )
        rawVecData[i] = m1RawData[i + i * nrows];
}

AMP::Scalar DenseSerialMatrixOperations::LinfNorm( MatrixData const &A ) const
{
    auto m1Data      = getDenseSerialMatrixData( A );
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;
    auto *m1RawData  = m1Data->d_M;

    double norm = 0.0;
    for ( size_t i = 0; i < nrows; i++ ) {
        double sum = 0.0;
        for ( size_t j = 0; j < ncols; j++ ) {
            sum += fabs( m1RawData[i + j * nrows] );
        }
        norm = std::max( norm, sum );
    }
    return norm;
}

void DenseSerialMatrixOperations::copy( const MatrixData &X, MatrixData &Y )
{
    auto m1Data      = getDenseSerialMatrixData( Y );
    auto *m1RawData  = m1Data->d_M;
    const auto nrows = m1Data->d_rows;
    const auto ncols = m1Data->d_cols;

    AMP_ASSERT( X.numGlobalRows() == m1Data->numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == m1Data->numGlobalColumns() );
    if ( X.type() != "DenseSerialMatrixData" ) {
        // X is an unknown matrix type
        std::vector<size_t> cols;
        std::vector<double> values;
        for ( size_t i = 0; i < nrows; i++ ) {
            X.getRowByGlobalID( static_cast<int>( i ), cols, values );
            for ( size_t j = 0; j < cols.size(); j++ )
                m1RawData[i + cols[j] * nrows] = values[j];
        }
    } else {
        // We are dealing with two DenseSerialMatrix classes
        auto m2Data = getDenseSerialMatrixData( X );
        AMP_ASSERT( m2Data );
        auto *m2RawData = m2Data->d_M;
        memcpy( m1RawData, m2RawData, ncols * nrows * sizeof( double ) );
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
void DenseSerialMatrixOperations::getRowSums( MatrixData const &, std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}
void DenseSerialMatrixOperations::getRowSumsAbsolute( MatrixData const &,
                                                      std::shared_ptr<Vector>,
                                                      const bool )
{
    AMP_ERROR( "Not implemented" );
}


} // namespace AMP::LinearAlgebra
