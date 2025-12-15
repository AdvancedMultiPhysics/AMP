#include "AMP/matrices/data/DenseSerialMatrixData.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/vectors/VectorBuilder.h"
#include <cstdio>
#include <cstring>

#include <numeric>

namespace AMP::LinearAlgebra {


/********************************************************
 * Constructor/Destructor                                *
 ********************************************************/
DenseSerialMatrixData::DenseSerialMatrixData( std::shared_ptr<MatrixParametersBase> inparams )
    : MatrixData( inparams )
{
    auto params = std::dynamic_pointer_cast<MatrixParameters>( inparams );
    AMP_ASSERT( params );
    d_DOFManagerLeft  = params->getLeftDOFManager();
    d_DOFManagerRight = params->getRightDOFManager();
    d_rows            = params->getGlobalNumberOfRows();
    d_cols            = params->getGlobalNumberOfColumns();
    d_M               = new double[d_rows * d_cols];
    memset( d_M, 0, d_rows * d_cols * sizeof( double ) );
}

DenseSerialMatrixData::~DenseSerialMatrixData()
{
    if ( d_M ) {
        delete[] d_M;
        d_M = nullptr;
    }
}

std::shared_ptr<MatrixData> DenseSerialMatrixData::cloneMatrixData() const
{
    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
        d_DOFManagerLeft, d_DOFManagerRight, getComm(), getLeftVariable(), getRightVariable() );
    // Create the matrix
    auto newMatrixData = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrixData>( params );
    double *M2         = newMatrixData->d_M;
    memcpy( M2, d_M, d_cols * d_rows * sizeof( double ) );
    return newMatrixData;
}

std::shared_ptr<MatrixData> DenseSerialMatrixData::transpose() const
{
    // Create the matrix parameters
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
        d_DOFManagerRight, d_DOFManagerLeft, getComm(), getLeftVariable(), getRightVariable() );
    // Create the matrix
    auto newMatrixData = std::make_shared<AMP::LinearAlgebra::DenseSerialMatrixData>( params );

    auto *m2RawData = newMatrixData->d_M;

    const auto *m1RawData = this->d_M;
    const auto nrows      = this->d_rows;
    const auto ncols      = this->d_cols;

    for ( size_t i = 0; i < nrows; ++i ) {
        for ( size_t j = 0; j < ncols; ++j ) {
            m2RawData[j + i * ncols] = m1RawData[i + j * nrows];
        }
    }

    return newMatrixData;
}

/********************************************************
 * Get/Set values                                        *
 ********************************************************/
void DenseSerialMatrixData::addValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            d_M[rows[0] + cols[0] * d_rows] += values[0];
        } else {
            for ( size_t i = 0; i < num_rows; i++ ) {
                for ( size_t j = 0; j < num_cols; j++ ) {
                    d_M[rows[i] + cols[j] * d_rows] += values[num_cols * i + j];
                }
            }
        }
    } else if ( id == getTypeID<float>() ) {
        auto values = reinterpret_cast<const float *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            d_M[rows[0] + cols[0] * d_rows] += static_cast<double>( values[0] );
        } else {
            for ( size_t i = 0; i < num_rows; i++ ) {
                for ( size_t j = 0; j < num_cols; j++ ) {
                    d_M[rows[i] + cols[j] * d_rows] +=
                        static_cast<double>( values[num_cols * i + j] );
                }
            }
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}
void DenseSerialMatrixData::setValuesByGlobalID(
    size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, void *vals, const typeID &id )
{
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<const double *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            d_M[rows[0] + cols[0] * d_rows] = values[0];
        } else {
            for ( size_t i = 0; i < num_rows; i++ ) {
                for ( size_t j = 0; j < num_cols; j++ ) {
                    d_M[rows[i] + cols[j] * d_rows] = values[num_cols * i + j];
                }
            }
        }
    } else if ( id == getTypeID<float>() ) {
        auto values = reinterpret_cast<const float *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            d_M[rows[0] + cols[0] * d_rows] = static_cast<double>( values[0] );
        } else {
            for ( size_t i = 0; i < num_rows; i++ ) {
                for ( size_t j = 0; j < num_cols; j++ ) {
                    d_M[rows[i] + cols[j] * d_rows] =
                        static_cast<double>( values[num_cols * i + j] );
                }
            }
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}
void DenseSerialMatrixData::getValuesByGlobalID( size_t num_rows,
                                                 size_t num_cols,
                                                 size_t *rows,
                                                 size_t *cols,
                                                 void *vals,
                                                 const typeID &id ) const
{
    if ( id == getTypeID<double>() ) {
        auto values = reinterpret_cast<double *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            values[0] = d_M[rows[0] + cols[0] * d_rows];
        } else {
            for ( size_t i = 0; i < num_rows; i++ )
                for ( size_t j = 0; j < num_cols; j++ )
                    values[i * num_cols + j] = d_M[rows[i] + cols[j] * d_rows];
        }
    } else if ( id == getTypeID<float>() ) {
        auto values = reinterpret_cast<float *>( vals );
        if ( num_rows == 1 && num_cols == 1 ) {
            values[0] = static_cast<float>( d_M[rows[0] + cols[0] * d_rows] );
        } else {
            for ( size_t i = 0; i < num_rows; i++ )
                for ( size_t j = 0; j < num_cols; j++ )
                    values[i * num_cols + j] =
                        static_cast<float>( d_M[rows[i] + cols[j] * d_rows] );
        }
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

/********************************************************
 * Get values/rows by global id                          *
 ********************************************************/
void DenseSerialMatrixData::getRowByGlobalID( size_t row,
                                              std::vector<size_t> &cols,
                                              std::vector<double> &values ) const
{
    AMP_ASSERT( row < d_rows );
    cols.resize( d_cols );
    values.resize( d_cols );
    for ( size_t i = 0; i < d_cols; i++ ) {
        cols[i]   = i;
        values[i] = d_M[row + i * d_rows];
    }
}

/********************************************************
 * Get column indices by global id                       *
 ********************************************************/
std::vector<size_t> DenseSerialMatrixData::getColumnIDs( size_t row ) const
{
    AMP_ASSERT( row < d_rows );

    std::vector<size_t> cols( d_cols );
    std::iota( cols.begin(), cols.end(), 0 );

    return cols;
}

/********************************************************
 * Get the left/right DOFManagers                       *
 ********************************************************/

std::shared_ptr<Discretization::DOFManager> DenseSerialMatrixData::getRightDOFManager() const
{
    return d_DOFManagerRight;
}
std::shared_ptr<Discretization::DOFManager> DenseSerialMatrixData::getLeftDOFManager() const
{
    return d_DOFManagerLeft;
}

} // namespace AMP::LinearAlgebra
