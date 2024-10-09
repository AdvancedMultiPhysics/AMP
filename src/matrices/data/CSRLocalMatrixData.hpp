#ifndef included_AMP_CSRLocalMatrixData_hpp
#define included_AMP_CSRLocalMatrixData_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/CSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_UMPIRE
    #include "umpire/Allocator.hpp"
    #include "umpire/ResourceManager.hpp"
#endif

#ifdef USE_DEVICE
    #include "AMP/matrices/data/DeviceDataHelpers.h"
#endif

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <type_traits>

namespace AMP::LinearAlgebra {

template<typename Policy>
bool isColValid( typename Policy::gidx_t col,
                 bool is_diag,
                 typename Policy::gidx_t first_col,
                 typename Policy::gidx_t last_col )
{
    bool dValid  = is_diag && ( first_col <= col && col < last_col );
    bool odValid = !is_diag && ( col < first_col || last_col <= col );
    return ( dValid || odValid );
}

template<typename size_type, class data_allocator>
std::shared_ptr<typename data_allocator::value_type[]> sharedArrayBuilder( size_type N,
                                                                           data_allocator &alloc )
{
    AMP_DEBUG_ASSERT( std::is_integral_v<size_type> );
    return std::shared_ptr<typename data_allocator::value_type[]>(
        alloc.allocate( N ), [N, &alloc]( auto p ) -> void { alloc.deallocate( p, N ); } );
}

template<typename data_type>
std::shared_ptr<data_type[]> sharedArrayWrapper( data_type *raw_array )
{
    return std::shared_ptr<data_type[]>( raw_array, []( auto p ) -> void { (void) p; } );
}

template<typename Policy, class Allocator>
CSRLocalMatrixData<Policy, Allocator>::CSRLocalMatrixData(
    std::shared_ptr<MatrixParametersBase> params,
    AMP::Utilities::MemoryType memory_location,
    typename Policy::gidx_t first_row,
    typename Policy::gidx_t last_row,
    typename Policy::gidx_t first_col,
    typename Policy::gidx_t last_col,
    bool is_diag )
    : d_memory_location( memory_location ),
      d_first_row( first_row ),
      d_last_row( last_row ),
      d_first_col( first_col ),
      d_last_col( last_col ),
      d_is_diag( is_diag ),
      d_num_rows( last_row - first_row )
{
    AMPManager::incrementResource( "CSRLocalMatrixData" );
    d_pParameters  = params;
    auto csrParams = std::dynamic_pointer_cast<CSRMatrixParameters<Policy>>( d_pParameters );
    auto matParams = std ::dynamic_pointer_cast<MatrixParameters>( d_pParameters );

    if ( csrParams ) {
        // Pull out block specific parameters
        auto &blParams = d_is_diag ? csrParams->d_diag : csrParams->d_off_diag;

        // count nnz and decide if block is empty
        if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
            d_nnz =
                std::accumulate( blParams.d_nnz_per_row, blParams.d_nnz_per_row + d_num_rows, 0 );
        } else {
#ifdef USE_DEVICE
            d_nnz = AMP::LinearAlgebra::DeviceDataHelpers<lidx_t>::accumulate(
                blParams.d_nnz_per_row, d_num_rows, 0 );
#else
            AMP_ERROR( "Invalid memory type" );
#endif
        }
        if ( d_nnz == 0 ) {
            d_is_empty = true;
            return;
        }
        d_is_empty = false;

        // Wrap raw pointers from blParams to match internal
        // shared_ptr<T[]> type
        d_nnz_per_row = sharedArrayWrapper( blParams.d_nnz_per_row );
        d_row_starts  = sharedArrayWrapper( blParams.d_row_starts );
        d_cols        = sharedArrayWrapper( blParams.d_cols );
        d_cols_loc    = sharedArrayWrapper( blParams.d_cols_loc );
        d_coeffs      = sharedArrayWrapper( blParams.d_coeffs );
        AMP_ERROR( "Don't forget to add d_cols_unq here too" );
    } else if ( matParams ) {
        // Getting device memory support in this constructor mode will be very challenging
        AMP_ASSERT( d_memory_location != AMP::Utilities::MemoryType::device );

        const auto &getRow = matParams->getRowFunction();
        AMP_INSIST( getRow,
                    "Explicitly defined getRow function must be present in MatrixParameters"
                    " to construct CSRMatrixData and CSRLocalMatrixData" );

        // Count number of nonzeros depending on block type
        std::unordered_map<gidx_t, lidx_t> colMap;
        d_nnz       = 0;
        d_ncols_unq = 0;
        for ( gidx_t i = d_first_row; i < d_last_row; ++i ) {
            for ( auto &&col : getRow( i ) ) {
                if ( isColValid<Policy>( col, d_is_diag, d_first_col, d_last_col ) ) {
                    ++d_nnz;
                    // if offd then count unique columns and form map to local indices
                    if ( !d_is_diag ) {
                        const bool inserted = colMap.insert( { col, d_ncols_unq } ).second;
                        if ( inserted ) {
                            ++d_ncols_unq;
                        }
                    }
                }
            }
        }

        // bail out for degenerate case with no nnz
        // may happen in off-diagonal blocks
        if ( d_nnz == 0 ) {
            d_is_empty = true;
            return;
        }

        // Otherwise nonempty. Continue on.
        d_is_empty = false;

        // Allocate internal arrays
        d_nnz_per_row = sharedArrayBuilder( d_num_rows, lidxAllocator );
        d_row_starts  = sharedArrayBuilder( d_num_rows + 1, lidxAllocator );
        d_cols        = sharedArrayBuilder( d_nnz, gidxAllocator );
        d_cols_loc    = sharedArrayBuilder( d_nnz, lidxAllocator );
        d_coeffs      = sharedArrayBuilder( d_nnz, scalarAllocator );
        if ( !d_is_diag ) {
            d_cols_unq = sharedArrayBuilder( d_ncols_unq, gidxAllocator );
        }

        // Fill cols and nnz based on local row extents and on/off diag status
        lidx_t cli       = 0; // index into local array of columns as it is filled in
        lidx_t nnzFilled = 0;
        d_max_row_len    = 0;
        for ( lidx_t i = 0; i < d_num_rows; ++i ) {
            d_nnz_per_row[i] = 0;
            auto cols        = getRow( d_first_row + i );
            for ( auto &&col : cols ) {
                if ( isColValid<Policy>( col, d_is_diag, d_first_col, d_last_col ) ) {
                    d_nnz_per_row[i]++;
                    d_cols[cli] = col;
                    if ( d_is_diag ) {
                        d_cols_loc[cli] = static_cast<lidx_t>( col - d_first_col );
                    } else {
                        lidx_t cl       = colMap[col];
                        d_cols_loc[cli] = cl;
                        d_cols_unq[cl]  = col;
                    }
                    d_coeffs[cli] = 0.0;
                    ++cli;
                    ++nnzFilled;
                }
            }
            d_max_row_len = std::max( d_max_row_len, d_nnz_per_row[i] );
        }

        // scan nnz counts to get starting index of each row
        if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
            std::exclusive_scan(
                d_nnz_per_row.get(), d_nnz_per_row.get() + d_num_rows, d_row_starts.get(), 0 );
        } else {
#ifdef USE_DEVICE
            AMP::LinearAlgebra::DeviceDataHelpers<lidx_t>::exclusive_scan(
                d_nnz_per_row.get(), d_num_rows, d_row_starts.get(), 0 );
#else
            AMP_ERROR( "Invalid memory type" );
#endif
        }
        d_row_starts[d_num_rows] = d_row_starts[d_num_rows - 1] + d_nnz_per_row[d_num_rows - 1];

        // Ensure that the right number of nnz were actually filled in
        AMP_DEBUG_ASSERT( nnzFilled == d_nnz );
    } else {
        AMP_ERROR( "Check supplied MatrixParameter object" );
    }
}

template<typename Policy, class Allocator>
void CSRLocalMatrixData<Policy, Allocator>::sortColumnsHypre()
{
    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRSerialMatrixData::sortColumns not implemented for device memory" );

    // only used for Hypre compatibility
    std::vector<std::tuple<lidx_t, gidx_t, scalar_t>> rTpl( d_max_row_len );
    for ( lidx_t row = 0; row < d_num_rows; ++row ) {
        const auto rs      = d_row_starts[row];
        const auto row_len = d_row_starts[row + 1] - rs;

        for ( lidx_t k = 0; k < row_len; ++k ) {
            rTpl[k] = std::make_tuple( d_cols_loc[rs + k], d_cols[rs + k], d_coeffs[rs + k] );
        }

        if ( d_is_diag ) {
            // special "<" predicate for diagonal block to bring diagonal entry
            // to front. Adapted from Hypre: hypreDevice_StableSortTupleByTupleKey
            // option 2
            std::sort( rTpl.begin(),
                       rTpl.begin() + row_len,
                       [row]( const std::tuple<lidx_t, gidx_t, scalar_t> &a,
                              const std::tuple<lidx_t, gidx_t, scalar_t> &b ) -> bool {
                           const lidx_t lca = std::get<0>( a ), lcb = std::get<0>( b );
                           return row != lcb && ( lca < lcb || lca == row );
                       } );
        } else {
            std::sort( rTpl.begin(),
                       rTpl.begin() + row_len,
                       []( const std::tuple<lidx_t, gidx_t, scalar_t> &a,
                           const std::tuple<lidx_t, gidx_t, scalar_t> &b ) -> bool {
                           return std::get<0>( a ) < std::get<0>( b );
                       } );
        }

        for ( lidx_t k = 0; k < row_len; ++k ) {
            d_cols_loc[rs + k] = std::get<0>( rTpl[k] );
            d_cols[rs + k]     = std::get<1>( rTpl[k] );
            d_coeffs[rs + k]   = std::get<2>( rTpl[k] );
        }
    }

    // re-write column map to match now permuted columns
    if ( !d_is_diag ) {
        for ( lidx_t n = 0; n < d_nnz; ++n ) {
            d_cols_unq[d_cols_loc[n]] = d_cols[n];
        }
    }
}

template<typename Policy, class Allocator>
CSRLocalMatrixData<Policy, Allocator>::~CSRLocalMatrixData()
{
    AMPManager::decrementResource( "CSRLocalMatrixData" );
}

template<typename Policy, class Allocator>
std::shared_ptr<CSRLocalMatrixData<Policy, Allocator>>
CSRLocalMatrixData<Policy, Allocator>::cloneMatrixData()
{
    std::shared_ptr<CSRLocalMatrixData> cloneData;

    cloneData = std::make_shared<CSRLocalMatrixData>( d_pParameters,
                                                      d_memory_location,
                                                      d_first_row,
                                                      d_last_row,
                                                      d_first_col,
                                                      d_last_col,
                                                      d_is_diag );

    cloneData->d_is_empty = d_is_empty;
    cloneData->d_nnz      = d_nnz;

    if ( !d_is_empty ) {
        cloneData->d_nnz_per_row = sharedArrayBuilder( d_num_rows, lidxAllocator );
        cloneData->d_row_starts  = sharedArrayBuilder( d_num_rows + 1, lidxAllocator );
        cloneData->d_cols        = sharedArrayBuilder( d_nnz, gidxAllocator );
        cloneData->d_cols_loc    = sharedArrayBuilder( d_nnz, lidxAllocator );
        cloneData->d_coeffs      = sharedArrayBuilder( d_nnz, scalarAllocator );

        if ( d_memory_location < AMP::Utilities::MemoryType::device ) {
            std::copy( d_nnz_per_row.get(),
                       d_nnz_per_row.get() + d_num_rows,
                       cloneData->d_nnz_per_row.get() );
            std::copy( d_row_starts.get(),
                       d_row_starts.get() + d_num_rows + 1,
                       cloneData->d_row_starts.get() );
            std::copy( d_cols.get(), d_cols.get() + d_nnz, cloneData->d_cols.get() );
            std::copy( d_cols_loc.get(), d_cols_loc.get() + d_nnz, cloneData->d_cols_loc.get() );
            std::fill( cloneData->d_coeffs.get(), cloneData->d_coeffs.get() + d_nnz, 0.0 );
        } else {
#ifdef USE_DEVICE
            AMP::LinearAlgebra::DeviceDataHelpers<lidx_t>::copy_n(
                d_nnz_per_row.get(), d_num_rows, cloneData->d_nnz_per_row.get() );
            AMP::LinearAlgebra::DeviceDataHelpers<lidx_t>::copy_n(
                d_row_starts.get(), d_num_rows + 1, cloneData->d_row_starts.get() );
            AMP::LinearAlgebra::DeviceDataHelpers<gidx_t>::copy_n(
                d_cols.get(), d_nnz, cloneData->d_cols.get() );
            AMP::LinearAlgebra::DeviceDataHelpers<lidx_t>::copy_n(
                d_cols_loc.get(), d_nnz, cloneData->d_cols_loc.get() );
            AMP::LinearAlgebra::DeviceDataHelpers<scalar_t>::fill_n(
                cloneData->d_coeffs.get(), d_nnz, 0.0 );
#else
            AMP_ERROR( "No device found!" );
#endif
        }
    } else {
        cloneData->d_nnz_per_row = nullptr;
        cloneData->d_row_starts  = nullptr;
        cloneData->d_cols        = nullptr;
        cloneData->d_cols_loc    = nullptr;
        cloneData->d_coeffs      = nullptr;
    }

    return cloneData;
}

template<typename Policy, class Allocator>
void CSRLocalMatrixData<Policy, Allocator>::getRowByGlobalID( const size_t local_row,
                                                              std::vector<size_t> &cols,
                                                              std::vector<double> &values ) const
{
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getRowByGlobalID not implemented for device memory" );

    const auto row_offset = static_cast<size_t>( local_row );
    const auto offset     = d_row_starts[local_row];
    auto n                = d_nnz_per_row[row_offset];

    cols.resize( n );
    values.resize( n );

    if constexpr ( std::is_same_v<size_t, gidx_t> ) {
        std::copy( &d_cols[offset], &d_cols[offset] + n, cols.begin() );
    } else {
        std::transform( &d_cols[offset],
                        &d_cols[offset] + n,
                        cols.begin(),
                        []( size_t col ) -> gidx_t { return col; } );
    }

    if constexpr ( std::is_same_v<double, scalar_t> ) {
        std::copy( &d_coeffs[offset], &d_coeffs[offset] + n, values.begin() );
    } else {
        std::transform( &d_coeffs[offset],
                        &d_coeffs[offset] + n,
                        values.begin(),
                        []( size_t val ) -> scalar_t { return val; } );
    }
}

template<typename Policy, class Allocator>
void CSRLocalMatrixData<Policy, Allocator>::getValuesByGlobalID( const size_t local_row,
                                                                 const size_t col,
                                                                 void *values,
                                                                 const typeID &id ) const
{
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( getTypeID<scalar_t>() == id,
                "CSRLocalMatrixData::getValuesByGlobalID called with inconsistent typeID" );

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];

    for ( lidx_t i = start; i < end; ++i ) {
        if ( d_cols[i] == static_cast<gidx_t>( col ) ) {
            *( reinterpret_cast<scalar_t *>( values ) ) = d_coeffs[i];
        }
    }
}

template<typename Policy, class Allocator>
void CSRLocalMatrixData<Policy, Allocator>::addValuesByGlobalID( const size_t num_cols,
                                                                 const size_t local_row,
                                                                 const size_t *cols,
                                                                 const scalar_t *vals,
                                                                 const typeID &id )
{
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( getTypeID<scalar_t>() == id,
                "CSRLocalMatrixData::addValuesByGlobalID called with inconsistent typeID" );

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::addValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];

    // Inefficient because we don't assume order
    // not sure it's worth optimizing for our use cases
    for ( size_t icol = 0; icol < num_cols; ++icol ) {
        for ( lidx_t j = start; j < end; ++j ) {
            if ( d_cols[j] == static_cast<gidx_t>( cols[icol] ) ) {
                d_coeffs[j] += vals[icol];
            }
        }
    }
}

template<typename Policy, class Allocator>
void CSRLocalMatrixData<Policy, Allocator>::setValuesByGlobalID( const size_t num_cols,
                                                                 const size_t local_row,
                                                                 const size_t *cols,
                                                                 const scalar_t *vals,
                                                                 const typeID &id )
{
    if ( d_is_empty ) {
        return;
    }

    AMP_INSIST( getTypeID<scalar_t>() == id,
                "CSRLocalMatrixData::setValuesByGlobalID called with inconsistent typeID" );

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::setValuesByGlobalID not implemented for device memory" );

    const auto start = d_row_starts[local_row];
    auto end         = d_row_starts[local_row + 1];

    // Inefficient because we don't assume order
    // not sure it's worth optimizing for our use cases
    for ( size_t icol = 0; icol < num_cols; ++icol ) {
        for ( lidx_t j = start; j < end; ++j ) {
            if ( d_cols[j] == static_cast<gidx_t>( cols[icol] ) ) {
                d_coeffs[j] = vals[icol];
            }
        }
    }
}

template<typename Policy, class Allocator>
std::vector<size_t>
CSRLocalMatrixData<Policy, Allocator>::getColumnIDs( const size_t local_row ) const
{
    // Don't do anything on empty matrices
    if ( d_is_empty ) {
        return std::vector<size_t>();
    }

    AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRLocalMatrixData::getColumnIDs not implemented for device memory" );

    AMP_INSIST( d_cols && d_nnz_per_row,
                "CSRLocalMatrixData::getColumnIDs nnz layout must be initialized" );

    std::vector<size_t> cols;
    const auto row_offset = static_cast<size_t>( local_row );
    const auto offset     = d_row_starts[local_row];
    auto n                = d_nnz_per_row[row_offset];

    if constexpr ( std::is_same_v<size_t, gidx_t> ) {
        std::copy( &d_cols[offset], &d_cols[offset] + n, std::back_inserter( cols ) );
    } else {
        std::transform( &d_cols[offset],
                        &d_cols[offset] + n,
                        std::back_inserter( cols ),
                        []( size_t col ) -> gidx_t { return col; } );
    }
    return cols;
}

} // namespace AMP::LinearAlgebra

#endif
