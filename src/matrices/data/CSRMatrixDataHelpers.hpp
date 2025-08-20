#ifndef included_AMP_CSRMatrixDataHelpers_hpp
#define included_AMP_CSRMatrixDataHelpers_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixDataHelpers.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
#endif

#include "ProfilerApp.h"

#include <algorithm>
#include <numeric>

namespace AMP {
namespace LinearAlgebra {

#ifdef AMP_USE_DEVICE
// function to sort a single row in-place
template<typename lidx_t, typename gidx_t, typename scalar_t>
__device__ void sort_row( gidx_t *cols, scalar_t *coeffs, const lidx_t row_len )
{
    for ( lidx_t i = 0; i < row_len; ++i ) {
        for ( lidx_t j = i + 1; j < row_len; ++j ) {
            if ( cols[j] < cols[i] ) {
                const gidx_t col_tmp     = cols[j];
                const scalar_t coeff_tmp = coeffs[j];
                cols[j]                  = cols[i];
                coeffs[j]                = coeffs[i];
                cols[i]                  = col_tmp;
                coeffs[i]                = coeff_tmp;
                break;
            }
        }
    }
}

// move diagonal value to front then sort remainder of row
template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void sort_row_diag( const lidx_t *row_starts,
                               gidx_t *cols,
                               scalar_t *coeffs,
                               const lidx_t num_rows,
                               const gidx_t first_col )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        const lidx_t rs = row_starts[i];

        // find diagonal and swap
        const gidx_t diag_idx = first_col + static_cast<gidx_t>( i );
        for ( lidx_t j = rs + 1; j < row_starts[i + 1]; ++j ) {
            if ( cols[j] == diag_idx ) {
                const scalar_t coeff_tmp = coeffs[j];
                cols[j]                  = cols[rs];
                coeffs[j]                = coeffs[rs];
                cols[rs]                 = diag_idx;
                coeffs[rs]               = coeff_tmp;
                break;
            }
        }

        // sort rest of row in usual way
        const lidx_t row_len = row_starts[i + 1] - rs;
        sort_row( &cols[rs + 1], &coeffs[rs + 1], row_len - 1 );
    }
}

// directly sort each row
template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void
sort_row_offd( const lidx_t *row_starts, gidx_t *cols, scalar_t *coeffs, const lidx_t num_rows )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        const auto rs      = row_starts[i];
        const auto row_len = row_starts[i + 1] - rs;
        sort_row( &cols[rs], &coeffs[rs], row_len );
    }
}
#endif

template<typename Config>
void CSRMatrixDataHelpers<Config>::SortColumnsDiag( typename Config::lidx_t *row_starts,
                                                    typename Config::gidx_t *cols,
                                                    typename Config::scalar_t *coeffs,
                                                    typename Config::lidx_t num_rows,
                                                    typename Config::gidx_t first_col )
{
    PROFILE( "CSRMatrixDataHelpers::SortColumnsDiag" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        std::vector<lidx_t> row_indices;
        std::vector<gidx_t> cols_tmp;
        std::vector<scalar_t> coeffs_tmp;
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            const auto rs      = row_starts[row];
            const auto row_len = row_starts[row + 1] - rs;
            if ( row_len == 0 )
                continue;

            row_indices.resize( row_len );
            cols_tmp.resize( row_len );
            coeffs_tmp.resize( row_len );

            // initial row numbers
            std::iota( row_indices.begin(), row_indices.end(), 0 );

            // sort row_indices using column indices
            const auto cols_ptr   = &cols[rs];
            const gidx_t diag_idx = first_col + static_cast<gidx_t>( row );
            // diag block puts diag entry first, then ascending order on local col
            std::sort( row_indices.begin(),
                       row_indices.end(),
                       [diag_idx, cols_ptr]( const lidx_t &a, const lidx_t &b ) -> bool {
                           return diag_idx != cols_ptr[b] &&
                                  ( cols_ptr[a] < cols_ptr[b] || cols_ptr[a] == diag_idx );
                       } );

            // use row_indices to fill sorted col and coeff vectors
            for ( lidx_t k = 0; k < row_len; ++k ) {
                cols_tmp[k]   = cols[rs + row_indices[k]];
                coeffs_tmp[k] = coeffs[rs + row_indices[k]];
            }
            std::copy( cols_tmp.begin(), cols_tmp.end(), &cols[rs] );
            std::copy( coeffs_tmp.begin(), coeffs_tmp.end(), &coeffs[rs] );
        }
    } else {
#ifdef AMP_USE_DEVICE
        AMP_ASSERT( AMP::Utilities::getMemoryType( row_starts ) >
                    AMP::Utilities::MemoryType::host );
        AMP_ASSERT( AMP::Utilities::getMemoryType( cols ) > AMP::Utilities::MemoryType::host );
        AMP_ASSERT( AMP::Utilities::getMemoryType( coeffs ) > AMP::Utilities::MemoryType::host );
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        deviceSynchronize();
        sort_row_diag<<<GridDim, BlockDim>>>( row_starts, cols, coeffs, num_rows, first_col );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixDataHelpers::SortColumnsDiag" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::SortColumnsDiag Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::SortColumnsOffd( typename Config::lidx_t *row_starts,
                                                    typename Config::gidx_t *cols,
                                                    typename Config::scalar_t *coeffs,
                                                    typename Config::lidx_t num_rows )
{
    PROFILE( "CSRMatrixDataHelpers::SortColumnsOffd" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        std::vector<lidx_t> row_indices;
        std::vector<gidx_t> cols_tmp;
        std::vector<scalar_t> coeffs_tmp;
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            const auto rs      = row_starts[row];
            const auto row_len = row_starts[row + 1] - rs;
            if ( row_len == 0 )
                continue;

            row_indices.resize( row_len );
            cols_tmp.resize( row_len );
            coeffs_tmp.resize( row_len );

            // initial row numbers
            std::iota( row_indices.begin(), row_indices.end(), 0 );

            // sort row_indices using column indices
            const auto cols_ptr = &cols[rs];
            // offd block is plain ascending order on local col
            std::sort( row_indices.begin(),
                       row_indices.end(),
                       [cols_ptr]( const lidx_t &a, const lidx_t &b ) -> bool {
                           return cols_ptr[a] < cols_ptr[b];
                       } );

            // use row_indices to fill sorted col and coeff vectors
            for ( lidx_t k = 0; k < row_len; ++k ) {
                cols_tmp[k]   = cols[rs + row_indices[k]];
                coeffs_tmp[k] = coeffs[rs + row_indices[k]];
            }
            std::copy( cols_tmp.begin(), cols_tmp.end(), &cols[rs] );
            std::copy( coeffs_tmp.begin(), coeffs_tmp.end(), &coeffs[rs] );
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        deviceSynchronize();
        sort_row_offd<<<GridDim, BlockDim>>>( row_starts, cols, coeffs, num_rows );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixDataHelpers::SortColumnsOffd" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::SortColumnsOffd Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::GlobalToLocalDiag( typename Config::gidx_t *cols,
                                                      typename Config::lidx_t nnz,
                                                      typename Config::gidx_t first_col,
                                                      typename Config::lidx_t *cols_loc )
{
    PROFILE( "CSRMatrixDataHelpers::GlobalToLocalDiag" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        std::transform( cols, cols + nnz, cols_loc, [first_col]( const gidx_t gc ) -> lidx_t {
            return static_cast<lidx_t>( gc - first_col );
        } );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::transform( thrust::device,
                           cols,
                           cols + nnz,
                           cols_loc,
                           [first_col] __device__( const gidx_t gc ) -> lidx_t {
                               return static_cast<lidx_t>( gc - first_col );
                           } );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixDataHelpers::GlobalToLocalDiag" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::GlobalToLocalDiag Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::GlobalToLocalOffd( typename Config::gidx_t *cols,
                                                      typename Config::lidx_t nnz,
                                                      typename Config::gidx_t *cols_unq,
                                                      typename Config::lidx_t ncols_unq,
                                                      typename Config::lidx_t *cols_loc )
{
    PROFILE( "CSRMatrixDataHelpers::GlobalToLocalOffd" );
    // copy and modify from AMP::Utilities::findfirst to suit task
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        std::transform( cols, cols + nnz, cols_loc, [cols_unq, ncols_unq]( gidx_t gc ) -> lidx_t {
            AMP_DEBUG_ASSERT( cols_unq[0] <= gc && gc <= cols_unq[ncols_unq - 1] );
            lidx_t lower = 0, upper = ncols_unq - 1, idx;
            while ( ( upper - lower ) > 1 ) {
                idx = ( upper + lower ) / 2;
                if ( cols_unq[idx] == gc ) {
                    return idx;
                } else if ( cols_unq[idx] > gc ) {
                    upper = idx - 1;
                } else {
                    lower = idx + 1;
                }
            }
            return gc == cols_unq[upper] ? upper : lower;
        } );
    } else {
#ifdef AMP_USE_DEVICE
        thrust::transform( thrust::device,
                           cols,
                           cols + nnz,
                           cols_loc,
                           [cols_unq, ncols_unq] __device__( gidx_t gc ) -> lidx_t {
                               lidx_t lower = 0, upper = ncols_unq - 1, idx;
                               while ( ( upper - lower ) > 1 ) {
                                   idx = ( upper + lower ) / 2;
                                   if ( cols_unq[idx] == gc ) {
                                       return idx;
                                   } else if ( cols_unq[idx] > gc ) {
                                       upper = idx - 1;
                                   } else {
                                       lower = idx + 1;
                                   }
                               }
                               return gc == cols_unq[upper] ? upper : lower;
                           } );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixDataHelpers::GlobalToLocalOffd" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::GlobalToLocalOffd Undefined memory location" );
#endif
    }
}

} // namespace LinearAlgebra
} // namespace AMP

#endif
