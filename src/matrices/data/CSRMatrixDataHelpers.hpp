#ifndef included_AMP_CSRMatrixDataHelpers_hpp
#define included_AMP_CSRMatrixDataHelpers_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixDataHelpers.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_ptr.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/iterator/constant_iterator.h>
    #include <thrust/iterator/zip_iterator.h>
    #include <thrust/scatter.h>
    #include <thrust/sort.h>
    #include <thrust/transform.h>
    #include <thrust/tuple.h>
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
    lidx_t i = 0;
    while ( i < row_len ) {
        auto j = i;
        while ( j > 0 && cols[j - 1] > cols[j] ) {
            const auto j_col = cols[j];
            const auto j_val = coeffs[j];
            cols[j]          = cols[j - 1];
            coeffs[j]        = coeffs[j - 1];
            cols[j - 1]      = j_col;
            coeffs[j - 1]    = j_val;
            --j;
        }
        ++i;
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

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void diag_to_coo( const lidx_t *in_row_starts,
                             const lidx_t *in_cols_loc,
                             const scalar_t *in_coeffs,
                             const lidx_t num_rows,
                             const gidx_t first_col,
                             lidx_t *out_rows,
                             gidx_t *out_cols,
                             scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        for ( auto k = in_row_starts[i]; k < in_row_starts[i + 1]; ++k ) {
            out_rows[k]   = in_cols_loc[k];
            out_cols[k]   = static_cast<gidx_t>( i ) + first_col;
            out_coeffs[k] = in_coeffs[k];
        }
    }
}

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void offd_to_coo( const lidx_t *in_row_starts,
                             const gidx_t *in_cols,
                             const scalar_t *in_coeffs,
                             const lidx_t num_rows,
                             const gidx_t in_first_col,
                             const gidx_t out_first_col,
                             lidx_t *out_rows,
                             gidx_t *out_cols,
                             scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        for ( auto k = in_row_starts[i]; k < in_row_starts[i + 1]; ++k ) {
            out_rows[k]   = static_cast<lidx_t>( in_cols[k] - in_first_col );
            out_cols[k]   = static_cast<gidx_t>( i ) + out_first_col;
            out_coeffs[k] = in_coeffs[k];
        }
    }
}

template<typename lidx_t, typename gidx_t>
__global__ void row_sub_count( const gidx_t *rows,
                               const lidx_t num_rows,
                               const gidx_t first_row,
                               const lidx_t *diag_row_starts,
                               const lidx_t *offd_row_starts,
                               lidx_t *counts )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        const auto row_loc = static_cast<lidx_t>( rows[i] - first_row );

        counts[i] = diag_row_starts[row_loc + 1] - diag_row_starts[row_loc];
        counts[i] += offd_row_starts[row_loc + 1] - offd_row_starts[row_loc];
    }
}

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void row_sub_fill( const gidx_t *rows,
                              const lidx_t num_rows,
                              const gidx_t first_row,
                              const gidx_t first_col,
                              const lidx_t *diag_row_starts,
                              const lidx_t *offd_row_starts,
                              const lidx_t *diag_cols_loc,
                              const lidx_t *offd_cols_loc,
                              const scalar_t *diag_coeffs,
                              const scalar_t *offd_coeffs,
                              const gidx_t *offd_colmap,
                              const lidx_t *out_row_starts,
                              gidx_t *out_cols,
                              scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        const auto row_loc = static_cast<lidx_t>( rows[i] - first_row );
        const auto diag_rs = diag_row_starts[row_loc], diag_re = diag_row_starts[row_loc + 1];
        const auto offd_rs = offd_row_starts[row_loc], offd_re = offd_row_starts[row_loc + 1];
        lidx_t pos = out_row_starts[i];

        for ( lidx_t k = diag_rs; k < diag_re; ++k ) {
            out_cols[pos]   = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
            out_coeffs[pos] = diag_coeffs[k];
            ++pos;
        }
        for ( lidx_t k = offd_rs; k < offd_re; ++k ) {
            out_cols[pos]   = offd_colmap[offd_cols_loc[k]];
            out_coeffs[pos] = offd_coeffs[k];
            ++pos;
        }
    }
}

template<typename lidx_t, typename gidx_t>
__global__ void col_sub_count( const gidx_t idx_lo,
                               const gidx_t idx_up,
                               const gidx_t first_col,
                               const lidx_t *diag_row_starts,
                               const lidx_t *diag_cols_loc,
                               const lidx_t *offd_row_starts,
                               const lidx_t *offd_cols_loc,
                               const gidx_t *offd_cols_unq,
                               const lidx_t num_rows,
                               lidx_t *out_row_starts )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        out_row_starts[i] = 0;
        for ( auto k = diag_row_starts[i]; k < diag_row_starts[i + 1]; ++k ) {
            const auto col = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
            if ( idx_lo <= col && col < idx_up ) {
                out_row_starts[i]++;
            }
        }
        for ( auto k = offd_row_starts[i]; k < offd_row_starts[i + 1]; ++k ) {
            const auto col = offd_cols_unq[offd_cols_loc[k]];
            if ( idx_lo <= col && col < idx_up ) {
                out_row_starts[i]++;
            }
        }
    }
}

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void col_sub_fill( const gidx_t idx_lo,
                              const gidx_t idx_up,
                              const gidx_t first_col,
                              const lidx_t *diag_row_starts,
                              const lidx_t *diag_cols_loc,
                              const scalar_t *diag_coeffs,
                              const lidx_t *offd_row_starts,
                              const lidx_t *offd_cols_loc,
                              const gidx_t *offd_cols_unq,
                              const scalar_t *offd_coeffs,
                              const lidx_t num_rows,
                              lidx_t *out_row_starts,
                              gidx_t *out_cols,
                              scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        auto pos = out_row_starts[i];
        for ( auto k = diag_row_starts[i]; k < diag_row_starts[i + 1]; ++k ) {
            const auto col = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
            if ( idx_lo <= col && col < idx_up ) {
                out_cols[pos]   = col;
                out_coeffs[pos] = diag_coeffs[k];
                pos++;
            }
        }
        for ( auto k = offd_row_starts[i]; k < offd_row_starts[i + 1]; ++k ) {
            const auto col = offd_cols_unq[offd_cols_loc[k]];
            if ( idx_lo <= col && col < idx_up ) {
                out_cols[pos]   = col;
                out_coeffs[pos] = offd_coeffs[k];
                pos++;
            }
        }
    }
}

template<typename lidx_t>
__global__ void
horz_cat_count( const lidx_t *in_row_starts, const lidx_t num_rows, lidx_t *out_row_starts )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        out_row_starts[i] += ( in_row_starts[i + 1] - in_row_starts[i] );
    }
}

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void horz_cat_fill( const lidx_t *in_row_starts,
                               const gidx_t *in_cols,
                               const scalar_t *in_coeffs,
                               const lidx_t num_rows,
                               const lidx_t *out_row_starts,
                               lidx_t *row_nnz_ctrs,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        for ( auto n = in_row_starts[i]; n < in_row_starts[i + 1]; ++n ) {
            const auto rs        = out_row_starts[i];
            const auto ctr       = row_nnz_ctrs[i];
            out_cols[rs + ctr]   = in_cols[n];
            out_coeffs[rs + ctr] = in_coeffs[n];
            row_nnz_ctrs[i]++;
        }
    }
}

template<typename lidx_t, typename gidx_t>
__global__ void vert_cat_count( const lidx_t *row_starts,
                                const gidx_t *cols,
                                const lidx_t num_rows,
                                const gidx_t first_col,
                                const gidx_t last_col,
                                const bool keep_inside,
                                lidx_t *counts )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        lidx_t row_nnz = 0;
        for ( lidx_t k = row_starts[i]; k < row_starts[i + 1]; ++k ) {
            const bool inside = first_col <= cols[k] && cols[k] < last_col;
            if ( ( keep_inside && inside ) || ( !keep_inside && !inside ) ) {
                ++row_nnz;
            }
        }
        counts[i] = row_nnz;
    }
}

template<typename lidx_t, typename gidx_t, typename scalar_t>
__global__ void vert_cat_fill( const lidx_t *in_row_starts,
                               const gidx_t *in_cols,
                               const scalar_t *in_coeffs,
                               const lidx_t num_rows,
                               const gidx_t first_col,
                               const gidx_t last_col,
                               const bool keep_inside,
                               const lidx_t row_offset,
                               const lidx_t *out_row_starts,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        lidx_t cat_pos = out_row_starts[i + row_offset];
        for ( lidx_t k = in_row_starts[i]; k < in_row_starts[i + 1]; ++k ) {
            const auto c      = in_cols[k];
            const auto v      = in_coeffs[k];
            const bool inside = first_col <= c && c < last_col;
            if ( ( keep_inside && inside ) || ( !keep_inside && !inside ) ) {
                out_cols[cat_pos]   = c;
                out_coeffs[cat_pos] = v;
                ++cat_pos;
            }
        }
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
        sort_row_diag<<<GridDim, BlockDim>>>( row_starts, cols, coeffs, num_rows, first_col );
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
        sort_row_offd<<<GridDim, BlockDim>>>( row_starts, cols, coeffs, num_rows );
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
        getLastDeviceError( "CSRMatrixDataHelpers::GlobalToLocalOffd" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::GlobalToLocalOffd Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::TransposeDiag(
    const typename Config::lidx_t *in_row_starts,
    const typename Config::lidx_t *in_cols_loc,
    const typename Config::scalar_t *in_coeffs,
    const typename Config::lidx_t in_num_rows,
    const typename Config::lidx_t out_num_rows,
    const typename Config::gidx_t out_first_col,
    [[maybe_unused]] const typename Config::lidx_t tot_nnz,
    typename Config::lidx_t *out_row_starts,
    [[maybe_unused]] typename Config::lidx_t *out_cols_loc,
    typename Config::gidx_t *out_cols,
    typename Config::scalar_t *out_coeffs,
    typename Config::lidx_t *counters,
    [[maybe_unused]] typename Config::lidx_t *reduce_space )
{
    PROFILE( "CSRMatrixDataHelpers::TransposeDiag" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        // count occurrences of each column to set up nnz per row of output
        for ( lidx_t row = 0; row < in_num_rows; ++row ) {
            for ( lidx_t k = in_row_starts[row]; k < in_row_starts[row + 1]; ++k ) {
                const auto icl = in_cols_loc[k];
                out_row_starts[icl]++;
            }
        }

        // do cumulative sum of row counts to turn into offsets
        AMP::Utilities::Algorithms<lidx_t>::exclusive_scan(
            out_row_starts, out_num_rows + 1, out_row_starts, 0 );

        // second pass fill in entries using extra space for row position counters
        AMP::Utilities::Algorithms<lidx_t>::fill_n( counters, out_num_rows, 0 );
        for ( lidx_t row = 0; row < in_num_rows; ++row ) {
            for ( lidx_t k = in_row_starts[row]; k < in_row_starts[row + 1]; ++k ) {
                const auto icl  = in_cols_loc[k];
                const auto pos  = out_row_starts[icl] + counters[icl];
                out_cols[pos]   = static_cast<gidx_t>( row ) + out_first_col;
                out_coeffs[pos] = in_coeffs[k];
                counters[icl]++;
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        // the device approach is rather different from the host/serial one
        // The destination matrix has storage allocated for both the global
        // and local columns. The local columns are used temporarily as
        // storage for local row indices with the matrix in COO format.
        // Initially these will not be ordered in any useful way. Doing a
        // sort_by_key with these as keys and cols/coeffs as values will
        // rearrange everything to be back in CSR format. Now that the
        // rows are correctly ordered, a reduce_by_key against an iterator
        // of all ones will give the NNZ per row. Finally copy these into
        // the row starts and accumulate them.

        // write coo representation
        {
            dim3 BlockDim;
            dim3 GridDim;
            setKernelDims( in_num_rows, BlockDim, GridDim );
            diag_to_coo<<<GridDim, BlockDim>>>( in_row_starts,
                                                in_cols_loc,
                                                in_coeffs,
                                                in_num_rows,
                                                out_first_col,
                                                out_cols_loc,
                                                out_cols,
                                                out_coeffs );
            getLastDeviceError( "CSRMatrixDataHelpers::TransposeDiag (to COO)" );
        }

        // make zip-iterator to tie cols and coeffs together
        using GidxIter         = thrust::device_ptr<gidx_t>;
        using ScalarIter       = thrust::device_ptr<scalar_t>;
        using IterTuple        = thrust::tuple<GidxIter, ScalarIter>;
        using ZipIter          = thrust::zip_iterator<IterTuple>;
        GidxIter cols_iter     = thrust::device_pointer_cast( out_cols );
        ScalarIter coeffs_iter = thrust::device_pointer_cast( out_coeffs );
        ZipIter vals_iter( thrust::make_tuple( cols_iter, coeffs_iter ) );

        // sort by key to rearrange cols and coeffs
        thrust::sort_by_key( thrust::device, out_cols_loc, out_cols_loc + tot_nnz, vals_iter );
        getLastDeviceError( "CSRMatrixDataHelpers::TransposeDiag (sort by key)" );

        // reduce by key
        auto it = thrust::reduce_by_key( thrust::device,
                                         out_cols_loc,
                                         out_cols_loc + tot_nnz,
                                         thrust::constant_iterator<lidx_t>( 1 ),
                                         reduce_space,
                                         counters )
                      .first;
        ptrdiff_t num_unq = it - reduce_space;
        getLastDeviceError( "CSRMatrixDataHelpers::TransposeDiag (reduce by key)" );

        // copy into row starts and accumulate
        thrust::scatter(
            thrust::device, counters, counters + num_unq, reduce_space, out_row_starts );
        AMP::Utilities::Algorithms<lidx_t>::exclusive_scan(
            out_row_starts, out_num_rows + 1, out_row_starts, 0 );

        getLastDeviceError( "CSRMatrixDataHelpers::TransposeDiag" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::TransposeDiag Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::TransposeOffd(
    const typename Config::lidx_t *in_row_starts,
    const typename Config::gidx_t *in_cols,
    const typename Config::scalar_t *in_coeffs,
    const typename Config::lidx_t in_num_rows,
    const typename Config::gidx_t in_first_col,
    const typename Config::lidx_t out_num_rows,
    const typename Config::gidx_t out_first_col,
    [[maybe_unused]] const typename Config::lidx_t tot_nnz,
    typename Config::lidx_t *out_row_starts,
    [[maybe_unused]] typename Config::lidx_t *out_cols_loc,
    typename Config::gidx_t *out_cols,
    typename Config::scalar_t *out_coeffs,
    typename Config::lidx_t *counters,
    [[maybe_unused]] typename Config::lidx_t *reduce_space )
{
    PROFILE( "CSRMatrixDataHelpers::TransposeOffd" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        // count occurrences of each column to set up nnz per row of output
        for ( lidx_t row = 0; row < in_num_rows; ++row ) {
            for ( lidx_t k = in_row_starts[row]; k < in_row_starts[row + 1]; ++k ) {
                const auto icl = in_cols[k] - in_first_col;
                out_row_starts[icl]++;
            }
        }

        // do cumulative sum of row counts to turn into offsets
        AMP::Utilities::Algorithms<lidx_t>::exclusive_scan(
            out_row_starts, out_num_rows + 1, out_row_starts, 0 );

        // second pass fill in entries using extra space for row position counters
        AMP::Utilities::Algorithms<lidx_t>::fill_n( counters, out_num_rows, 0 );
        for ( lidx_t row = 0; row < in_num_rows; ++row ) {
            for ( lidx_t k = in_row_starts[row]; k < in_row_starts[row + 1]; ++k ) {
                const auto icl  = in_cols[k] - in_first_col;
                const auto pos  = out_row_starts[icl] + counters[icl];
                out_cols[pos]   = static_cast<gidx_t>( row ) + out_first_col;
                out_coeffs[pos] = in_coeffs[k];
                counters[icl]++;
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        // the off-diagonal device transpose is nearly the same as for diag blocks
        // with slightly different indexing to define the output rows

        // write coo representation
        {
            dim3 BlockDim;
            dim3 GridDim;
            setKernelDims( in_num_rows, BlockDim, GridDim );
            offd_to_coo<<<GridDim, BlockDim>>>( in_row_starts,
                                                in_cols,
                                                in_coeffs,
                                                in_num_rows,
                                                in_first_col,
                                                out_first_col,
                                                out_cols_loc,
                                                out_cols,
                                                out_coeffs );
            getLastDeviceError( "CSRMatrixDataHelpers::TransposeOffd (to COO)" );
        }

        // make zip-iterator to tie cols and coeffs together
        using GidxIter         = thrust::device_ptr<gidx_t>;
        using ScalarIter       = thrust::device_ptr<scalar_t>;
        using IterTuple        = thrust::tuple<GidxIter, ScalarIter>;
        using ZipIter          = thrust::zip_iterator<IterTuple>;
        GidxIter cols_iter     = thrust::device_pointer_cast( out_cols );
        ScalarIter coeffs_iter = thrust::device_pointer_cast( out_coeffs );
        ZipIter vals_iter( thrust::make_tuple( cols_iter, coeffs_iter ) );

        // sort by key to rearrange cols and coeffs
        thrust::sort_by_key( thrust::device, out_cols_loc, out_cols_loc + tot_nnz, vals_iter );
        getLastDeviceError( "CSRMatrixDataHelpers::TransposeOffd (sort by key)" );

        // reduce by key
        auto it = thrust::reduce_by_key( thrust::device,
                                         out_cols_loc,
                                         out_cols_loc + tot_nnz,
                                         thrust::constant_iterator<lidx_t>( 1 ),
                                         reduce_space,
                                         counters )
                      .first;
        ptrdiff_t num_unq = it - reduce_space;
        getLastDeviceError( "CSRMatrixDataHelpers::TransposeOffd (reduce by key)" );

        // copy into row starts and accumulate
        thrust::scatter(
            thrust::device, counters, counters + num_unq, reduce_space, out_row_starts );
        AMP::Utilities::Algorithms<lidx_t>::exclusive_scan(
            out_row_starts, out_num_rows + 1, out_row_starts, 0 );

        getLastDeviceError( "CSRMatrixDataHelpers::TransposeOffd" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::TransposeOffd Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::RowSubsetCountNNZ(
    const typename Config::gidx_t *rows,
    const typename Config::lidx_t num_rows,
    const typename Config::gidx_t first_row,
    const typename Config::lidx_t *diag_row_starts,
    const typename Config::lidx_t *offd_row_starts,
    typename Config::lidx_t *counts )
{
    PROFILE( "CSRMatrixDataHelpers::RowSubsetCountNNZ" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t n = 0; n < num_rows; ++n ) {
            const auto row_loc = static_cast<lidx_t>( rows[n] - first_row );

            counts[n] = diag_row_starts[row_loc + 1] - diag_row_starts[row_loc];
            counts[n] += offd_row_starts[row_loc + 1] - offd_row_starts[row_loc];
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        row_sub_count<<<GridDim, BlockDim>>>(
            rows, num_rows, first_row, diag_row_starts, offd_row_starts, counts );
        getLastDeviceError( "CSRMatrixDataHelpers::RowSubsetCountNNZ" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::RowSubsetCountNNZ Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::RowSubsetFill( const typename Config::gidx_t *rows,
                                                  const typename Config::lidx_t num_rows,
                                                  const typename Config::gidx_t first_row,
                                                  const typename Config::gidx_t first_col,
                                                  const typename Config::lidx_t *diag_row_starts,
                                                  const typename Config::lidx_t *offd_row_starts,
                                                  const typename Config::lidx_t *diag_cols_loc,
                                                  const typename Config::lidx_t *offd_cols_loc,
                                                  const typename Config::scalar_t *diag_coeffs,
                                                  const typename Config::scalar_t *offd_coeffs,
                                                  const typename Config::gidx_t *offd_colmap,
                                                  const typename Config::lidx_t *out_row_starts,
                                                  typename Config::gidx_t *out_cols,
                                                  typename Config::scalar_t *out_coeffs )
{
    PROFILE( "CSRMatrixDataHelpers::RowSubsetFill" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t n = 0; n < num_rows; ++n ) {
            const auto row_loc = static_cast<lidx_t>( rows[n] - first_row );
            const auto diag_rs = diag_row_starts[row_loc], diag_re = diag_row_starts[row_loc + 1];
            const auto offd_rs = offd_row_starts[row_loc], offd_re = offd_row_starts[row_loc + 1];
            lidx_t pos = out_row_starts[n];

            for ( lidx_t k = diag_rs; k < diag_re; ++k ) {
                out_cols[pos]   = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
                out_coeffs[pos] = diag_coeffs[k];
                ++pos;
            }
            for ( lidx_t k = offd_rs; k < offd_re; ++k ) {
                out_cols[pos]   = offd_colmap[offd_cols_loc[k]];
                out_coeffs[pos] = offd_coeffs[k];
                ++pos;
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        row_sub_fill<<<GridDim, BlockDim>>>( rows,
                                             num_rows,
                                             first_row,
                                             first_col,
                                             diag_row_starts,
                                             offd_row_starts,
                                             diag_cols_loc,
                                             offd_cols_loc,
                                             diag_coeffs,
                                             offd_coeffs,
                                             offd_colmap,
                                             out_row_starts,
                                             out_cols,
                                             out_coeffs );
        getLastDeviceError( "CSRMatrixDataHelpers::RowSubsetFill" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::RowSubsetFill Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ColSubsetCountNNZ(
    const typename Config::gidx_t idx_lo,
    const typename Config::gidx_t idx_up,
    const typename Config::gidx_t first_col,
    const typename Config::lidx_t *diag_row_starts,
    const typename Config::lidx_t *diag_cols_loc,
    const typename Config::lidx_t *offd_row_starts,
    const typename Config::lidx_t *offd_cols_loc,
    const typename Config::gidx_t *offd_cols_unq,
    const typename Config::lidx_t num_rows,
    typename Config::lidx_t *out_row_starts )
{
    PROFILE( "CSRMatrixDataHelpers::ColSubsetCountNNZ" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            out_row_starts[row] = 0;
            for ( auto k = diag_row_starts[row]; k < diag_row_starts[row + 1]; ++k ) {
                const auto col = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
                if ( idx_lo <= col && col < idx_up ) {
                    out_row_starts[row]++;
                }
            }
            for ( auto k = offd_row_starts[row]; k < offd_row_starts[row + 1]; ++k ) {
                const auto col = offd_cols_unq[offd_cols_loc[k]];
                if ( idx_lo <= col && col < idx_up ) {
                    out_row_starts[row]++;
                }
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        col_sub_count<<<GridDim, BlockDim>>>( idx_lo,
                                              idx_up,
                                              first_col,
                                              diag_row_starts,
                                              diag_cols_loc,
                                              offd_row_starts,
                                              offd_cols_loc,
                                              offd_cols_unq,
                                              num_rows,
                                              out_row_starts );
        getLastDeviceError( "CSRMatrixDataHelpers::ColSubsetCountNNZ" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ColSubsetCountNNZ Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ColSubsetFill( const typename Config::gidx_t idx_lo,
                                                  const typename Config::gidx_t idx_up,
                                                  const typename Config::gidx_t first_col,
                                                  const typename Config::lidx_t *diag_row_starts,
                                                  const typename Config::lidx_t *diag_cols_loc,
                                                  const typename Config::scalar_t *diag_coeffs,
                                                  const typename Config::lidx_t *offd_row_starts,
                                                  const typename Config::lidx_t *offd_cols_loc,
                                                  const typename Config::gidx_t *offd_cols_unq,
                                                  const typename Config::scalar_t *offd_coeffs,
                                                  const typename Config::lidx_t num_rows,
                                                  typename Config::lidx_t *out_row_starts,
                                                  typename Config::gidx_t *out_cols,
                                                  typename Config::scalar_t *out_coeffs )
{
    PROFILE( "CSRMatrixDataHelpers::ColSubsetFill" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            auto pos = out_row_starts[row];
            for ( auto k = diag_row_starts[row]; k < diag_row_starts[row + 1]; ++k ) {
                const auto col = static_cast<gidx_t>( diag_cols_loc[k] ) + first_col;
                if ( idx_lo <= col && col < idx_up ) {
                    out_cols[pos]   = col;
                    out_coeffs[pos] = diag_coeffs[k];
                    pos++;
                }
            }
            for ( auto k = offd_row_starts[row]; k < offd_row_starts[row + 1]; ++k ) {
                const auto col = offd_cols_unq[offd_cols_loc[k]];
                if ( idx_lo <= col && col < idx_up ) {
                    out_cols[pos]   = col;
                    out_coeffs[pos] = offd_coeffs[k];
                    pos++;
                }
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        col_sub_fill<<<GridDim, BlockDim>>>( idx_lo,
                                             idx_up,
                                             first_col,
                                             diag_row_starts,
                                             diag_cols_loc,
                                             diag_coeffs,
                                             offd_row_starts,
                                             offd_cols_loc,
                                             offd_cols_unq,
                                             offd_coeffs,
                                             num_rows,
                                             out_row_starts,
                                             out_cols,
                                             out_coeffs );
        getLastDeviceError( "CSRMatrixDataHelpers::ColSubsetFill" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ColSubsetFill Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ConcatHorizontalCountNNZ(
    const typename Config::lidx_t *in_row_starts,
    const typename Config::lidx_t num_rows,
    typename Config::lidx_t *out_row_starts )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            out_row_starts[row] += ( in_row_starts[row + 1] - in_row_starts[row] );
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        horz_cat_count<<<GridDim, BlockDim>>>( in_row_starts, num_rows, out_row_starts );
        getLastDeviceError( "CSRMatrixDataHelpers::ConcatHorizontalCountNNZ" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ConcatHorizontalCountNNZ Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ConcatHorizontalFill(
    const typename Config::lidx_t *in_row_starts,
    const typename Config::gidx_t *in_cols,
    const typename Config::scalar_t *in_coeffs,
    const typename Config::lidx_t num_rows,
    const typename Config::lidx_t *out_row_starts,
    typename Config::lidx_t *row_nnz_ctrs,
    typename Config::gidx_t *out_cols,
    typename Config::scalar_t *out_coeffs )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            for ( auto n = in_row_starts[row]; n < in_row_starts[row + 1]; ++n ) {
                const auto rs        = out_row_starts[row];
                const auto ctr       = row_nnz_ctrs[row];
                out_cols[rs + ctr]   = in_cols[n];
                out_coeffs[rs + ctr] = in_coeffs[n];
                row_nnz_ctrs[row]++;
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        horz_cat_fill<<<GridDim, BlockDim>>>( in_row_starts,
                                              in_cols,
                                              in_coeffs,
                                              num_rows,
                                              out_row_starts,
                                              row_nnz_ctrs,
                                              out_cols,
                                              out_coeffs );
        getLastDeviceError( "CSRMatrixDataHelpers::ConcatHorizontalFill" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ConcatHorizontalFill Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ConcatVerticalCountNNZ(
    const typename Config::lidx_t *row_starts,
    const typename Config::gidx_t *cols,
    const typename Config::lidx_t num_rows,
    const typename Config::gidx_t first_col,
    const typename Config::gidx_t last_col,
    const bool keep_inside,
    typename Config::lidx_t *counts )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            lidx_t row_nnz = 0;
            for ( lidx_t k = row_starts[row]; k < row_starts[row + 1]; ++k ) {
                const bool inside = first_col <= cols[k] && cols[k] < last_col;
                if ( ( keep_inside && inside ) || ( !keep_inside && !inside ) ) {
                    ++row_nnz;
                }
            }
            counts[row] = row_nnz;
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        vert_cat_count<<<GridDim, BlockDim>>>(
            row_starts, cols, num_rows, first_col, last_col, keep_inside, counts );
        getLastDeviceError( "CSRMatrixDataHelpers::ConcatVerticalCountNNZ" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ConcatVerticalCountNNZ Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::ConcatVerticalFill(
    const typename Config::lidx_t *in_row_starts,
    const typename Config::gidx_t *in_cols,
    const typename Config::scalar_t *in_coeffs,
    const typename Config::lidx_t num_rows,
    const typename Config::gidx_t first_col,
    const typename Config::gidx_t last_col,
    bool const keep_inside,
    const typename Config::lidx_t row_offset,
    const typename Config::lidx_t *out_row_starts,
    typename Config::gidx_t *out_cols,
    typename Config::scalar_t *out_coeffs )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            lidx_t cat_pos = out_row_starts[row + row_offset];
            for ( lidx_t k = in_row_starts[row]; k < in_row_starts[row + 1]; ++k ) {
                const auto c      = in_cols[k];
                const auto v      = in_coeffs[k];
                const bool inside = first_col <= c && c < last_col;
                if ( ( keep_inside && inside ) || ( !keep_inside && !inside ) ) {
                    out_cols[cat_pos]   = c;
                    out_coeffs[cat_pos] = v;
                    ++cat_pos;
                }
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        vert_cat_fill<<<GridDim, BlockDim>>>( in_row_starts,
                                              in_cols,
                                              in_coeffs,
                                              num_rows,
                                              first_col,
                                              last_col,
                                              keep_inside,
                                              row_offset,
                                              out_row_starts,
                                              out_cols,
                                              out_coeffs );
        getLastDeviceError( "CSRMatrixDataHelpers::ConcatVerticalFill" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::ConcatVerticalFill Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::MaskCountNNZ( const typename Config::lidx_t *in_row_starts,
                                                 const unsigned char *mask,
                                                 const bool keep_first,
                                                 const typename Config::lidx_t num_rows,
                                                 typename Config::lidx_t *out_row_starts )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        const lidx_t kf = keep_first ? 1 : 0; // if keeping then start count at one and skip entry
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            lidx_t row_nnz = kf;
            for ( lidx_t c = in_row_starts[row] + kf; c < in_row_starts[row + 1]; ++c ) {
                AMP_DEBUG_ASSERT( mask[c] == 0 || mask[c] == 1 );
                row_nnz += static_cast<lidx_t>( mask[c] );
            }
            out_row_starts[row] = row_nnz;
        }
    } else {
#ifdef AMP_USE_DEVICE
        AMP_ERROR( "CSRMatrixDataHelpers::MaskCountNNZ not implemented on device yet" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::MaskCountNNZ Undefined memory location" );
#endif
    }
}

template<typename Config>
void CSRMatrixDataHelpers<Config>::MaskFillDiag( const typename Config::lidx_t *in_row_starts,
                                                 const typename Config::lidx_t *in_cols_loc,
                                                 const typename Config::scalar_t *in_coeffs,
                                                 const unsigned char *mask,
                                                 const bool keep_first,
                                                 const typename Config::lidx_t num_rows,
                                                 const typename Config::lidx_t *out_row_starts,
                                                 typename Config::lidx_t *out_cols_loc,
                                                 typename Config::scalar_t *out_coeffs )
{
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        for ( lidx_t row = 0; row < num_rows; ++row ) {
            auto pos = out_row_starts[row];
            for ( lidx_t c = in_row_starts[row]; c < in_row_starts[row + 1]; ++c ) {
                if ( mask[c] == 1 || ( keep_first && c == in_row_starts[row] ) ) {
                    out_cols_loc[pos] = in_cols_loc[c];
                    if ( out_coeffs != nullptr ) {
                        out_coeffs[pos] = in_coeffs[c];
                    }
                    ++pos;
                }
            }
        }
    } else {
#ifdef AMP_USE_DEVICE
        AMP_ERROR( "CSRMatrixDataHelpers::MaskFillDiag not implemented on device yet" );
#else
        AMP_ERROR( "CSRMatrixDataHelpers::MaskFillDiag Undefined memory location" );
#endif
    }
}

} // namespace LinearAlgebra
} // namespace AMP

#endif
