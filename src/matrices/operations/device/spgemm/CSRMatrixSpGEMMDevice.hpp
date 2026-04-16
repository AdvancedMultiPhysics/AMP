#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/operations/device/spgemm/CSRMatrixSpGEMMDevice.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
#endif

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::multiplyLocal( std::shared_ptr<localmatrixdata_t> A_data,
                                                   std::shared_ptr<localmatrixdata_t> B_data,
                                                   std::shared_ptr<localmatrixdata_t> C_data )
{
    AMP_DEBUG_ASSERT( A_data != nullptr );
    AMP_DEBUG_ASSERT( B_data != nullptr );
    AMP_DEBUG_ASSERT( C_data != nullptr );

    if ( A_data->isEmpty() || B_data->isEmpty() ) {
        return;
    }

    // shapes of A and B
    const auto A_nrows = static_cast<int64_t>( A_data->numLocalRows() );
    const auto A_ncols = A_data->isDiag() ? static_cast<int64_t>( A_data->numLocalColumns() ) :
                                            static_cast<int64_t>( A_data->numUniqueColumns() );
    const auto B_ncols = B_data->isDiag() ? static_cast<int64_t>( B_data->numLocalColumns() ) :
                                            static_cast<int64_t>( B_data->numUniqueColumns() );

    // all fields from blocks involved
    lidx_t *A_rs = nullptr, *A_cols_loc = nullptr;
    gidx_t *A_cols     = nullptr;
    scalar_t *A_coeffs = nullptr;

    lidx_t *B_rs = nullptr, *B_cols_loc = nullptr;
    gidx_t *B_cols     = nullptr;
    scalar_t *B_coeffs = nullptr;

    // Extract data fields from A and B
    std::tie( A_rs, A_cols, A_cols_loc, A_coeffs ) = A_data->getDataFields();
    std::tie( B_rs, B_cols, B_cols_loc, B_coeffs ) = B_data->getDataFields();
    const auto A_nnz = static_cast<int64_t>( A_data->numberOfNonZeros() );
    const auto B_nnz = static_cast<int64_t>( B_data->numberOfNonZeros() );

    // C has row pointers allocated but unfilled
    lidx_t *C_rs = C_data->getRowStarts();

    // Create vendor SpGEMM object and trigger internal allocs
    VendorSpGEMM<lidx_t, lidx_t, scalar_t> spgemm( A_nrows,
                                                   B_ncols,
                                                   A_ncols,
                                                   A_nnz,
                                                   A_rs,
                                                   A_cols_loc,
                                                   A_coeffs,
                                                   B_nnz,
                                                   B_rs,
                                                   B_cols_loc,
                                                   B_coeffs,
                                                   C_rs );

    // Get nnz for C and allocate internals
    auto C_nnz = static_cast<lidx_t>( spgemm.getCnnz() );
    C_data->setNNZ( C_nnz );

    // pull out the now allocated C internals
    lidx_t *C_cols_loc                             = nullptr;
    gidx_t *C_cols                                 = nullptr;
    scalar_t *C_coeffs                             = nullptr;
    std::tie( C_rs, C_cols, C_cols_loc, C_coeffs ) = C_data->getDataFields();

    // Compute SpGEMM
    spgemm.compute( C_rs, C_cols_loc, C_coeffs );

    // Convert the local indices to globals to make merges easier
    if ( C_data->isDiag() ) {
        const auto first_col = C_data->beginCol();
        thrust::transform( thrust::device,
                           C_cols_loc,
                           C_cols_loc + C_nnz,
                           C_cols,
                           [first_col] __device__( const lidx_t lc ) -> gidx_t {
                               return static_cast<gidx_t>( lc ) + first_col;
                           } );
    } else {
        const auto colmap = B_data->getColumnMap();
        thrust::transform(
            thrust::device,
            C_cols_loc,
            C_cols_loc + C_nnz,
            C_cols,
            [colmap] __device__( const lidx_t lc ) -> gidx_t { return colmap[lc]; } );
    }
    // exiting function destructs spgemm wrapper and frees its internals
}

template<typename gidx_t, typename lidx_t>
__global__ void merge_row_count( const lidx_t num_rows,
                                 const lidx_t *A_rs,
                                 const gidx_t *A_cols,
                                 const lidx_t *B_rs,
                                 const gidx_t *B_cols,
                                 lidx_t *C_rs )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        // all of A counts in automatically
        C_rs[row] = A_rs[row + 1] - A_rs[row];
        // column values are sorted, so walk through A cols and B cols
        // simultaneously to find matches
        lidx_t num_repeats = 0, A_ptr = A_rs[row], B_ptr = B_rs[row];
        for ( ; A_ptr < A_rs[row + 1] && B_ptr < B_rs[row + 1]; ) {
            const auto Ac = A_cols[A_ptr], Bc = B_cols[B_ptr];
            if ( Ac == Bc ) {
                // entries match, increment counter and both ptrs
                ++num_repeats;
                ++A_ptr;
                ++B_ptr;
            } else if ( Ac < Bc ) {
                // A lags B, increment A ptr to check further in row
                ++A_ptr;
            } else {
                // B lags A, Bc not a repeat, so increment B ptr to check next
                ++B_ptr;
            }
        }
        // either all A cols or all B cols have been checked
        // either way, all possible repeats accounted for
        // add in length of B row, minus repeats
        C_rs[row] += B_rs[row + 1] - B_rs[row] - num_repeats;
    }
}

template<typename gidx_t, typename lidx_t, typename scalar_t>
__global__ void merge_row_fill( const lidx_t num_rows,
                                const lidx_t *A_rs,
                                const gidx_t *A_cols,
                                const scalar_t *A_coeffs,
                                const lidx_t *B_rs,
                                const gidx_t *B_cols,
                                const scalar_t *B_coeffs,
                                lidx_t *C_rs,
                                gidx_t *C_cols,
                                scalar_t *C_coeffs )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        const auto A_start = A_rs[row], A_len = A_rs[row + 1] - A_start;
        const auto B_start = B_rs[row], B_len = B_rs[row + 1] - B_start;
        const auto C_start = C_rs[row], C_len = C_rs[row + 1] - C_start;
        // all of A counts in automatically
        for ( lidx_t off = 0; off < A_len; ++off ) {
            C_cols[C_start + off]   = A_cols[A_start + off];
            C_coeffs[C_start + off] = A_coeffs[A_start + off];
        }

        lidx_t C_app = A_len, search_start = C_start;
        for ( lidx_t B_ptr = B_start; B_ptr < B_rs[row + 1]; ++B_ptr ) {
            const auto Bc = B_cols[B_ptr];
            const auto Bv = B_coeffs[B_ptr];
            bool matched  = false;
            for ( lidx_t C_ptr = search_start; C_ptr < C_rs[row + 1]; ++C_ptr ) {
                const auto Cc = C_cols[C_ptr];
                if ( Cc == Bc ) {
                    // have a matching column index, add to the current coeff,
                    // flag that a match was found
                    C_coeffs[C_ptr] += Bv;
                    matched = true;
                    // column idxs are ordered, so no need to look at any
                    // entries from here back in later searches
                    search_start = C_ptr + 1;
                    break;
                } else if ( Cc > Bc ) {
                    // C column is larger than the B column we are looking
                    // for, no need to look any further
                    break;
                }
            }
            if ( !matched ) {
                C_cols[C_start + C_app]   = Bc;
                C_coeffs[C_start + C_app] = Bv;
                ++C_app;
            }
        }
    }
}

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::merge( std::shared_ptr<localmatrixdata_t> inL,
                                           std::shared_ptr<localmatrixdata_t> inR,
                                           std::shared_ptr<localmatrixdata_t> out )
{
    PROFILE( "CSRMatrixSpGEMMDevice::merge" );

    // handle special case where either (or both) inputs are empty/null
    if ( inL.get() == nullptr && inR.get() == nullptr ) {
        return;
    }
    if ( inR.get() == nullptr || inR->isEmpty() ) {
        out->swapDataFields( *inL );
        return;
    }
    if ( inL.get() == nullptr || inL->isEmpty() ) {
        out->swapDataFields( *inR );
        return;
    }

    // pull out fields from blocks to merge and row pointers from output
    const auto num_rows = out->numLocalRows();
    AMP_ASSERT( num_rows == inL->numLocalRows() && num_rows == inR->numLocalRows() );
    lidx_t *inL_rs, *inR_rs, *out_rs;
    lidx_t *inL_cols_loc, *inR_cols_loc;
    gidx_t *inL_cols, *inR_cols;
    scalar_t *inL_coeffs, *inR_coeffs;

    std::tie( inL_rs, inL_cols, inL_cols_loc, inL_coeffs ) = inL->getDataFields();
    std::tie( inR_rs, inR_cols, inR_cols_loc, inR_coeffs ) = inR->getDataFields();
    out_rs                                                 = out->getRowStarts();

    // count unique entries in each row
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        merge_row_count<<<GridDim, BlockDim>>>(
            num_rows, inL_rs, inL_cols, inR_rs, inR_cols, out_rs );
        getLastDeviceError( "CSRMatrixSpGEMMDevice::merge::merge_row_count" );
    }

    // trigger allocation of output internals and set up row pointers
    out->setNNZ( true );

    // get fields from output
    lidx_t *out_cols_loc;
    gidx_t *out_cols;
    scalar_t *out_coeffs;
    std::tie( out_rs, out_cols, out_cols_loc, out_coeffs ) = out->getDataFields();

    // fill rows of output as sums of each block
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        merge_row_fill<gidx_t, lidx_t, scalar_t><<<GridDim, BlockDim>>>( num_rows,
                                                                         inL_rs,
                                                                         inL_cols,
                                                                         inL_coeffs,
                                                                         inR_rs,
                                                                         inR_cols,
                                                                         inR_coeffs,
                                                                         out_rs,
                                                                         out_cols,
                                                                         out_coeffs );
        getLastDeviceError( "CSRMatrixSpGEMMDevice::merge::merge_row_count" );
    }
}

} // namespace AMP::LinearAlgebra
