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

} // namespace AMP::LinearAlgebra
