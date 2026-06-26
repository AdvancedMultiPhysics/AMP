#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/operations/kokkos/spgemm/CSRMatrixSpGEMMKokkos.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
#endif

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config, class ExecSpace>
void CSRMatrixSpGEMMKokkos<Config, ExecSpace>::multiplyLocal(
    std::shared_ptr<localmatrixdata_t> A_data,
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

    // all fields from blocks involved as spgemm compatible views
    auto [A_rowmap, A_entries, A_values] = wrapDataFields( A_data );
    auto [B_rowmap, B_entries, B_values] = wrapDataFields( B_data );

    // set up kokkos-kernels handle for state persistent across symbolic/numeric phases
    handle_t handle;
    handle.set_team_work_size( 16 );
    handle.set_dynamic_scheduling( true );
    handle.create_spgemm_handle( KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK_MEMORY );

    {
        // C has row pointers allocated but unfilled
        // wrap it for symbolic call but scope it so that we can
        // wrap all C fields after setting NNZ
        rowmap_t C_rowmap( C_data->getRowStarts(), A_nrows + 1 );
        KokkosSparse::spgemm_symbolic( &handle,
                                       A_nrows,
                                       B_ncols,
                                       A_ncols,
                                       A_rowmap,
                                       A_entries,
                                       false,
                                       B_rowmap,
                                       B_entries,
                                       false,
                                       C_rowmap,
                                       false );
    }

    // Get nnz for C and allocate internals
    const lidx_t C_nnz = handle.get_spgemm_handle()->get_c_nnz();
    C_data->setNNZ( C_nnz );

    // pull out the now allocated C internals
    auto [C_rowmap, C_entries, C_values] = wrapDataFields( C_data );

    // numeric phase
    KokkosSparse::spgemm_numeric( &handle,
                                  A_nrows,
                                  B_ncols,
                                  A_ncols,
                                  A_rowmap,
                                  A_entries,
                                  A_values,
                                  false,
                                  B_rowmap,
                                  B_entries,
                                  B_values,
                                  false,
                                  C_rowmap,
                                  C_entries,
                                  C_values );

    // can now free handle internals
    handle.destroy_spgemm_handle();

    // Convert the local indices to globals to make merges easier
    lidx_t *C_rs = nullptr, *C_cols_loc = nullptr;
    gidx_t *C_cols                                 = nullptr;
    scalar_t *C_coeffs                             = nullptr;
    std::tie( C_rs, C_cols, C_cols_loc, C_coeffs ) = C_data->getDataFields();
    if ( C_data->isDiag() ) {


        const auto first_col = C_data->beginCol();

        if ( !alloc_info<Config::allocator>::device_accessible ) {
            std::transform(
                C_cols_loc, C_cols_loc + C_nnz, C_cols, [first_col]( const lidx_t lc ) -> gidx_t {
                    return static_cast<gidx_t>( lc ) + first_col;
                } );
        } else {
#ifdef AMP_USE_DEVICE
            thrust::transform( thrust::device.on( Utilities::device_context_default.stream ),
                               C_cols_loc,
                               C_cols_loc + C_nnz,
                               C_cols,
                               [first_col] __device__( const lidx_t lc ) -> gidx_t {
                                   return static_cast<gidx_t>( lc ) + first_col;
                               } );
#else
            AMP_ERROR( "CSRMatrixSpGEMMKokkos::multiply: Unrecognized memory space" );
#endif
        }
    } else {
        const auto colmap = B_data->getColumnMap();
        if ( !alloc_info<Config::allocator>::device_accessible ) {
            std::transform( C_cols_loc,
                            C_cols_loc + C_nnz,
                            C_cols,
                            [colmap]( const lidx_t lc ) -> gidx_t { return colmap[lc]; } );
        } else {
#ifdef AMP_USE_DEVICE
            thrust::transform(
                thrust::device.on( Utilities::device_context_default.stream ),
                C_cols_loc,
                C_cols_loc + C_nnz,
                C_cols,
                [colmap] __device__( const lidx_t lc ) -> gidx_t { return colmap[lc]; } );
#else
            AMP_ERROR( "CSRMatrixSpGEMMKokkos::multiply: Unrecognized memory space" );
#endif
        }
    }
}

} // namespace AMP::LinearAlgebra
