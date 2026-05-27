#include "AMP/IO/PIO.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_DEVICE
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/transform.h>
    #define AMP_FUNCTION_HD __host__ __device__
#else
    #define AMP_FUNCTION_HD
#endif

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixSpGEMMCommon<Config>::multiply()
{
    PROFILE( "CSRMatrixSpGEMMCommon::multiply" );

    // start communication to build BRemote before doing anything
    if ( A->hasOffDiag() ) {
        startBRemoteComm();
    }

    C_diag_diag = std::make_shared<localmatrixdata_t>(
        nullptr, C->beginRow(), C->endRow(), C->beginCol(), C->endCol(), true );
    C_diag_offd = std::make_shared<localmatrixdata_t>(
        nullptr, C->beginRow(), C->endRow(), C->beginCol(), C->endCol(), false );

    {
        PROFILE( "CSRMatrixSpGEMMCommon::multiply (local)" );
        multiplyLocal( A_diag, B_diag, C_diag_diag );
        multiplyLocal( A_diag, B_offd, C_diag_offd );
    }

    if ( A->hasOffDiag() ) {
        endBRemoteComm();
        PROFILE( "CSRMatrixSpGEMMCommon::multiply (remote)" );
        if ( BR_diag.get() != nullptr ) {
            C_offd_diag = std::make_shared<localmatrixdata_t>(
                nullptr, C->beginRow(), C->endRow(), C->beginCol(), C->endCol(), true );
            multiplyLocal( A_offd, BR_diag, C_offd_diag );
            merge( C_diag_diag, C_offd_diag, C_diag );
            C_diag_diag.reset();
            C_offd_diag.reset();
        } else {
            C_diag->swapDataFields( *C_diag_diag );
        }
        if ( BR_offd.get() != nullptr ) {
            C_offd_offd = std::make_shared<localmatrixdata_t>(
                nullptr, C->beginRow(), C->endRow(), C->beginCol(), C->endCol(), false );
            multiplyLocal( A_offd, BR_offd, C_offd_offd );
            merge( C_diag_offd, C_offd_offd, C_offd );
            C_diag_offd.reset();
            C_offd_offd.reset();
        } else {
            C_offd->swapDataFields( *C_diag_offd );
        }
    } else {
        C_diag->swapDataFields( *C_diag_diag );
        C_offd->swapDataFields( *C_diag_offd );
    }

    C->assemble( true );
}

#if 0
template<typename gidx_t, typename lidx_t>
AMP_FUNCTION_HD void merge_row_count( const lidx_t row,
                                      const lidx_t *A_rs,
                                      const gidx_t *A_cols,
                                      const lidx_t *B_rs,
                                      const gidx_t *B_cols,
                                      lidx_t *C_rs )
{
    const auto A_start = A_rs[row];
    const auto A_len   = A_rs[row + 1] - A_start;
    const auto B_start = B_rs[row];

    C_rs[row] = A_len;
    for ( lidx_t B_cur = B_start; B_cur < B_rs[row + 1]; ++B_cur ) {
        const auto Bc = B_cols[B_cur];
        bool matched  = false;
        for ( lidx_t A_cur = A_start; A_cur < A_rs[row + 1]; ++A_cur ) {
            const auto Ac = A_cols[A_cur];
            if ( Ac == Bc ) {
                matched = true;
                break;
            }
        }
        if ( !matched ) {
            C_rs[row]++;
        }
    }
}

template<typename gidx_t, typename lidx_t, typename scalar_t>
AMP_FUNCTION_HD void merge_row_fill( const lidx_t row,
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
    const auto A_start = A_rs[row];
    const auto A_len   = A_rs[row + 1] - A_start;
    const auto B_start = B_rs[row];
    const auto C_start = C_rs[row], C_search_end = C_start + A_len;
    // all of A counts in automatically
    for ( lidx_t off = 0; off < A_len; ++off ) {
        C_cols[C_start + off]   = A_cols[A_start + off];
        C_coeffs[C_start + off] = A_coeffs[A_start + off];
    }

    lidx_t C_app = A_len;
    for ( lidx_t B_cur = B_start; B_cur < B_rs[row + 1]; ++B_cur ) {
        const auto Bc = B_cols[B_cur];
        const auto Bv = B_coeffs[B_cur];
        bool matched  = false;
        for ( lidx_t C_cur = C_start; C_cur < C_search_end; ++C_cur ) {
            const auto Cc = C_cols[C_cur];
            if ( Cc == Bc ) {
                C_coeffs[C_cur] += Bv;
                matched = true;
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
#else
template<typename gidx_t, typename lidx_t, typename scalar_t>
AMP_FUNCTION_HD void merge_row_count( const lidx_t row,
                                      const lidx_t *A_rs,
                                      const gidx_t *A_cols,
                                      const scalar_t *A_coeffs,
                                      const lidx_t *B_rs,
                                      const gidx_t *B_cols,
                                      const scalar_t *B_coeffs,
                                      lidx_t *C_rs )
{
    const auto A_start = A_rs[row], A_end = A_rs[row + 1];
    const auto B_start = B_rs[row], B_end = B_rs[row + 1];

    // Count only actual non-zeros from A
    for ( lidx_t A_cur = A_start; A_cur < A_end; ++A_cur ) {
        const auto Av = A_coeffs[A_cur];
        if ( Av != 0 ) {
            C_rs[row]++;
        }
    }

    // Count from B that are non-zero and don't overlap
    // non-zeros from A
    for ( lidx_t B_cur = B_start; B_cur < B_end; ++B_cur ) {
        const auto Bv = B_coeffs[B_cur];
        if ( Bv == 0 ) {
            continue;
        }
        const auto Bc = B_cols[B_cur];
        bool matched  = false;
        for ( lidx_t A_cur = A_start; A_cur < A_end; ++A_cur ) {
            const auto Av = A_coeffs[A_cur];
            if ( Av == 0 ) {
                continue;
            }
            const auto Ac = A_cols[A_cur];
            if ( Ac == Bc ) {
                matched = true;
                break;
            }
        }
        if ( !matched ) {
            C_rs[row]++;
        }
    }
}

template<typename gidx_t, typename lidx_t, typename scalar_t>
AMP_FUNCTION_HD void merge_row_fill( const lidx_t row,
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
    const auto A_start = A_rs[row], A_end = A_rs[row + 1];
    const auto B_start = B_rs[row], B_end = B_rs[row + 1];
    const auto C_start = C_rs[row];
    lidx_t C_app       = C_start;

    // Insert only actual non-zeros from A
    for ( lidx_t A_cur = A_start; A_cur < A_end; ++A_cur ) {
        const auto Av = A_coeffs[A_cur];
        if ( Av != 0 ) {
            C_cols[C_app]   = A_cols[A_cur];
            C_coeffs[C_app] = Av;
            ++C_app;
        }
    }

    // Add in entries from B that are non-zero
    // either summing into existing position or appending
    const auto C_search_end = C_app;
    for ( lidx_t B_cur = B_start; B_cur < B_end; ++B_cur ) {
        const auto Bv = B_coeffs[B_cur];
        if ( Bv == 0 ) {
            continue;
        }
        const auto Bc = B_cols[B_cur];
        bool matched  = false;
        for ( lidx_t C_cur = C_start; C_cur < C_search_end; ++C_cur ) {
            const auto Cc = C_cols[C_cur];
            if ( Cc == Bc ) {
                C_coeffs[C_cur] += Bv;
                matched = true;
                break;
            }
        }
        if ( !matched ) {
            C_cols[C_app]   = Bc;
            C_coeffs[C_app] = Bv;
            ++C_app;
        }
    }
}
#endif

template<typename Config>
void CSRMatrixSpGEMMCommon<Config>::merge( std::shared_ptr<localmatrixdata_t> inL,
                                           std::shared_ptr<localmatrixdata_t> inR,
                                           std::shared_ptr<localmatrixdata_t> out )
{
    PROFILE( "CSRMatrixSpGEMMCommon::merge" );

    AMP_ASSERT( inL && inR && out );
    // handle special case where either (or both) inputs are empty/null
    if ( inL->isEmpty() && inR->isEmpty() ) {
        return;
    }
    if ( inR->isEmpty() ) {
        out->swapDataFields( *inL );
        return;
    }
    if ( inL->isEmpty() ) {
        out->swapDataFields( *inR );
        return;
    }

    // pull out fields from blocks to merge and row pointers from output
    const auto num_rows = out->numLocalRows();
    AMP_ASSERT( num_rows == inL->numLocalRows() && num_rows == inR->numLocalRows() );
    lidx_t *inL_rs, *inR_rs, *out_rs;
    gidx_t *inL_cols, *inR_cols;
    scalar_t *inL_coeffs, *inR_coeffs;

    std::tie( inL_rs, inL_cols, std::ignore, inL_coeffs ) = inL->getDataFields();
    std::tie( inR_rs, inR_cols, std::ignore, inR_coeffs ) = inR->getDataFields();
    out_rs                                                = out->getRowStarts();

    // count unique entries in each row
    {
        auto merge_row_count_all =
            [inL_rs, inL_cols, inL_coeffs, inR_rs, inR_cols, inR_coeffs, out_rs] AMP_FUNCTION_HD(
                const lidx_t row ) -> void {
            merge_row_count(
                row, inL_rs, inL_cols, inL_coeffs, inR_rs, inR_cols, inR_coeffs, out_rs );
        };
        if constexpr ( !alloc_info<Config::allocator>::device_accessible ) {
            for ( lidx_t row = 0; row < num_rows; ++row ) {
                merge_row_count_all( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( num_rows ),
                              merge_row_count_all );
            getLastDeviceError( "CSRMatrixSpGEMMCommon::merge::merge_row_count" );
#else
            AMP_ERROR( "CSRMatrixSpGEMMCommon::merge Unrecognized memory space" );
#endif
        }
    }

    // trigger allocation of output internals and set up row pointers
    out->setNNZ( true );

    // get fields from output
    gidx_t *out_cols;
    scalar_t *out_coeffs;
    std::tie( out_rs, out_cols, std::ignore, out_coeffs ) = out->getDataFields();

    // fill rows of output as sums of each block
    {
        auto merge_row_fill_all = [inL_rs,
                                   inL_cols,
                                   inL_coeffs,
                                   inR_rs,
                                   inR_cols,
                                   inR_coeffs,
                                   out_rs,
                                   out_cols,
                                   out_coeffs] AMP_FUNCTION_HD( const lidx_t row ) -> void {
            merge_row_fill( row,
                            inL_rs,
                            inL_cols,
                            inL_coeffs,
                            inR_rs,
                            inR_cols,
                            inR_coeffs,
                            out_rs,
                            out_cols,
                            out_coeffs );
        };
        if constexpr ( !alloc_info<Config::allocator>::device_accessible ) {
            for ( lidx_t row = 0; row < num_rows; ++row ) {
                merge_row_fill_all( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( num_rows ),
                              merge_row_fill_all );
            getLastDeviceError( "CSRMatrixSpGEMMCommon::merge::merge_row_fill" );
#else
            AMP_ERROR( "CSRMatrixSpGEMMCommon::merge Unrecognized memory space" );
#endif
        }
    }
}

template<typename Config>
void CSRMatrixSpGEMMCommon<Config>::setupBRemoteComm()
{
    /*
     * Setting up the comms is somewhat involved. A high level overview
     * of the steps involved is:
     * 1. Collect comm list info and needed remote rows
     * 2. Trim down lists from steps 3 and 4 to ranks that are actually needed
     * 3. Record which specific rows are needed from each process
     * 4. Send row ids from 6 to owners of those rows

     * NOTES:
     *  Step 4 uses non-blocking recvs and blocking sends.
     */

    PROFILE( "CSRMatrixSpGEMMCommon::setupBRemoteComm" );

    using lidx_t = typename Config::lidx_t;

    auto comm_size = comm.getSize();

    // 1. Query comm list info and get offd colmap
    auto comm_list            = A->getRightCommList();
    auto rows_per_rank_recv   = comm_list->getReceiveSizes();
    auto rows_per_rank_send   = comm_list->getSendSizes();
    auto B_last_rows          = comm_list->getPartition();
    const auto A_col_map_size = A_offd->numUniqueColumns();
    std::vector<gidx_t> A_col_map;
    A_offd->getColumnMap( A_col_map );

    // 2. the above rows per rank lists generally include lots of zeros
    // trim down to the ranks that actually need to communicate
    int total_send = 0, total_recv = 0;
    for ( int r = 0; r < comm_size; ++r ) {
        const auto nsend = rows_per_rank_send[r];
        if ( nsend > 0 ) {
            d_dest_info.insert( std::make_pair( r, SpGEMMCommInfo( nsend ) ) );
        }
        const auto nrecv = rows_per_rank_recv[r];
        if ( nrecv > 0 ) {
            d_src_info.insert( std::make_pair( r, SpGEMMCommInfo( nrecv ) ) );
        }
        total_send += nsend;
        total_recv += nrecv;
    }

    // 3. Scan over column map now writing into the trimmed down src list
    for ( lidx_t n = 0; n < A_col_map_size; ++n ) {
        const auto col = static_cast<std::size_t>( A_col_map[n] );
        int owner      = -1;
        if ( col < B_last_rows[0] ) {
            owner = 0;
        } else {
            for ( int r = 1; r < comm_size; ++r ) {
                auto rs = B_last_rows[r - 1], re = B_last_rows[r];
                if ( rs <= col && col < re ) {
                    owner = r;
                    break;
                }
            }
        }
        d_src_info[owner].rowids.push_back( col );
    }

    // 4. send rowids to their owners
    // start by posting the irecvs
    const int TAG = comm.newTag();
    std::vector<AMP_MPI::Request> irecvs;
    for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
        it->second.rowids.resize( it->second.numrow );
        irecvs.push_back(
            comm.Irecv( it->second.rowids.data(), it->second.numrow, it->first, TAG ) );
    }
    // now send all the rows we want from other ranks
    for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
        comm.send( it->second.rowids.data(), it->second.numrow, it->first, TAG );
    }
    // wait for receives to finish
    comm.waitAll( static_cast<int>( irecvs.size() ), irecvs.data() );
}

template<typename Config>
void CSRMatrixSpGEMMCommon<Config>::startBRemoteComm()
{
    if ( comm.getSize() == 1 ) {
        return;
    }

    PROFILE( "CSRMatrixSpGEMMCommon::startBRemoteComm" );

    // check if the communicator information is available and create if needed
    if ( d_dest_info.empty() ) {
        setupBRemoteComm();
    }

    // subset matrices by rows that other ranks need and send them out
    for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
        auto block = B->subsetRows( it->second.rowids );
        d_send_matrices.insert( { it->first, block } );
    }
    d_csr_comm.sendMatrices( d_send_matrices );
}

template<typename Config>
void CSRMatrixSpGEMMCommon<Config>::endBRemoteComm()
{
    if ( comm.getSize() == 1 ) {
        return;
    }

    PROFILE( "CSRMatrixSpGEMMCommon::endBRemoteComm" );

    d_recv_matrices = d_csr_comm.recvMatrices( 0, 0, 0, B->numGlobalColumns() );

    if ( d_recv_matrices.size() > 0 ) {
        // BRemotes do not need any particular parameters object internally
        BR_diag = localmatrixdata_t::ConcatVertical(
            nullptr, d_recv_matrices, B->beginCol(), B->endCol(), true );
        BR_offd = localmatrixdata_t::ConcatVertical(
            nullptr, d_recv_matrices, B->beginCol(), B->endCol(), false );
    }
    // comms are done and BR_{diag,offd} filled, deallocate send/recv blocks
    d_send_matrices.clear();
    d_recv_matrices.clear();

    // trigger remotes to build local indices
    // and test shape of concatenated matrices
    if ( BR_diag ) {
        BR_diag->globalToLocalColumns();
        AMP_DEBUG_ASSERT( A_offd->numUniqueColumns() == BR_diag->numLocalRows() );
    }
    if ( BR_offd ) {
        BR_offd->globalToLocalColumns();
        AMP_DEBUG_ASSERT( A_offd->numUniqueColumns() == BR_offd->numLocalRows() );
    }
}

} // namespace AMP::LinearAlgebra

#undef AMP_FUNCTION_HD
