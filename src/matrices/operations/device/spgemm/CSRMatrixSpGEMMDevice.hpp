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
void CSRMatrixSpGEMMDevice<Config>::multiply()
{
    PROFILE( "CSRMatrixSpGEMMDevice::multiply" );

    // start communication to build BRemote before doing anything
    if ( A->hasOffDiag() ) {
        startBRemoteComm();
    }

    C_diag_diag = std::make_shared<localmatrixdata_t>( nullptr,
                                                       C->getMemoryLocation(),
                                                       C->beginRow(),
                                                       C->endRow(),
                                                       C->beginCol(),
                                                       C->endCol(),
                                                       true );
    C_diag_offd = std::make_shared<localmatrixdata_t>( nullptr,
                                                       C->getMemoryLocation(),
                                                       C->beginRow(),
                                                       C->endRow(),
                                                       C->beginCol(),
                                                       C->endCol(),
                                                       false );

    {
        PROFILE( "CSRMatrixSpGEMMDevice::multiply (local)" );
        multiply( A_diag, B_diag, C_diag_diag );
        multiply( A_diag, B_offd, C_diag_offd );
    }

    if ( A->hasOffDiag() ) {
        endBRemoteComm();
        PROFILE( "CSRMatrixSpGEMMDevice::multiply (remote)" );
        if ( BR_diag.get() != nullptr ) {
            C_offd_diag = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               true );
            multiply( A_offd, BR_diag, C_offd_diag );
        }
        if ( BR_offd.get() != nullptr ) {
            C_offd_offd = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               false );
            multiply( A_offd, BR_offd, C_offd_offd );
        }
    }
    deviceSynchronize();

    merge( C_diag_diag, C_offd_diag, C_diag );
    C_diag_diag.reset();
    C_offd_diag.reset();

    merge( C_diag_offd, C_offd_offd, C_offd );
    C_diag_offd.reset();
    C_offd_offd.reset();

    deviceSynchronize();

    C->assemble( true );

    if ( comm.getRank() == 0 ) {
        std::cout << "Device post-assemble";
        C_diag->printStats( true, false );
        C_offd->printStats( true, false );
    }
}

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::multiply( std::shared_ptr<localmatrixdata_t> A_data,
                                              std::shared_ptr<localmatrixdata_t> B_data,
                                              std::shared_ptr<localmatrixdata_t> C_data )
{
    AMP_DEBUG_ASSERT( A_data != nullptr );
    AMP_DEBUG_ASSERT( B_data != nullptr );
    AMP_DEBUG_ASSERT( C_data != nullptr );

    if ( A_data->isEmpty() || B_data->isEmpty() ) {
        return;
    }

    if ( comm.getRank() == 0 ) {
        std::cout << "Device intermediate product inputs";
        A_data->printStats( true, false );
        B_data->printStats( true, false );
    }

    // shapes of A and B
    const auto A_nrows = static_cast<int64_t>( A_data->numLocalRows() );
    const auto A_ncols = static_cast<int64_t>( A_data->numLocalColumns() );
    auto B_ncols       = static_cast<int64_t>( B_data->numLocalColumns() );

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

    if ( comm.getRank() == 0 ) {
        std::cout << "Device intermediate product";
        C_data->printStats( true, false );
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
                                scalar_t *C_coeffs,
                                const bool gz )
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
        // column values are sorted, so walk through A cols and B cols
        // simultaneously to find matches
        lidx_t B_off = 0, C_off = 0, C_app = A_len;
        for ( ; B_off < B_len && C_off < C_len; ) {
            const auto Bc = B_cols[B_start + B_off], Cc = C_cols[C_start + C_off];
            if ( Bc == Cc ) {
                // columns match, add in B coeff and increment offsets
                C_coeffs[C_start + C_off] += B_coeffs[B_start + B_off];
                ++B_off;
                ++C_off;
            } else if ( Cc < Bc ) {
                // C lags B, increment C ptr to check further in row
                ++C_off;
            } else {
                // B lags C, Bc not a repeat, so append to C row and increment
                C_cols[C_start + C_app]   = B_cols[B_start + B_off];
                C_coeffs[C_start + C_app] = B_coeffs[B_start + B_off];
                ++C_app;
                ++B_off;
            }
        }
        if ( gz ) {
            for ( lidx_t off = 0; off < C_len; ++off ) {
                assert( C_cols[C_start + off] > 0 );
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
        C_offd->swapDataFields( *inL );
        return;
    }
    if ( inL.get() == nullptr || inL->isEmpty() ) {
        C_offd->swapDataFields( *inR );
        return;
    }

    // pull out fields from blocks to merge and row pointers from output
    const auto num_rows = out->numLocalRows();
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
        deviceSynchronize();
        merge_row_count<<<GridDim, BlockDim>>>(
            num_rows, inL_rs, inL_cols, inR_rs, inR_cols, out_rs );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixSpGEMMDevice::mergeDiag::merge_row_count" );
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
        bool gz = comm.getRank() == 0 && !out->isDiag();
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        deviceSynchronize();
        merge_row_fill<gidx_t, lidx_t, scalar_t><<<GridDim, BlockDim>>>( num_rows,
                                                                         inL_rs,
                                                                         inL_cols,
                                                                         inL_coeffs,
                                                                         inR_rs,
                                                                         inR_cols,
                                                                         inR_coeffs,
                                                                         out_rs,
                                                                         out_cols,
                                                                         out_coeffs,
                                                                         gz );
        deviceSynchronize();
        getLastDeviceError( "CSRMatrixSpGEMMDevice::mergeDiag::merge_row_count" );
    }

    if ( comm.getRank() == 0 ) {
        std::cout << "Device out merged";
        out->printStats( true, false );
    }
}

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::setupBRemoteComm()
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

    PROFILE( "CSRMatrixSpGEMMDevice::setupBRemoteComm" );

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
    const int TAG = 7800;
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
void CSRMatrixSpGEMMDevice<Config>::startBRemoteComm()
{
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
void CSRMatrixSpGEMMDevice<Config>::endBRemoteComm()
{
    PROFILE( "CSRMatrixSpGEMMDevice::endBRemoteComm" );

    d_recv_matrices = d_csr_comm.recvMatrices( 0, 0, 0, B->numGlobalColumns() );

    // BRemotes do not need any particular parameters object internally
    BR_diag = localmatrixdata_t::ConcatVertical(
        nullptr, d_recv_matrices, B->beginCol(), B->endCol(), true );
    BR_offd = localmatrixdata_t::ConcatVertical(
        nullptr, d_recv_matrices, B->beginCol(), B->endCol(), false );

    // trigger remotes to build local indices
    BR_diag->globalToLocalColumns();
    BR_offd->globalToLocalColumns();


    if ( comm.getRank() == 0 ) {
        std::cout << "Device BRemote blocks";
        BR_diag->printStats( true, false );
        BR_offd->printStats( true, false );
    }

    // comms are done and BR_{diag,offd} filled, deallocate send/recv blocks
    d_send_matrices.clear();
    d_recv_matrices.clear();
}

} // namespace AMP::LinearAlgebra
