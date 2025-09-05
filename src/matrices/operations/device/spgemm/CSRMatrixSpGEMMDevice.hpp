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
        endBRemoteComm();
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
        multiply<BlockType::DIAG>( A_diag, B_diag, C_diag_diag );
        multiply<BlockType::OFFD>( A_diag, B_offd, C_diag_offd );
    }

    if ( A->hasOffDiag() ) {
        PROFILE( "CSRMatrixSpGEMMDevice::multiply (remote)" );
        if ( BR_diag.get() != nullptr ) {
            C_offd_diag = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               true );
            multiply<BlockType::DIAG>( A_offd, BR_diag, C_offd_diag );
        }
        if ( BR_offd.get() != nullptr ) {
            C_offd_offd = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               false );
            multiply<BlockType::OFFD>( A_offd, BR_offd, C_offd_offd );
        }
    }
    deviceSynchronize();

    mergeDiag();
    mergeOffd();
    deviceSynchronize();

    C->assemble();
}

template<typename Config>
template<typename CSRMatrixSpGEMMDevice<Config>::BlockType block_t>
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

    // shapes of A and B
    const auto A_nrows = static_cast<int64_t>( A_data->numLocalRows() );
    const auto A_ncols = static_cast<int64_t>( A_data->numLocalColumns() );
    const auto B_ncols = static_cast<int64_t>( B_data->numLocalColumns() );

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

    // specific SpGEMM types and inputs depend on bloct type of A
    // if A is diag do everything with lidx_t indices, otherwise
    // need gidx_t columns
    if constexpr ( block_t == BlockType::DIAG ) {
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
    } else {
        // For A being offd still want entries from A_cols_loc, but need
        // to cast them up to gidx_t's
        using llAllocator_t =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<long long>;
        llAllocator_t alloc;
        long long *A_cols_loc_casted = alloc.allocate( A_nnz );
        AMP::Utilities::copy( A_nnz, A_cols_loc, A_cols_loc_casted );

        // proceed as above, using gidx_t cols throughout
        VendorSpGEMM<lidx_t, long long, scalar_t> spgemm( A_nrows,
                                                          B_ncols,
                                                          A_ncols,
                                                          A_nnz,
                                                          A_rs,
                                                          A_cols_loc_casted,
                                                          A_coeffs,
                                                          B_nnz,
                                                          B_rs,
                                                          reinterpret_cast<long long *>( B_cols ),
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
        spgemm.compute( C_rs, reinterpret_cast<long long *>( C_cols ), C_coeffs );

        // free upcasted A_cols_loc
        alloc.deallocate( A_cols_loc_casted, A_nnz );
    }

    // exiting function destructs spgemm wrapper and frees its internals
}

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::mergeDiag()
{
    PROFILE( "CSRMatrixSpGEMMDevice::mergeDiag" );

    const auto first_col = C_diag->beginCol();

    // handle special case where C_diag_offd is empty
    if ( C_diag_offd.get() == nullptr || C_diag_offd->isEmpty() ) {
        C_diag->swapDataFields( *C_diag_diag );
        return;
    }

    AMP_ERROR( "CSRMatrixSpGEMMDevice::mergeDiag: Not implemented yet" );

    // pull out fields from blocks to merge and row pointers from C_diag
    lidx_t *C_dd_rs, *C_od_rs, *C_rs;
    lidx_t *C_dd_cols_loc, *C_od_cols_loc;
    gidx_t *C_dd_cols, *C_od_cols;
    scalar_t *C_dd_coeffs, *C_od_coeffs;

    std::tie( C_dd_rs, C_dd_cols, C_dd_cols_loc, C_dd_coeffs ) = C_diag_diag->getDataFields();
    std::tie( C_od_rs, C_od_cols, C_od_cols_loc, C_od_coeffs ) = C_offd_diag->getDataFields();
    C_rs                                                       = C_diag->getRowStarts();

    C_diag_diag.reset();
    C_offd_diag.reset();
}

template<typename Config>
void CSRMatrixSpGEMMDevice<Config>::mergeOffd()
{
    PROFILE( "CSRMatrixSpGEMMDevice::mergeOffd" );

    // handle special case where either C_diag_offd or C_offd_offd is empty
    if ( C_diag_offd.get() == nullptr && C_offd_offd.get() == nullptr ) {
        return;
    }
    if ( C_offd_offd.get() == nullptr || C_offd_offd->isEmpty() ) {
        C_offd->swapDataFields( *C_diag_offd );
        return;
    }
    if ( C_diag_offd.get() == nullptr || C_diag_offd->isEmpty() ) {
        C_offd->swapDataFields( *C_offd_offd );
        return;
    }

    AMP_ERROR( "CSRMatrixSpGEMMDevice::mergeOffd: Not implemented yet" );

    // pull out fields from blocks to merge and row pointers from C_offd
    lidx_t *C_do_rs, *C_oo_rs, *C_rs;
    lidx_t *C_do_cols_loc, *C_oo_cols_loc;
    gidx_t *C_do_cols, *C_oo_cols;
    scalar_t *C_do_coeffs, *C_oo_coeffs;

    std::tie( C_do_rs, C_do_cols, C_do_cols_loc, C_do_coeffs ) = C_diag_offd->getDataFields();
    std::tie( C_oo_rs, C_oo_cols, C_oo_cols_loc, C_oo_coeffs ) = C_offd_offd->getDataFields();
    C_rs                                                       = C_offd->getRowStarts();

    C_diag_offd.reset();
    C_offd_offd.reset();
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
    const auto A_col_map      = A_offd->getColumnMap();

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

    // comms are done and BR_{diag,offd} filled, deallocate send/recv blocks
    d_send_matrices.clear();
    d_recv_matrices.clear();
}

} // namespace AMP::LinearAlgebra
