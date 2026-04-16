#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/operations/spgemm/CSRMatrixSpGEMMCommon.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"

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
        PROFILE( "CSRMatrixSpGEMMCommon::multiply (local)" );
        multiplyLocal( A_diag, B_diag, C_diag_diag );
        multiplyLocal( A_diag, B_offd, C_diag_offd );
    }

    if ( A->hasOffDiag() ) {
        endBRemoteComm();
        PROFILE( "CSRMatrixSpGEMMCommon::multiply (remote)" );
        if ( BR_diag.get() != nullptr ) {
            C_offd_diag = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               true );
            multiplyLocal( A_offd, BR_diag, C_offd_diag );
        }
        if ( BR_offd.get() != nullptr ) {
            C_offd_offd = std::make_shared<localmatrixdata_t>( nullptr,
                                                               C->getMemoryLocation(),
                                                               C->beginRow(),
                                                               C->endRow(),
                                                               C->beginCol(),
                                                               C->endCol(),
                                                               false );
            multiplyLocal( A_offd, BR_offd, C_offd_offd );
        }
    }

    merge( C_diag_diag, C_offd_diag, C_diag );
    C_diag_diag.reset();
    C_offd_diag.reset();

    merge( C_diag_offd, C_offd_offd, C_offd );
    C_diag_offd.reset();
    C_offd_offd.reset();

    C->assemble( true );
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
    BR_diag->globalToLocalColumns();
    BR_offd->globalToLocalColumns();

    // test shape of concatenated matrices
    if ( BR_diag ) {
        AMP_DEBUG_ASSERT( A_offd->numUniqueColumns() == BR_diag->numLocalRows() );
    }
    if ( BR_offd ) {
        AMP_DEBUG_ASSERT( A_offd->numUniqueColumns() == BR_offd->numLocalRows() );
    }
}

} // namespace AMP::LinearAlgebra
