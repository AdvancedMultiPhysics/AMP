#ifndef included_AMP_CSRMatrixCommunicator_hpp
#define included_AMP_CSRMatrixCommunicator_hpp

#include "AMP/matrices/data/CSRMatrixCommunicator.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixCommunicator<Config>::sendMatrices(
    const std::map<int, std::shared_ptr<localmatrixdata_t>> &matrices )
{
    PROFILE( "CSRMatrixCommunicator::sendMatrices" );

    if ( MIGRATE_DEV ) {
        migrateToHost( matrices );
        d_migrate_comm->sendMatrices( d_send_mat_migrate );
        d_send_called = true;
        return;
    }

    // At present we allow that the held communication list refer to a
    // super-set of the communications that need to be sent. First count
    // how many sources we actually expect
    countSources( matrices );

    // post all of the sends for the matrices
    for ( auto it : matrices ) {
        const int dest     = it.first;
        auto matrix        = it.second;
        const auto num_rs  = matrix->d_num_rows + 1;
        const auto num_nnz = matrix->d_nnz;
        AMP_DEBUG_ASSERT( !matrix->isEmpty() );
        d_send_requests.emplace_back(
            d_comm.Isend( matrix->d_row_starts.get(), num_rs, dest, ROW_TAG ) );
        d_send_requests.emplace_back(
            d_comm.Isend( matrix->d_cols.get(), num_nnz, dest, COL_TAG ) );
        d_send_requests.emplace_back(
            d_comm.Isend( matrix->d_coeffs.get(), num_nnz, dest, COEFF_TAG ) );
    }
    d_send_called = true;
}

template<typename Config>
void CSRMatrixCommunicator<Config>::countSources(
    const std::map<int, std::shared_ptr<localmatrixdata_t>> &matrices )
{
    PROFILE( "CSRMatrixCommunicator::countSources" );

    // verify that send list actually contains all destinations
    for ( [[maybe_unused]] const auto &it : matrices ) {
        AMP_DEBUG_INSIST( std::find( d_allowed_dest.begin(), d_allowed_dest.end(), it.first ) !=
                              d_allowed_dest.end(),
                          "CSRMatrixCommunicator invalid destination" );
    }

    // to count sources send an empty message to every rank in our
    // send-list with tag COMM_USED if we will actually communicate
    // with them later and tag COMM_UNUSED otherwise
    std::vector<AMP_MPI::Request> count_dest_reqs;
    std::vector<int> dest_used( d_allowed_dest.size() );
    for ( size_t n = 0; n < d_allowed_dest.size(); ++n ) {
        const auto r = d_allowed_dest[n];
        dest_used[n] = matrices.count( r ) > 0 ? 1 : 0;
        count_dest_reqs.push_back( d_comm.Isend( &dest_used[n], 1, r, COMM_TEST ) );
    }

    // Similarly, look for messages from all in our recv-list to tell
    // us what comms will happen.
    d_num_sources = 0;
    for ( int n = 0; n < d_num_allowed_sources; ++n ) {
        auto [source, tag, num_bytes] = d_comm.probe( -1, COMM_TEST );
        AMP_DEBUG_ASSERT( tag == COMM_TEST );
        int result = 0;
        d_comm.recv( &result, 1, source, COMM_TEST );
        if ( result == 1 ) {
            d_num_sources++;
        }
    }

    // wait out the sends and return
    if ( count_dest_reqs.size() > 0 ) {
        d_comm.waitAll( static_cast<int>( count_dest_reqs.size() ), count_dest_reqs.data() );
    }
}

template<typename Config>
std::map<int, std::shared_ptr<CSRLocalMatrixData<Config>>>
CSRMatrixCommunicator<Config>::recvMatrices( typename Config::gidx_t first_row,
                                             typename Config::gidx_t last_row,
                                             typename Config::gidx_t first_col,
                                             typename Config::gidx_t last_col )
{
    PROFILE( "CSRMatrixCommunicator::recvMatrices" );

    using lidx_t = typename Config::lidx_t;
    using gidx_t = typename Config::gidx_t;

    AMP_INSIST( d_send_called,
                "CSRMatrixCommunicator::sendMatrices must be called before recvMatrices" );

    if ( MIGRATE_DEV ) {
        const auto host_blocks =
            d_migrate_comm->recvMatrices( first_row, last_row, first_col, last_col );
        d_send_called = false;
        d_send_mat_migrate.clear();
        return migrateFromHost( host_blocks );
    }

    std::map<int, std::shared_ptr<localmatrixdata_t>> blocks;
    const auto mem_loc = AMP::Utilities::getAllocatorMemoryType<allocator_type>();

    // there are d_num_sources matrices to receive
    // always sent in order row_starts, cols, coeffs
    // start with probe on any source with ROW_TAG
    for ( int ns = 0; ns < d_num_sources; ++ns ) {
        auto [source, tag, num_bytes] = d_comm.probe( -1, ROW_TAG );
        AMP_ASSERT( tag == ROW_TAG );
        // remember row_starts has extra entry
        const lidx_t num_rows = ( num_bytes / sizeof( lidx_t ) ) - 1;
        // if last_row is zero then choose based on num_rows,
        // otherwise test that recv'd matrix is valid with layout
        gidx_t fr, lr;
        if ( last_row == 0 ) {
            fr = 0;
            lr = static_cast<gidx_t>( num_rows );
        } else {
            AMP_INSIST( num_rows == static_cast<lidx_t>( last_row - first_row ),
                        "Received matrix with invalid layout" );
            fr = first_row;
            lr = last_row;
        }
        auto [it, inserted] =
            blocks.insert( { source,
                             std::make_shared<localmatrixdata_t>(
                                 nullptr, mem_loc, fr, lr, first_col, last_col, false ) } );
        AMP_ASSERT( inserted );
        auto block = ( *it ).second;
        // matrix now exists and has row_starts buffer, recv it and trigger allocations
        d_comm.recv( block->d_row_starts.get(), num_rows + 1, source, ROW_TAG );
        block->setNNZ( false );
        // buffers for cols and coeffs now allocated, recv them and continue to next probe
        d_comm.recv( block->d_cols.get(), block->d_nnz, source, COL_TAG );
        d_comm.recv( block->d_coeffs.get(), block->d_nnz, source, COEFF_TAG );
    }

    // ensure that any outstanding sends complete
    if ( d_send_requests.size() > 0 ) {
        d_comm.waitAll( static_cast<int>( d_send_requests.size() ), d_send_requests.data() );
        d_send_requests.clear();
    }

    // comm done, reset send flag in case this gets re-used
    d_send_called = false;

    return blocks;
}

template<typename Config>
void CSRMatrixCommunicator<Config>::migrateToHost(
    const std::map<int, std::shared_ptr<CSRLocalMatrixData<Config>>> &matrices )
{
    for ( auto it : matrices ) {
        const int dest = it.first;
        auto matrix    = it.second;
        d_send_mat_migrate.insert( { dest, matrix->template migrate<ConfigHost>() } );
    }
}

template<typename Config>
std::map<int, std::shared_ptr<CSRLocalMatrixData<Config>>>
CSRMatrixCommunicator<Config>::migrateFromHost(
    const std::map<
        int,
        std::shared_ptr<CSRLocalMatrixData<typename Config::template set_alloc<alloc::host>::type>>>
        &matrices )
{
    std::map<int, std::shared_ptr<localmatrixdata_t>> blocks;
    for ( auto it : matrices ) {
        const int src = it.first;
        auto matrix   = it.second;
        blocks.insert( { src, matrix->template migrate<Config>() } );
    }
    return blocks;
}

} // namespace AMP::LinearAlgebra

#endif
