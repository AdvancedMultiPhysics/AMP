#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.h"

#include "ProfilerApp.h"

#include <iostream>
#include <map>
#include <set>

namespace AMP::LinearAlgebra {

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::symbolicMultiply()
{
    PROFILE( "CSRMatrixSpGEMMDefault::symbolicMultiply" );

    using lidx_t = typename Policy::lidx_t;
    using gidx_t = typename Policy::gidx_t;

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    auto A_diag = A->getDiagMatrix();
    auto A_offd = A->getOffdMatrix();
    auto B_diag = B->getDiagMatrix();
    auto B_offd = B->getOffdMatrix();

    const auto col_diag_start = C->beginCol();
    const auto col_diag_end   = C->endCol();

    // vector of sets for each row makes zipping on/off process patterns
    // together easier. More memory intensive though...
    std::vector<std::set<gidx_t>> C_cols_diag( nRows, std::set<gidx_t>() );
    std::vector<std::set<gidx_t>> C_cols_offd( nRows, std::set<gidx_t>() );

    // Process diagonal block of A acting on whole local part of B
    {
        PROFILE( "CSRMatrixSpGEMMDefault::symbolicMultiply (local)" );
        symbolicMultiply( A_diag, B_diag, col_diag_start, col_diag_end, true, C_cols_diag );
        if ( B->hasOffDiag() ) {
            symbolicMultiply( A_diag, B_offd, col_diag_start, col_diag_end, false, C_cols_offd );
        }
    }

    // off-diagonal block requires fetching non-local rows of B
    if ( A->hasOffDiag() ) {
        createBRemoteSymbolic();
        symbolicMultiply( A_offd, BRemote, col_diag_start, col_diag_end, true, C_cols_diag );
        symbolicMultiply( A_offd, BRemote, col_diag_start, col_diag_end, false, C_cols_offd );
    }

    // count non-zeros per row
    std::vector<lidx_t> nnz_diag( nRows, 0 ), nnz_offd( nRows, 0 );
    lidx_t total_nnz_diag = 0, total_nnz_offd = 0;
    for ( lidx_t row = 0; row < nRows; ++row ) {
        nnz_diag[row] = static_cast<lidx_t>( C_cols_diag[row].size() );
        nnz_offd[row] = static_cast<lidx_t>( C_cols_offd[row].size() );
        total_nnz_diag += nnz_diag[row];
        total_nnz_offd += nnz_offd[row];
    }

    // Give C the nnz counts so that it can allocate space internally
    C->setNNZ( nnz_diag, nnz_offd );

    // Finally, populate structure of C and find local column indices
    auto C_diag                                           = C->getDiagMatrix();
    auto [C_rs_d, C_cols_d, C_cols_loc_d, C_coeffs_d]     = C_diag->getDataFields();
    auto C_offd                                           = C->getOffdMatrix();
    auto [C_rs_od, C_cols_od, C_cols_loc_od, C_coeffs_od] = C_offd->getDataFields();

    lidx_t rp_d = 0, rp_od = 0;
    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( auto it = C_cols_diag[row].begin(); it != C_cols_diag[row].end(); ++it ) {
            C_cols_d[rp_d] = *it;
            ++rp_d;
        }

        for ( auto it = C_cols_offd[row].begin(); it != C_cols_offd[row].end(); ++it ) {
            C_cols_od[rp_od] = *it;
            ++rp_od;
        }
    }

    C->globalToLocalColumns();
    C->resetDOFManagers();
}

template<typename Policy, class Allocator, class DiagMatrixData>
template<class AMatrixData, class BMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::symbolicMultiply(
    std::shared_ptr<AMatrixData> A_data,
    std::shared_ptr<BMatrixData> B_data,
    const typename Policy::gidx_t col_diag_start,
    const typename Policy::gidx_t col_diag_end,
    const bool is_diag,
    std::vector<std::set<typename Policy::gidx_t>> &C_cols )
{
    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto idx_test = [col_diag_start, col_diag_end, is_diag]( const gidx_t col ) -> bool {
        return is_diag ? ( col_diag_start <= col && col < col_diag_end ) :
                         ( col < col_diag_start || col_diag_end <= col );
    };

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    auto [A_rs, A_cols, A_cols_loc, A_coeffs] = A_data->getDataFields();

    // can't capture structured bindings so pull out B fields via std::tie
    lidx_t *B_rs = nullptr, *B_cols_loc = nullptr;
    gidx_t *B_cols     = nullptr;
    scalar_t *B_coeffs = nullptr;

    std::tie( B_rs, B_cols, B_cols_loc, B_coeffs ) = B_data->getDataFields();

    // may or may not have access to B global column indices
    // set up conversion function from local indices
    auto B_colmap          = B_data->getColumnMap();
    const auto B_first_col = B_data->beginCol();
    const bool have_B_cols = ( B_cols != nullptr );

    auto B_to_global = [B_cols, B_cols_loc, B_first_col, B_colmap, is_diag, have_B_cols](
                           const lidx_t k ) -> gidx_t {
        return have_B_cols ? B_cols[k] :
                             ( is_diag ? B_first_col + B_cols_loc[k] : B_colmap[B_cols_loc[k]] );
    };

    // for each row in A block
    for ( lidx_t row = 0; row < nRows; ++row ) {
        // get rows in B block from the A column indices
        for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
            auto Acl = A_cols_loc[j];
            // then row of C is union of those B row nz patterns
            for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                const auto bc = B_to_global( k );
                if ( idx_test( bc ) ) {
                    C_cols[row].insert( bc );
                }
            }
        }
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::createBRemoteCommInfo()
{
    /*
     * Setting up the comms is somewhat involved. A high level overview
     * of the steps involved is:
     * 1. Gather final row of B from each rank so that we know who owns what
     * 2. Get column map for A offd, these indices are the rows of B needed
     * 3. Count # rows needed from each rank, >0 implies dependency on that rank
     * 4. Do all-to-all of count from step 3 to # rows to send to each rank
     * 5. Trim down lists from steps 3 and 4 to ranks that are actually needed
     * 6. Record which specific rows are needed from each process
     * 7. Send row ids from 6 to owners of those rows
     * 8. Record total number of non-zeros from the row ids we need to send out
     * 9. Send row lengths from 7 to requestors of those rows

     * NOTES:
     *  Steps 1 and 4 involve global blocking communication and should be
     *  considered for improvement in the future.
     *
     *  Steps 7 and 9 use non-blocking recvs and blocking sends. Locations
     *  for irecvs, sends, waits should be examined more closely
     */

    PROFILE( "CSRMatrixSpGEMMDefault::createBRemoteCommInfo" );

    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto B_diag                                       = B->getDiagMatrix();
    auto [B_rs_d, B_cols_d, B_cols_loc_d, B_coeffs_d] = B_diag->getDataFields();

    lidx_t *B_rs_od = nullptr, *B_cols_loc_od = nullptr;
    gidx_t *B_cols_od     = nullptr;
    scalar_t *B_coeffs_od = nullptr;
    if ( B->hasOffDiag() ) {
        auto B_offd                                                = B->getOffdMatrix();
        std::tie( B_rs_od, B_cols_od, B_cols_loc_od, B_coeffs_od ) = B_offd->getDataFields();
    }

    auto comm_size = comm.getSize();

    // 1. gather last row of B from each process
    auto B_last_rows = comm.allGather( B->endRow() );

    // 2. get the column map of A_offd
    const auto A_col_map_size = A->getOffdMatrix()->numUniqueColumns();
    const auto A_col_map      = A->getOffdMatrix()->getColumnMap();

    // 3. record how many rows needed from each rank
    std::vector<int> rows_per_rank_recv( comm_size, 0 );
    for ( lidx_t n = 0; n < A_col_map_size; ++n ) {
        const auto col = static_cast<std::size_t>( A_col_map[n] );
        if ( col < B_last_rows[0] ) {
            rows_per_rank_recv[0]++;
            continue;
        }
        for ( int r = 1; r < comm_size; ++r ) {
            auto rs = B_last_rows[r - 1], re = B_last_rows[r];
            if ( rs <= col && col < re ) {
                rows_per_rank_recv[r]++;
                break;
            }
        }
    }

    // 4. find how many rows need to get sent to each rank
    std::vector<int> rows_per_rank_send( comm_size, 0 );
    comm.allToAll( 1, rows_per_rank_recv.data(), rows_per_rank_send.data() );

    // 5. the above rows per rank lists (should) include lots of zeros
    // trim down to the ranks that actually need to communicate
    for ( int r = 0; r < comm_size; ++r ) {
        const auto nsend = rows_per_rank_send[r];
        if ( nsend > 0 ) {
            // ensure that rowids vector has space to recieve that list
            // from requesters
            d_dest_info.insert( std::make_pair( r, SpGEMMCommInfo( nsend ) ) );
        }
        const auto nrecv = rows_per_rank_recv[r];
        if ( nrecv > 0 ) {
            // do not pre-set the size of the rowids vector since it is easier
            // to fill it via push_back
            d_src_info.insert( std::make_pair( r, SpGEMMCommInfo( nrecv ) ) );
        }
    }

    // 6. repeat scan over column map now writing into the trimmed down src list
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
        d_src_info[owner].brow.push_back( n );
    }

    // 7. send rowids to their owners
    // start by posting the irecvs
    {
        const int TAG = 0;
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

    // 8. We now have all global rowids this rank owns and needs to send out
    //    Convert them to local rowids and record the nnz for all rows
    const auto B_first_row = B->beginRow();
    for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
        for ( auto &row : it->second.rowids ) {
            row -= B_first_row;
            const auto nnzd = B_rs_d[row + 1] - B_rs_d[row];
            lidx_t rnnz     = B_rs_od != nullptr ? nnzd + B_rs_od[row + 1] - B_rs_od[row] : nnzd;
            it->second.rownnz.push_back( rnnz );
        }
    }

    // 9. Reverse the pattern of step 7 to send row sizes back to
    //    the reqestors
    {
        const int TAG = 0;
        std::vector<AMP_MPI::Request> irecvs;
        for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
            it->second.rownnz.resize( it->second.numrow );
            irecvs.push_back(
                comm.Irecv( it->second.rownnz.data(), it->second.numrow, it->first, TAG ) );
        }
        // now send all the rows we want from other ranks
        for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
            comm.send( it->second.rownnz.data(), it->second.numrow, it->first, TAG );
        }
        // wait for receives to finish
        comm.waitAll( static_cast<int>( irecvs.size() ), irecvs.data() );
    }
}


template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::createBRemoteSymbolic()
{
    PROFILE( "CSRMatrixSpGEMMDefault::createBRemoteSymbolic" );

    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto B_diag                                       = B->getDiagMatrix();
    auto [B_rs_d, B_cols_d, B_cols_loc_d, B_coeffs_d] = B_diag->getDataFields();
    const auto B_first_col_d                          = B_diag->beginCol();

    lidx_t *B_rs_od = nullptr, *B_cols_loc_od = nullptr;
    gidx_t *B_cols_od     = nullptr;
    gidx_t *B_colmap_od   = nullptr;
    scalar_t *B_coeffs_od = nullptr;
    if ( B->hasOffDiag() ) {
        auto B_offd                                                = B->getOffdMatrix();
        std::tie( B_rs_od, B_cols_od, B_cols_loc_od, B_coeffs_od ) = B_offd->getDataFields();
        B_colmap_od                                                = B_offd->getColumnMap();
    }

    // First set up all of the comm information
    createBRemoteCommInfo();

    // Accumulate num rows from all source ranks to find size of BRemote blocks
    lidx_t BR_nrows = 0;
    {
        // count up number of rows to expect from all other ranks
        for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
            BR_nrows += it->second.numrow;
        }

        // count total non-zeros and record nnz per row
        std::vector<lidx_t> BR_nnz( BR_nrows, 0 );
        for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
            for ( lidx_t r = 0; r < it->second.numrow; ++r ) {
                BR_nnz[it->second.brow[r]] = it->second.rownnz[r];
            }
        }

        BRemote = std::make_shared<DiagMatrixData>(
            nullptr, B->d_memory_location, 0, BR_nrows, 0, B->numGlobalColumns(), true );
        BRemote->setNNZ( BR_nnz );
    }

    auto [BR_rs, BR_cols, BR_cols_loc, BR_coeffs] = BRemote->getDataFields();

    // setup non-zero structure
    {
        const int TAG = 0;
        std::vector<AMP_MPI::Request> irecvs;
        for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
            for ( lidx_t r = 0; r < it->second.numrow; ++r ) {
                const auto in_idx = BR_rs[it->second.brow[r]];
                irecvs.push_back(
                    comm.Irecv( &BR_cols[in_idx], it->second.rownnz[r], it->first, TAG ) );
            }
        }
        // then send all NZ structures to ranks that need them
        for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
            for ( lidx_t r = 0; r < it->second.numrow; ++r ) {
                const auto rid = it->second.rowids[r];
                std::vector<gidx_t> send_cols;
                for ( lidx_t n = B_rs_d[rid]; n < B_rs_d[rid + 1]; ++n ) {
                    send_cols.push_back( B_first_col_d + B_cols_loc_d[n] );
                }
                if ( B_rs_od != nullptr ) {
                    for ( lidx_t n = B_rs_od[rid]; n < B_rs_od[rid + 1]; ++n ) {
                        send_cols.push_back( B_colmap_od[B_cols_loc_od[n]] );
                    }
                }
                comm.send( send_cols.data(), it->second.rownnz[r], it->first, TAG );
            }
        }
        // wait for receives to finish
        comm.waitAll( static_cast<int>( irecvs.size() ), irecvs.data() );
    }
}

// TODO: this is redundant with the last step of the symbolic phase
//       abstract the comms pattern to work for cols or coeffs
template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::fillBRemoteNumeric()
{
    PROFILE( "CSRMatrixSpGEMMDefault::fillBRemoteNumeric" );

    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto B_diag                                       = B->getDiagMatrix();
    auto [B_rs_d, B_cols_d, B_cols_loc_d, B_coeffs_d] = B_diag->getDataFields();

    lidx_t *B_rs_od = nullptr, *B_cols_loc_od = nullptr;
    gidx_t *B_cols_od     = nullptr;
    scalar_t *B_coeffs_od = nullptr;
    if ( B->hasOffDiag() ) {
        auto B_offd                                                = B->getOffdMatrix();
        std::tie( B_rs_od, B_cols_od, B_cols_loc_od, B_coeffs_od ) = B_offd->getDataFields();
    }

    auto [BR_rs, BR_cols, BR_cols_loc, BR_coeffs] = BRemote->getDataFields();

    // setup non-zero structure
    {
        const int TAG = 0;
        std::vector<AMP_MPI::Request> irecvs;
        for ( auto it = d_src_info.begin(); it != d_src_info.end(); ++it ) {
            for ( lidx_t r = 0; r < it->second.numrow; ++r ) {
                const auto in_idx = BR_rs[it->second.brow[r]];
                irecvs.push_back(
                    comm.Irecv( &BR_coeffs[in_idx], it->second.rownnz[r], it->first, TAG ) );
            }
        }
        // then send all NZ values to ranks that need them
        for ( auto it = d_dest_info.begin(); it != d_dest_info.end(); ++it ) {
            for ( lidx_t r = 0; r < it->second.numrow; ++r ) {
                const auto rid = it->second.rowids[r];
                std::vector<scalar_t> send_coeffs;
                for ( lidx_t n = B_rs_d[rid]; n < B_rs_d[rid + 1]; ++n ) {
                    send_coeffs.push_back( B_coeffs_d[n] );
                }
                if ( B_rs_od != nullptr ) {
                    for ( lidx_t n = B_rs_od[rid]; n < B_rs_od[rid + 1]; ++n ) {
                        send_coeffs.push_back( B_coeffs_od[n] );
                    }
                }
                comm.send( send_coeffs.data(), it->second.rownnz[r], it->first, TAG );
            }
        }
        // wait for receives to finish
        comm.waitAll( static_cast<int>( irecvs.size() ), irecvs.data() );
    }
}


template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::numericMultiply()
{
    PROFILE( "CSRMatrixSpGEMMDefault::numericMultiply" );

    auto A_diag = A->getDiagMatrix();
    auto A_offd = A->getOffdMatrix();
    auto B_diag = B->getDiagMatrix();
    auto B_offd = B->getOffdMatrix();
    auto C_diag = C->getDiagMatrix();
    auto C_offd = C->getOffdMatrix();

    // Process diagonal block of A acting on whole local part of B
    {
        PROFILE( "CSRMatrixSpGEMMDefault::numericMultiply (local)" );
        numericMultiply( A_diag, B_diag, C_diag );
        if ( B->hasOffDiag() ) {
            numericMultiply( A_diag, B_offd, C_offd );
        }
    }

    // off-diagonal block requires fetching non-local rows of B
    if ( A->hasOffDiag() ) {
        PROFILE( "CSRMatrixSpGEMMDefault::numericMultiply (remote)" );
        fillBRemoteNumeric();
        numericMultiply( A_offd, BRemote, C_diag );
        numericMultiply( A_offd, BRemote, C_offd );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
template<class AMatrixData, class BMatrixData, class CMatrixData>
void CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>::numericMultiply(
    std::shared_ptr<AMatrixData> A_data,
    std::shared_ptr<BMatrixData> B_data,
    std::shared_ptr<CMatrixData> C_data )
{
    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    const bool is_diag          = C_data->isDiag();
    const gidx_t col_diag_start = C_data->beginCol();
    const gidx_t col_diag_end   = C_data->endCol();

    auto idx_test = [col_diag_start, col_diag_end, is_diag]( const gidx_t col ) -> bool {
        return is_diag ? ( col_diag_start <= col && col < col_diag_end ) :
                         ( col < col_diag_start || col_diag_end <= col );
    };

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    auto [A_rs, A_cols, A_cols_loc, A_coeffs] = A_data->getDataFields();

    // can't capture structured bindings so pull out B fields via std::tie
    lidx_t *B_rs = nullptr, *B_cols_loc = nullptr;
    gidx_t *B_cols     = nullptr;
    scalar_t *B_coeffs = nullptr;

    std::tie( B_rs, B_cols, B_cols_loc, B_coeffs ) = B_data->getDataFields();

    // same for C fields
    lidx_t *C_rs = nullptr, *C_cols_loc = nullptr;
    gidx_t *C_cols     = nullptr;
    scalar_t *C_coeffs = nullptr;

    std::tie( C_rs, C_cols, C_cols_loc, C_coeffs ) = C_data->getDataFields();

    // may or may not have access to B global column indices
    // set up conversion function from local indices
    auto B_colmap          = B_data->getColumnMap();
    const auto B_first_col = B_data->beginCol();
    const bool have_B_cols = ( B_cols != nullptr );

    auto B_to_global = [B_cols, B_cols_loc, B_first_col, B_colmap, is_diag, have_B_cols](
                           const lidx_t k ) -> gidx_t {
        return have_B_cols ? B_cols[k] :
                             ( is_diag ? B_first_col + B_cols_loc[k] : B_colmap[B_cols_loc[k]] );
    };

    // and similar for C, except never have access to global cols
    auto C_colmap          = C_data->getColumnMap();
    const auto C_first_col = C_data->beginCol();

    auto C_to_global = [C_cols_loc, C_first_col, C_colmap, is_diag]( const lidx_t k ) -> gidx_t {
        return is_diag ? C_first_col + C_cols_loc[k] : C_colmap[C_cols_loc[k]];
    };

    // for each row in A block
    std::map<gidx_t, scalar_t> C_colval;
    for ( lidx_t row = 0; row < nRows; ++row ) {
        C_colval.clear();
        for ( lidx_t j = A_rs[row]; j < A_rs[row + 1]; ++j ) {
            const auto Acl = A_cols_loc[j];
            const auto Av  = A_coeffs[j];
            for ( lidx_t k = B_rs[Acl]; k < B_rs[Acl + 1]; ++k ) {
                const auto Bc = B_to_global( k );
                if ( idx_test( Bc ) ) {
                    const auto val = Av * B_coeffs[k];
                    auto in        = C_colval.insert( { Bc, val } );
                    if ( !in.second ) {
                        C_colval[Bc] += val;
                    }
                }
            }
        }
        // Unpack col<->val maps into coeffs of C
        for ( lidx_t c = C_rs[row]; c < C_rs[row + 1]; ++c ) {
            C_coeffs[c] += C_colval[C_to_global( c )];
        }
    }
}

} // namespace AMP::LinearAlgebra
