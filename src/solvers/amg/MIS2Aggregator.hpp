#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/MIS2Aggregator.h"
#include "AMP/solvers/amg/Strength.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/CommunicationList.h"

#include "ProfilerApp.h"

#include <bitset>
#include <cstdint>
#include <limits>
#include <numeric>

namespace AMP::Solver::AMG {

int MIS2Aggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids )
{
    AMP_DEBUG_INSIST( A->numLocalRows() == A->numLocalColumns(),
                      "MIS2Aggregator::assignLocalAggregates input matrix must be square" );
    AMP_DEBUG_ASSERT( agg_ids != nullptr );

    return LinearAlgebra::csrVisit( A, [this, agg_ids]( auto csr_ptr ) {
        return this->assignLocalAggregates( csr_ptr, agg_ids );
    } );
}

template<typename Config>
int MIS2Aggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                           int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregates" );
    if constexpr ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
        return assignLocalAggregatesHost( A, agg_ids );
    } else {
#ifdef AMP_USE_DEVICE
        AMP_ERROR( "Not implemented yet" );
        return assignLocalAggregatesDevice( A, agg_ids );
#else
        AMP_ERROR( "MIS2Aggregator::assignLocalAggregates Undefined memory location" );
#endif
    }
}

template<typename Config>
void testSymmetry( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A )
{
    using lidx_t                              = typename Config::lidx_t;
    const auto A_nrows                        = static_cast<lidx_t>( A->numLocalRows() );
    const auto A_nnz                          = static_cast<lidx_t>( A->numLocalRows() );
    auto [A_rs, A_cols, A_cols_loc, A_coeffs] = A->getDataFields();

    // assemble crude symbolic transpose
    std::vector<std::set<lidx_t>> At_cols_loc( A_nrows );
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = A_rs[row], re = A_rs[row + 1];
        for ( lidx_t k = rs; k < re; ++k ) {
            const auto c = A_cols_loc[k];
            At_cols_loc[c].insert( row );
        }
    }

    // loop again and check that all NZs match up
    int num_fail = 0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = A_rs[row], re = A_rs[row + 1];
        for ( lidx_t k = rs; k < re; ++k ) {
            const auto c    = A_cols_loc[k];
            auto it         = At_cols_loc[row].find( c );
            const auto rlen = re - rs;
            const auto clen = A_rs[c + 1] - A_rs[c];
            if ( it == At_cols_loc[row].end() && rlen > 1 && clen > 1 ) {
                ++num_fail;
                if ( num_fail < 25 ) {
                    AMP::pout << "A vs At mismatch, (i,j) = (" << row << "," << c << ")"
                              << std::endl;
                }
            }
        }
    }
}

template<typename Config>
int MIS2Aggregator::assignLocalAggregatesHost( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                               int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregatesHost" );

    using lidx_t            = typename Config::lidx_t;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    // Get diag block from A and mask it using SoC
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();
    // testSymmetry( A_diag );

    std::shared_ptr<localmatrixdata_t> A_masked;
    if ( d_strength_measure == "classical_abs" ) {
        AMP_WARN_ONCE( "MIS2 aggregation: Use of a symmetric strength measure is advised" );
        auto S = compute_soc<classical_strength<norm::abs>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    } else if ( d_strength_measure == "classical_min" ) {
        AMP_WARN_ONCE( "MIS2 aggregation: Use of a symmetric strength measure is advised" );
        auto S = compute_soc<classical_strength<norm::min>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    } else if ( d_strength_measure == "symagg_abs" ) {
        auto S   = compute_soc<symagg_strength<norm::abs>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    } else {
        if ( d_strength_measure != "symagg_min" ) {
            AMP_WARN_ONCE( "Unrecognized strength measure, reverting to symagg_min" );
        }
        auto S   = compute_soc<symagg_strength<norm::min>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    }

    // pull out data fields from A_masked
    // only care about row starts and local cols
    auto [Am_rs, Am_cols, Am_cols_loc, Am_coeffs] = A_masked->getDataFields();

    // initially un-aggregated
    AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_ids, A_nrows, UNASSIGNED );

    // Create temporary storage for aggregate sizes
    std::vector<lidx_t> agg_size;

    // label each vertex as in or out of MIS-2
    std::vector<uint64_t> labels( A_nrows, OUT );

    // Classify vertices, first pass considers all rows with length >= 2
    lidx_t num_isolated = 0;
    std::vector<lidx_t> worklist;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        if ( re - rs > 1 ) {
            worklist.push_back( row );
        } else {
            num_isolated++;
            agg_ids[row] = INVALID;
        }
    }
    classifyVerticesHost<Config>(
        A_masked, worklist, labels, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids );

    // initialize aggregates from nodes flagged as IN and all of their neighbors
    lidx_t num_agg = 0, num_unagg = A_nrows;
    double total_agg = 0.0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] != IN ) {
            // not root node, continue ahead
            continue;
        }

        // node labelled IN better be valid and unaggregated
        AMP_ASSERT( agg_ids[row] == UNASSIGNED );

        // have root node, push new aggregate and set ids
        agg_size.push_back( 0 );
        for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
            if ( agg_ids[Am_cols_loc[c]] != UNASSIGNED ) {
                continue;
            }
            if ( c > Am_rs[row] ) {
                AMP_ASSERT( labels[Am_cols_loc[c]] != IN );
                labels[Am_cols_loc[c]] = OUT;
            }
            agg_ids[Am_cols_loc[c]] = num_agg;
            agg_size[num_agg]++;
            --num_unagg;
        }
        total_agg += agg_size[num_agg];
        // increment current id to start working on next aggregate
        ++num_agg;
    }
    AMP::pout << "Formed " << num_agg << " aggregates with average size " << total_agg / num_agg
              << ", leaving " << num_unagg << " unaggregated points" << std::endl;

    // do a second pass of classification and aggregation
    // reset worklist to be all vertices that are not part of
    // an aggregate and not isolated
    worklist.clear();
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( agg_ids[row] == UNASSIGNED ) {
            AMP_ASSERT( labels[row] != IN );
            worklist.push_back( row );
        }
    }
    AMP::Utilities::Algorithms<uint64_t>::fill_n( labels.data(), A_nrows, OUT );
    // AMP::pout << "Calling classify second time" << std::endl;
    classifyVerticesHost<Config>(
        A_masked, worklist, labels, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids );

    // on second pass only allow IN vertex to be root of aggregate if it has
    // at least 2 un-aggregated nbrs
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] != IN || agg_ids[row] != UNASSIGNED ) {
            // not a prospective root or already aggregated
            continue;
        }
        int n_nbrs = 0;
        for ( lidx_t c = Am_rs[row] + 1; c < Am_rs[row + 1]; ++c ) {
            if ( agg_ids[Am_cols_loc[c]] == UNASSIGNED ) {
                ++n_nbrs;
            }
        }
        if ( n_nbrs < 2 ) {
            // too small, skip
            continue;
        }
        agg_size.push_back( 0 );
        for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
            if ( agg_ids[Am_cols_loc[c]] == UNASSIGNED ) {
                agg_ids[Am_cols_loc[c]] = num_agg;
                agg_size[num_agg]++;
                --num_unagg;
                if ( c > Am_rs[row] ) {
                    AMP_DEBUG_ASSERT( labels[Am_cols_loc[c]] != IN );
                }
            }
        }
        total_agg += agg_size[num_agg];
        // increment current id to start working on next aggregate
        ++num_agg;
    }
    AMP::pout << "Formed " << num_agg << " aggregates with average size " << total_agg / num_agg
              << ", leaving " << num_unagg << " unaggregated points" << std::endl;

    // Add unmarked entries to the smallest aggregate they are nbrs with
    bool grew_agg;
    do {
        // AMP::pout << "  growing" << std::endl;
        grew_agg = false;
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            const auto rs = Am_rs[row], re = Am_rs[row + 1];
            if ( agg_ids[row] != UNASSIGNED ) {
                // already aggregated, nothing to do
                continue;
            }

            // find smallest neighboring aggregate
            lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
            for ( lidx_t c = rs; c < re; ++c ) {
                const auto agg = agg_ids[Am_cols_loc[c]];
                // only consider nbrs that are aggregated
                if ( agg != UNASSIGNED && ( agg_size[agg] < small_agg_size ) ) {
                    small_agg_size = agg_size[agg];
                    small_agg_id   = agg;
                }
            }

            // add to aggregate
            if ( small_agg_id >= 0 ) {
                agg_ids[row] = small_agg_id;
                agg_size[small_agg_id]++;
                total_agg += 1.0;
                --num_unagg;
                grew_agg = true;
            }
        }
    } while ( grew_agg );

    // DEBUG
    {
        total_agg          = 0.0;
        lidx_t largest_agg = 0, smallest_agg = A_nrows;
        for ( int n = 0; n < num_agg; ++n ) {
            total_agg += agg_size[n];
            largest_agg  = largest_agg < agg_size[n] ? agg_size[n] : largest_agg;
            smallest_agg = smallest_agg > agg_size[n] ? agg_size[n] : smallest_agg;
        }
        AMP::pout << "MIS2Aggregator found " << num_agg << " aggregates over " << A_nrows
                  << " rows, with average size " << total_agg / static_cast<double>( num_agg )
                  << ", and max/min " << largest_agg << "/" << smallest_agg << std::endl
                  << std::endl;
    }

    return num_agg;
}

template<typename Config>
int MIS2Aggregator::classifyVerticesHost(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    std::vector<typename Config::lidx_t> &worklist,
    std::vector<uint64_t> &Tv,
    const uint64_t num_gbl,
    int *agg_ids )
{
    PROFILE( "MIS2Aggregator::classifyVerticesHost" );

    using lidx_t = typename Config::lidx_t;

    // unpack diag block
    const auto A_nrows                            = static_cast<lidx_t>( A_diag->numLocalRows() );
    const auto begin_row                          = A_diag->beginRow();
    auto [Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // the packed representation uses minimal number of bits for ID part
    // of tuple, get log_2 of (num_gbl + 2)
    AMP_ASSERT( num_gbl < ( std::numeric_limits<uint64_t>::max() - 33 ) );
    const auto id_shift = []( uint64_t ng ) -> uint8_t {
        // log2 from stackoverflow. If only bit_width was c++17...
        uint8_t s = 1;
        while ( ng >>= 1 )
            ++s;
        return s;
    }( num_gbl );

    // make mask of just id_shift lowest bits for un-packing tuples
    const auto id_mask = std::numeric_limits<uint64_t>::max() >> ( 64 - id_shift );

    // hash is xorshift* as given on wikipedia
    auto hash = [=]( uint64_t x ) -> uint64_t {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return x * 0x2545F4914F6CDD1D;
    };

    // pack tuple of number of connections, hash value, and global id
    // give top bits to connection count to bias choices towards more
    // connected nodes. The connections take the highest 5 bits, the
    // id a number of bits determined above, and the hash the remaining middle bits.
    // create a mask for that region as
    const uint64_t conn_mask = ( (uint64_t) 31 ) << 59;
    const uint64_t hash_mask = ~( conn_mask | id_mask );
    auto pack_tuple          = [=]( lidx_t idx, uint8_t nconn, uint64_t ihash ) -> uint64_t {
        uint64_t tpl = nconn < 31 ? nconn : 31;
        tpl <<= 59;
        const auto v = static_cast<uint64_t>( begin_row + idx );
        const auto v_hash = hash_mask & hash( ihash ^ hash( v ) );
        tpl |= v_hash;
        tpl |= v;

        return tpl;
    };

    // can recover index from packed tuple by applying mask to
    // keep only low bits and un-offsetting result
    auto idx_from_tuple = [=]( uint64_t tpl ) -> lidx_t {
        const auto v = tpl & id_mask;
        return static_cast<lidx_t>( v - begin_row );
    };

    // temporary tuples for finding neighborhood max in two steps
    std::vector<uint64_t> Tv_hat( A_nrows, OUT );

    // now loop until worklist is empty
    const lidx_t max_iters = 20;
    int num_iters = 0, num_stag = 0;
    while ( worklist.size() > 0 ) {
        const auto iter_hash = hash( num_iters + 1 );

        // first update Tv entries from items in worklist
        // this is "refresh row" from paper
        for ( const auto n : worklist ) {
            AMP_ASSERT( Tv[n] != IN );
            if ( num_iters > 0 ) {
                AMP_ASSERT( Tv[n] != OUT );
            }
            const uint8_t conn = Ad_rs[n + 1] - Ad_rs[n] - 1; // ignore self connection
            Tv[n]              = pack_tuple( n, conn, iter_hash );

            AMP_ASSERT( n == idx_from_tuple( Tv[n] ) );
            AMP_ASSERT( Tv[n] != IN && Tv[n] != OUT );
            AMP_ASSERT( agg_ids[n] == UNASSIGNED );
        }

        // Store largest Tv entry from each neighborhood into Tv_hat,
        // then swap Tv and Tv_hat, repeat, leaving max 2-nbr entry
        // in Tv
        for ( int nrep = 0; nrep < 2; ++nrep ) {
            for ( const auto n : worklist ) {
                const auto rs = Ad_rs[n], re = Ad_rs[n + 1];
                Tv_hat[n] = Tv[n];
                for ( lidx_t k = rs; k < re; ++k ) {
                    const auto c = Ad_cols_loc[k];
                    Tv_hat[n]    = std::max( Tv[c], Tv_hat[n] );
                }
            }
            Tv.swap( Tv_hat );
        }

        // mark undecided as IN or OUT if possible
        for ( const auto n : worklist ) {
            const auto rs = Ad_rs[n], re = Ad_rs[n + 1], row_len = re - rs;
            AMP_ASSERT( Tv[n] != OUT );

            // Tv[n] is maximal tuple value in 2-nbrhood, so if IN
            // then row n is within two hops of some IN node
            // mark as OUT
            if ( Tv[n] == IN ) {
                const auto idx = idx_from_tuple( Tv[n] );
                AMP_ASSERT( n != idx );
                Tv[n] = OUT;
            }

            // If Tv[n] is neither IN nor OUT then check if n is part of the
            // tuple. If so then this n is the IN member of its 2-nbrhood
            if ( Tv[n] != IN && Tv[n] != OUT ) {
                const auto idx = idx_from_tuple( Tv[n] );
                if ( n == idx ) {
                    Tv[n] = IN;
                }
            }

            // only have directed graph, so there are edge cases where
            // neighbors both end up IN, revert them to to undecided
            if ( Tv[n] == IN ) {
                for ( lidx_t k = rs + 1; k < re; ++k ) {
                    const auto c = Ad_cols_loc[k];
                    if ( Tv[c] == IN ) {
                        AMP_WARN_ONCE( "Collision detected" );
                        Tv[n] = 1;
                        Tv[c] = 1;
                    }
                }
            }
        }

        // immediately mark nbrs of IN nodes as OUT, now guaranteed not to collide
        for ( const auto n : worklist ) {
            const auto rs = Ad_rs[n], re = Ad_rs[n + 1];
            if ( Tv[n] == IN ) {
                for ( lidx_t k = rs + 1; k < re; ++k ) {
                    const auto c = Ad_cols_loc[k];
                    Tv[c]        = OUT;
                }
            }
        }

        // update worklist as rows that remain undecided
        std::vector<lidx_t> worklist_new;
        for ( const auto n : worklist ) {
            if ( Tv[n] != IN && Tv[n] != OUT ) {
                worklist_new.push_back( n );
            }
        }

        // if ( worklist.size() == worklist_new.size() ) {
        //     ++num_stag;
        //     if ( num_stag == 3 ) {
        //         // AMP_WARNING( "MIS2Aggregator::classifyVertices worklist stagnated" );
        //         break;
        //     }
        // }

        if ( worklist_new.size() == 0 ) {
            AMP::pout << "classifyVertices finished in " << num_iters << " passes" << std::endl;
        }

        // swap updated worklists in and loop around
        worklist.swap( worklist_new );
        ++num_iters;
        if ( num_iters == max_iters && worklist.size() != 0 ) {
            AMP::pout << "classifyVertices stopped with " << worklist.size()
                      << " worklist items left" << std::endl;
            // AMP_WARNING( "MIS2Aggregator::classifyVertices failed to terminate" );
            break;
        }
    }

    return num_iters;
}

// Device specific implementations for aggregating and classifying
#ifdef AMP_USE_DEVICE
template<typename Config>
int MIS2Aggregator::assignLocalAggregatesDevice(
    std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregatesDevice" );
    return -1;

    // using lidx_t            = typename Config::lidx_t;
    // using allocator_type    = typename Config::allocator_type;
    // using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    // using matrixdata_t      = typename matrix_t::matrixdata_t;
    // using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    // using u64Allocator_t =
    //     typename std::allocator_traits<allocator_type>::template rebind_alloc<uint64_t>;

    // // label each vertex as in or out of MIS-2
    // u64Allocator_t u64_alloc;
    // auto labels = u64_alloc.allocate( A_nrows );
    // auto minima = u64_alloc.allocate( A_nrows );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( labels, A_nrows, OUT );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( minima, A_nrows, OUT );


    // u64_alloc.deallocate( labels, A_nrows );
    // u64_alloc.deallocate( minima, A_nrows );

    // return num_agg;
}

template<typename Config>
int MIS2Aggregator::classifyVerticesDevice(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    typename Config::lidx_t *wl1,
    typename Config::lidx_t *wl2,
    uint64_t *Tv,
    uint64_t *Mv,
    const uint64_t num_gbl,
    int *agg_ids )
{
    PROFILE( "MIS2Aggregator::classifyVerticesDevice" );

    return -1;
}
#endif

} // namespace AMP::Solver::AMG
