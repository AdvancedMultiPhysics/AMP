#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/MIS2Aggregator.h"
#include "AMP/solvers/amg/Strength.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/CommunicationList.h"

#ifdef AMP_USE_DEVICE
    #include "AMP/utils/device/Device.h"
    #include <thrust/binary_search.h>
    #include <thrust/copy.h>
    #include <thrust/count.h>
    #include <thrust/remove.h>
    #include <thrust/transform.h>
#endif

#include "ProfilerApp.h"

#include <algorithm>
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
        return assignLocalAggregatesDevice( A, agg_ids );
#else
        AMP_ERROR( "MIS2Aggregator::assignLocalAggregates Undefined memory location" );
#endif
    }
}

template<typename Config>
int MIS2Aggregator::assignLocalAggregatesHost( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                               int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregatesHost" );

    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    // Get diag block from A and mask it using SoC
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();

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
    lidx_t *Am_rs = nullptr, *Am_cols_loc = nullptr;
    gidx_t *Am_cols                                    = nullptr;
    scalar_t *Am_coeffs                                = nullptr;
    std::tie( Am_rs, Am_cols, Am_cols_loc, Am_coeffs ) = A_masked->getDataFields();

    // Create temporary storage for aggregate sizes
    std::vector<lidx_t> agg_size;

    // label each vertex as in or out of MIS-2
    std::vector<uint64_t> labels( A_nrows, OUT );

    // Initialize ids to either unassigned (default) or invalid (isolated)
    lidx_t num_isolated = 0;
    std::vector<lidx_t> worklist;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        if ( re - rs > 1 ) {
            worklist.push_back( row );
            agg_ids[row] = UNASSIGNED;
        } else {
            num_isolated++;
            agg_ids[row] = INVALID;
        }
    }
    classifyVerticesHost<Config>( A_masked, A->numGlobalRows(), worklist, labels, agg_ids );

    // initialize aggregates from nodes flagged as IN and all of their neighbors
    lidx_t num_agg = 0, num_unagg = A_nrows;
    double total_agg = 0.0;

    auto agg_from_row = [&]( lidx_t row, bool test_nbrs ) -> void {
        if ( labels[row] != IN || agg_ids[row] != UNASSIGNED ) {
            // not root node, nothing to do
            return;
        }
        if ( test_nbrs ) {
            int n_nbrs = 0;
            for ( lidx_t c = Am_rs[row] + 1; c < Am_rs[row + 1]; ++c ) {
                if ( agg_ids[Am_cols_loc[c]] == UNASSIGNED ) {
                    ++n_nbrs;
                }
            }
            if ( n_nbrs < 2 ) {
                // too small, skip
                return;
            }
        }
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
    };

    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        agg_from_row( row, false );
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
    classifyVerticesHost<Config>( A_masked, A->numGlobalRows(), worklist, labels, agg_ids );

    // on second pass only allow IN vertex to be root of aggregate if it has
    // at least 2 un-aggregated nbrs
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        agg_from_row( row, true );
    }
    AMP::pout << "Formed " << num_agg << " aggregates with average size " << total_agg / num_agg
              << ", leaving " << num_unagg << " unaggregated points" << std::endl;

    // Add unmarked entries to the smallest aggregate they are nbrs with
    bool grew_agg;
    do {
        grew_agg = false;
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            if ( agg_ids[row] != UNASSIGNED ) {
                // already aggregated, nothing to do
                continue;
            }

            // find smallest neighboring aggregate
            const auto rs = Am_rs[row], re = Am_rs[row + 1];
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

    return num_agg;
}

template<typename Config>
int MIS2Aggregator::classifyVerticesHost(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    const uint64_t num_gbl,
    std::vector<typename Config::lidx_t> &worklist,
    std::vector<uint64_t> &Tv,
    int *agg_ids )
{
    PROFILE( "MIS2Aggregator::classifyVerticesHost" );

    using lidx_t = typename Config::lidx_t;

    // unpack diag block
    const auto A_nrows                            = static_cast<lidx_t>( A_diag->numLocalRows() );
    const auto begin_row                          = A_diag->beginRow();
    auto [Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // hash is xorshift* as given on wikipedia
    auto hash = [=]( uint64_t x ) -> uint64_t {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return x * 0x2545F4914F6CDD1D;
    };

    // get masks for fields in packed tuples
    const auto id_mask   = getIdMask( num_gbl );
    const auto hash_mask = getHashMask( id_mask );

    // pack tuple of number of connections, hash value, and global id
    // give top bits to connection count to bias choices towards more
    // connected nodes. The connections take the highest 5 bits, the
    // id a number of bits determined above, and the hash the remaining middle bits.
    // create a mask for that region as
    auto pack_tuple = [=]( lidx_t idx, uint8_t nconn, uint64_t ihash ) -> uint64_t {
        uint64_t tpl = nconn < 31 ? nconn : 31;
        tpl <<= 59;
        const auto v      = static_cast<uint64_t>( begin_row + idx );
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

    const bool dbg = false;

    // now loop until worklist is empty
    const lidx_t max_iters = 20;
    int num_iters          = 0;
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

        if ( dbg ) {
            AMP::pout << "  dbg-cv[0]";
            const auto num_in  = std::count( Tv.cbegin(), Tv.cend(), IN );
            const auto num_out = std::count( Tv.cbegin(), Tv.cend(), OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
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

        if ( dbg ) {
            AMP::pout << "  dbg-cv[1]";
            const auto num_in  = std::count( Tv.cbegin(), Tv.cend(), IN );
            const auto num_out = std::count( Tv.cbegin(), Tv.cend(), OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
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

        if ( dbg ) {
            AMP::pout << "  dbg-cv[2]";
            const auto num_in  = std::count( Tv.cbegin(), Tv.cend(), IN );
            const auto num_out = std::count( Tv.cbegin(), Tv.cend(), OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
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

        if ( dbg ) {
            AMP::pout << "  dbg-cv[3]";
            const auto num_in  = std::count( Tv.cbegin(), Tv.cend(), IN );
            const auto num_out = std::count( Tv.cbegin(), Tv.cend(), OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
        }

        // update worklist as rows that remain undecided
        std::vector<lidx_t> worklist_new;
        for ( const auto n : worklist ) {
            if ( Tv[n] != IN && Tv[n] != OUT ) {
                worklist_new.push_back( n );
            }
        }

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

template<typename lidx_t>
__global__ void mark_undec_inv( const lidx_t num_rows, const lidx_t *row_start, lidx_t *agg_ids )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        const auto rs = row_start[row], re = row_start[row + 1];
        if ( re - rs > 1 ) {
            agg_ids[row] = MIS2Aggregator::UNASSIGNED;
        } else {
            agg_ids[row] = MIS2Aggregator::INVALID;
        }
    }
}

template<typename lidx_t>
__device__ void agg_from_row( const lidx_t row,
                              const bool test_nbrs,
                              const lidx_t *cols_loc,
                              const lidx_t *row_start,
                              uint64_t *labels,
                              lidx_t *agg_size,
                              lidx_t *agg_ids )
{
    if ( labels[row] != MIS2Aggregator::IN || agg_ids[row] != MIS2Aggregator::UNASSIGNED ) {
        // not root node, nothing to do
        return;
    }
    if ( test_nbrs ) {
        int n_nbrs = 0;
        for ( lidx_t c = row_start[row] + 1; c < row_start[row + 1]; ++c ) {
            if ( agg_ids[cols_loc[c]] == MIS2Aggregator::UNASSIGNED ) {
                ++n_nbrs;
            }
        }
        if ( n_nbrs < 2 ) {
            // too small, skip
            return;
        }
    }
    // have root node, push new aggregate and set ids
    agg_size[row] = 0;
    for ( lidx_t c = row_start[row]; c < row_start[row + 1]; ++c ) {
        if ( agg_ids[cols_loc[c]] != MIS2Aggregator::UNASSIGNED ) {
            continue;
        }
        if ( c > row_start[row] ) {
            labels[cols_loc[c]] = MIS2Aggregator::OUT;
        }
        agg_ids[cols_loc[c]] = row;
        agg_size[row]++;
    }
}

template<typename lidx_t>
__global__ void build_aggs( const lidx_t num_rows,
                            const bool test_nbrs,
                            const lidx_t *row_start,
                            const lidx_t *cols_loc,
                            uint64_t *labels,
                            lidx_t *agg_size,
                            lidx_t *agg_ids )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        agg_from_row( row, test_nbrs, cols_loc, row_start, labels, agg_size, agg_ids );
    }
}

template<typename lidx_t>
__global__ void grow_aggs( const lidx_t num_rows,
                           const lidx_t *row_start,
                           const lidx_t *cols_loc,
                           lidx_t *agg_size,
                           lidx_t *agg_ids )
{
    for ( int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows;
          row += blockDim.x * gridDim.x ) {
        if ( agg_ids[row] != MIS2Aggregator::UNASSIGNED ) {
            // already aggregated, nothing to do
            continue;
        }
        const auto rs = row_start[row], re = row_start[row + 1];

        // find smallest neighboring aggregate
        lidx_t small_agg_id = -1, small_agg_size = num_rows + 1;
        for ( lidx_t c = rs; c < re; ++c ) {
            const auto agg = agg_ids[cols_loc[c]];
            // only consider nbrs that are aggregated
            if ( agg == MIS2Aggregator::UNASSIGNED || agg == MIS2Aggregator::INVALID ) {
                continue;
            }
            if ( agg_size[agg] < small_agg_size ) {
                small_agg_size = agg_size[agg];
                small_agg_id   = agg;
            }
        }

        // add to aggregate
        if ( small_agg_id >= 0 ) {
            agg_ids[row] = small_agg_id;
        }
    }
}

template<typename Config>
int MIS2Aggregator::assignLocalAggregatesDevice(
    std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregatesDevice" );

    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;
    using allocator_type    = typename Config::allocator_type;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    using lidxAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<lidx_t>;
    using u64Allocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<uint64_t>;

    // Get diag block from A and mask it using SoC
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();

    AMP::pout << "    MIS2 got diag block of size: " << A_diag->numLocalRows() << " x "
              << A_diag->numLocalColumns() << ", nnz = " << A_diag->numberOfNonZeros() << std::endl;

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

    AMP::pout << "    SoC masked to block of size: " << A_masked->numLocalRows() << " x "
              << A_masked->numLocalColumns() << ", nnz = " << A_masked->numberOfNonZeros()
              << std::endl;

    // pull out data fields from A_masked
    // only care about row starts and local cols
    lidx_t *Am_rs = nullptr, *Am_cols_loc = nullptr;
    gidx_t *Am_cols                                    = nullptr;
    scalar_t *Am_coeffs                                = nullptr;
    std::tie( Am_rs, Am_cols, Am_cols_loc, Am_coeffs ) = A_masked->getDataFields();

    // get temporary storage for aggregate sizes and MIS2 labels
    u64Allocator_t u64_alloc;
    lidxAllocator_t lidx_alloc;
    auto Tv     = u64_alloc.allocate( A_nrows );
    auto Tv_hat = u64_alloc.allocate( A_nrows );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv, A_nrows, OUT );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv_hat, A_nrows, OUT );
    auto agg_size       = lidx_alloc.allocate( A_nrows );
    auto agg_root_ids   = lidx_alloc.allocate( A_nrows );
    auto worklist       = lidx_alloc.allocate( A_nrows );
    lidx_t worklist_len = A_nrows;
    AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_root_ids, A_nrows, UNASSIGNED );

    const bool dbg = false;

    deviceSynchronize();

    // Initialize ids to either unassigned (default) or invalid (isolated)
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( A_nrows, BlockDim, GridDim );
        mark_undec_inv<<<GridDim, BlockDim>>>( A_nrows, Am_rs, agg_root_ids );
        deviceSynchronize();
    }

    // initilalize worklist to all UNASSIGNED rows
    {
        lidx_t *new_end = thrust::copy_if( thrust::device,
                                           thrust::make_counting_iterator( 0 ),
                                           thrust::make_counting_iterator( A_nrows ),
                                           worklist,
                                           [agg_root_ids] __device__( const lidx_t n ) -> bool {
                                               return ( agg_root_ids[n] == UNASSIGNED );
                                           } );
        worklist_len    = static_cast<lidx_t>( new_end - worklist );
    }

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[0]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << ", wl len: " << worklist_len
                  << std::endl;
    }

    // First pass of MIS2 classification, ignores INVALID nodes
    classifyVerticesDevice<Config>(
        A_masked, A->numGlobalRows(), worklist, worklist_len, Tv, Tv_hat, agg_root_ids );
    deviceSynchronize();

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[1]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        const auto num_in  = thrust::count( thrust::device, Tv, Tv + A_nrows, IN );
        const auto num_out = thrust::count( thrust::device, Tv, Tv + A_nrows, OUT );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << std::endl;
        AMP::pout << "  num_in: " << num_in << ", num_out: " << num_out
                  << ", sum: " << num_in + num_out << std::endl;
    }

    // initialize aggregates from nodes flagged as IN and all of their neighbors
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( A_nrows, BlockDim, GridDim );
        build_aggs<<<GridDim, BlockDim>>>(
            A_nrows, false, Am_rs, Am_cols_loc, Tv, agg_size, agg_root_ids );
        deviceSynchronize();
    }

    // re-initilalize worklist to all UNASSIGNED rows
    {
        lidx_t *new_end = thrust::copy_if( thrust::device,
                                           thrust::make_counting_iterator( 0 ),
                                           thrust::make_counting_iterator( A_nrows ),
                                           worklist,
                                           [agg_root_ids] __device__( const lidx_t n ) -> bool {
                                               return ( agg_root_ids[n] == UNASSIGNED );
                                           } );
        worklist_len    = static_cast<lidx_t>( new_end - worklist );
    }

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[2]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        const auto num_in  = thrust::count( thrust::device, Tv, Tv + A_nrows, IN );
        const auto num_out = thrust::count( thrust::device, Tv, Tv + A_nrows, OUT );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << ", wl len: " << worklist_len
                  << std::endl;
        AMP::pout << "  num_in: " << num_in << ", num_out: " << num_out
                  << ", sum: " << num_in + num_out << std::endl;
    }

    // do a second pass of classification and aggregation
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv, A_nrows, OUT );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv_hat, A_nrows, OUT );
    classifyVerticesDevice<Config>(
        A_masked, A->numGlobalRows(), worklist, worklist_len, Tv, Tv_hat, agg_root_ids );
    deviceSynchronize();

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[3]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        const auto num_in  = thrust::count( thrust::device, Tv, Tv + A_nrows, IN );
        const auto num_out = thrust::count( thrust::device, Tv, Tv + A_nrows, OUT );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << std::endl;
        AMP::pout << "  num_in: " << num_in << ", num_out: " << num_out
                  << ", sum: " << num_in + num_out << std::endl;
    }

    // on second pass only allow IN vertex to be root of aggregate if it has
    // at least 2 un-aggregated nbrs
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( A_nrows, BlockDim, GridDim );
        build_aggs<<<GridDim, BlockDim>>>(
            A_nrows, false, Am_rs, Am_cols_loc, Tv, agg_size, agg_root_ids );
        deviceSynchronize();
    }

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[4]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        const auto num_in  = thrust::count( thrust::device, Tv, Tv + A_nrows, IN );
        const auto num_out = thrust::count( thrust::device, Tv, Tv + A_nrows, OUT );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << std::endl;
        AMP::pout << "  num_in: " << num_in << ", num_out: " << num_out
                  << ", sum: " << num_in + num_out << std::endl;
    }

    // deallocate Tv and Tv_hat, they are no longer needed
    u64_alloc.deallocate( Tv, A_nrows );
    u64_alloc.deallocate( Tv_hat, A_nrows );

    // Add unmarked entries to the smallest aggregate they are nbrs with
    // this differs from host version, here sizes are not updated during
    // growth to avoid race conditions. "smallest aggregate" now means
    // smallest from above steps only. Also, for simplicity it just gets
    // called twice
    {
        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( A_nrows, BlockDim, GridDim );
        grow_aggs<<<GridDim, BlockDim>>>( A_nrows, Am_rs, Am_cols_loc, agg_size, agg_root_ids );
        deviceSynchronize();
        grow_aggs<<<GridDim, BlockDim>>>( A_nrows, Am_rs, Am_cols_loc, agg_size, agg_root_ids );
        deviceSynchronize();
        grow_aggs<<<GridDim, BlockDim>>>( A_nrows, Am_rs, Am_cols_loc, agg_size, agg_root_ids );
        deviceSynchronize();
    }

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[5]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << std::endl;
    }

    // agg_root_ids currently uses root node of each aggregate as the ID
    // need to count number of unique ones to get total number of aggregates
    // agg_size is no longer needed, so copy ids to there, sort, call unique
    int num_agg = 0, num_dec = 0;
    lidx_t *unq_root_ids = agg_size; // rename for clarity
    {
        AMP::Utilities::Algorithms<lidx_t>::copy_n( agg_root_ids, A_nrows, unq_root_ids );
        deviceSynchronize();
        AMP::Utilities::Algorithms<lidx_t>::sort( unq_root_ids, A_nrows );
        deviceSynchronize();
        const auto nunq = AMP::Utilities::Algorithms<lidx_t>::unique( unq_root_ids, A_nrows );
        deviceSynchronize();
        // need to check first two entries of unique'd array
        // if we have UNDECIDED or INVALID need to decrement agg count
        lidx_t first_entries[2];
        AMP::Utilities::Algorithms<lidx_t>::copy_n( unq_root_ids, 2, first_entries );
        deviceSynchronize();
        const int dec_inv = ( first_entries[0] == INVALID || first_entries[1] == INVALID ) ? 1 : 0;
        const int dec_und =
            ( first_entries[0] == UNASSIGNED || first_entries[1] == UNASSIGNED ) ? 1 : 0;
        num_dec = dec_inv + dec_und;
        num_agg = static_cast<int>( nunq ) - num_dec;
    }

    if ( dbg ) {
        AMP::pout << "MIS3 dbg[6]:\n";
        const auto num_undec =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, UNASSIGNED );
        const auto num_invalid =
            thrust::count( thrust::device, agg_root_ids, agg_root_ids + A_nrows, INVALID );
        AMP::pout << "  num_undec: " << num_undec << ", num_invalid: " << num_invalid
                  << ", sum: " << num_undec + num_invalid << std::endl;
    }

    // finally, use compacted list of uniques to convert root node labels
    // to consecutive ascending labels. Do need an additional work vector to
    {
        // search for root node ids in list of uniques to create local ids
        thrust::lower_bound( thrust::device,
                             unq_root_ids,
                             unq_root_ids + num_agg,
                             agg_root_ids,
                             agg_root_ids + A_nrows,
                             agg_ids );
        deviceSynchronize();
        // subtract num_dec from all agg_ids so that INVALID and UNDECIDED
        // entries remain ignored. Can not do the lower_bound on an offset
        // (&unq_root_ids[num_dec]) because INV/UNDEC would get added to
        // the agg with smallest root id, instead of being ignored
        thrust::transform(
            thrust::device,
            agg_ids,
            agg_ids + A_nrows,
            agg_ids,
            [num_dec] __device__( const lidx_t aid ) -> lidx_t { return aid - num_dec; } );
        deviceSynchronize();
    }

    AMP::pout << "    MIS2 found " << num_agg << " aggregates" << std::endl;

    // deallocate sizes and return
    lidx_alloc.deallocate( agg_root_ids, A_nrows );
    lidx_alloc.deallocate( agg_size, A_nrows );

    return num_agg;
}

template<typename Config>
int MIS2Aggregator::classifyVerticesDevice(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    const uint64_t num_gbl,
    typename Config::lidx_t *worklist,
    typename Config::lidx_t worklist_len,
    uint64_t *Tv,
    uint64_t *Tv_hat,
    int *agg_ids )
{
    PROFILE( "MIS2Aggregator::classifyVerticesDevice" );

    using lidx_t   = typename Config::lidx_t;
    using gidx_t   = typename Config::gidx_t;
    using scalar_t = typename Config::scalar_t;

    // unpack diag block
    const auto A_nrows   = static_cast<lidx_t>( A_diag->numLocalRows() );
    const auto begin_row = A_diag->beginRow();
    lidx_t *Ad_rs = nullptr, *Ad_cols_loc = nullptr;
    gidx_t *Ad_cols                                    = nullptr;
    scalar_t *Ad_coeffs                                = nullptr;
    std::tie( Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs ) = A_diag->getDataFields();

    // hash is xorshift* as given on wikipedia
    auto hash = [] __host__ __device__( uint64_t x ) -> uint64_t {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return x * 0x2545F4914F6CDD1D;
    };

    // get masks for fields in packed tuples
    const auto id_mask   = getIdMask( num_gbl );
    const auto hash_mask = getHashMask( id_mask );

    // pack tuple of number of connections, hash value, and global id
    // give top bits to connection count to bias choices towards more
    // connected nodes. The connections take the highest 5 bits, the
    // id a number of bits determined above, and the hash the remaining middle bits.
    // create a mask for that region as
    auto pack_tuple = [begin_row, hash_mask, hash] __device__(
                          lidx_t idx, uint8_t nconn, uint64_t ihash ) -> uint64_t {
        uint64_t tpl = nconn < 31 ? nconn : 31;
        tpl <<= 59;
        const auto v      = static_cast<uint64_t>( begin_row + idx );
        const auto v_hash = hash_mask & hash( ihash ^ hash( v ) );
        tpl |= v_hash;
        tpl |= v;

        return tpl;
    };

    // can recover index from packed tuple by applying mask to
    // keep only low bits and un-offsetting result
    auto idx_from_tuple = [begin_row, id_mask] __device__( uint64_t tpl ) -> lidx_t {
        const auto v = tpl & id_mask;
        return static_cast<lidx_t>( v - begin_row );
    };

    const bool dbg = false;

    const lidx_t max_iters = 20;
    int num_iters          = 0;
    while ( worklist_len > 0 && num_iters < max_iters ) {
        const auto iter_hash = hash( num_iters + 1 );

        // first update Tv entries from items in worklist
        // this is "refresh row" from paper
        {
            thrust::for_each( thrust::device,
                              worklist,
                              worklist + worklist_len,
                              [Ad_rs, iter_hash, pack_tuple, Tv] __device__( lidx_t n ) -> void {
                                  const uint8_t conn =
                                      Ad_rs[n + 1] - Ad_rs[n] - 1; // ignore self connection
                                  Tv[n] = pack_tuple( n, conn, iter_hash );
                              } );
            deviceSynchronize();
        }

        if ( dbg ) {
            AMP::pout << "  dbg-cv[0]:";
            const auto num_in  = thrust::count( Tv, Tv + A_nrows, IN );
            const auto num_out = thrust::count( Tv, Tv + A_nrows, OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
        }

        // Store largest Tv entry from each neighborhood into Tv_hat,
        // then swap Tv and Tv_hat, repeat, leaving max 2-nbr entry
        // in Tv
        for ( int nrep = 0; nrep < 2; ++nrep ) {
            thrust::for_each(
                thrust::device,
                worklist,
                worklist + worklist_len,
                [Ad_rs, Ad_cols_loc, agg_ids, Tv, Tv_hat] __device__( lidx_t n ) -> void {
                    const auto rs = Ad_rs[n];
                    const auto re = Ad_rs[n + 1];
                    auto lmax     = Tv[n];
                    for ( lidx_t k = rs; k < re; ++k ) {
                        const auto c = Ad_cols_loc[k];
                        lmax         = lmax < Tv[c] ? Tv[c] : lmax;
                    }
                    Tv_hat[n] = lmax;
                } );
            deviceSynchronize();
            std::swap( Tv, Tv_hat );
        }

        if ( dbg ) {
            AMP::pout << "  dbg-cv[1]:";
            const auto num_in  = thrust::count( Tv, Tv + A_nrows, IN );
            const auto num_out = thrust::count( Tv, Tv + A_nrows, OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
        }

        // mark undecided as IN or OUT if possible
        {
            thrust::for_each( thrust::device,
                              worklist,
                              worklist + worklist_len,
                              [idx_from_tuple, agg_ids, Tv] __device__( const lidx_t n ) -> void {
                                  const auto tpl = Tv[n];
                                  if ( tpl == IN ) {
                                      Tv[n] = OUT;
                                  }
                                  if ( tpl != IN && tpl != OUT ) {
                                      const auto idx = idx_from_tuple( tpl );
                                      if ( idx == n ) {
                                          Tv[n] = IN;
                                      }
                                  }
                              } );
            deviceSynchronize();
        }

        if ( dbg ) {
            AMP::pout << "  dbg-cv[2]:";
            const auto num_in  = thrust::count( Tv, Tv + A_nrows, IN );
            const auto num_out = thrust::count( Tv, Tv + A_nrows, OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
        }

        // immediately mark nbrs of IN nodes as OUT
        {
            thrust::for_each(
                thrust::device,
                worklist,
                worklist + worklist_len,
                [Ad_rs, Ad_cols_loc, agg_ids, Tv] __device__( const lidx_t n ) -> void {
                    if ( Tv[n] == IN ) {
                        return;
                    }
                    const auto rs = Ad_rs[n];
                    const auto re = Ad_rs[n + 1];
                    bool nbr_in   = false;
                    for ( lidx_t k = rs; k < re; ++k ) {
                        const auto c = Ad_cols_loc[k];
                        nbr_in       = nbr_in || Tv[c] == IN;
                    }
                    Tv[n] = nbr_in ? OUT : Tv[n];
                } );
            deviceSynchronize();
        }

        if ( dbg ) {
            AMP::pout << "  dbg-cv[3]:";
            const auto num_in  = thrust::count( Tv, Tv + A_nrows, IN );
            const auto num_out = thrust::count( Tv, Tv + A_nrows, OUT );
            AMP::pout << "    iter " << num_iters << " marked " << num_in << " labels as IN and "
                      << num_out << " as out" << std::endl;
        }

        // test if we are done
        {
            lidx_t *new_end = thrust::remove_if( thrust::device,
                                                 worklist,
                                                 worklist + worklist_len,
                                                 [Tv] __device__( const lidx_t n ) -> bool {
                                                     return ( Tv[n] == IN || Tv[n] == OUT );
                                                 } );
            worklist_len    = static_cast<lidx_t>( new_end - worklist );
            deviceSynchronize();
        }

        ++num_iters;
    }

    if ( worklist_len > 0 ) {
        AMP::pout << "classifyVertices stopped with " << worklist_len << " worklist items left"
                  << std::endl;
    } else {
        AMP::pout << "classifyVertices finished in " << num_iters << " iterations" << std::endl;
    }

    return num_iters;
}
#endif

} // namespace AMP::Solver::AMG
