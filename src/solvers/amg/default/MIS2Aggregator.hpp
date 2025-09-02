#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/default/MIS2Aggregator.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/vectors/CommunicationList.h"

#include <cstdint>
#include <limits>
#include <numeric>

namespace AMP::Solver::AMG {

int MIS2Aggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids )
{
    AMP_DEBUG_INSIST( A->numLocalRows() == A->numLocalColumns(),
                      "SimpleAggregator::assignLocalAggregates input matrix must be square" );
    AMP_DEBUG_ASSERT( agg_ids != nullptr );

    return LinearAlgebra::csrVisit(
        A, [=]( auto csr_ptr ) { return assignLocalAggregates( csr_ptr, agg_ids ); } );
}

template<typename Config>
int MIS2Aggregator::classifyVertices( std::shared_ptr<LinearAlgebra::CSRMatrixData<Config>> A,
                                      std::vector<typename Config::lidx_t> &wl1,
                                      std::vector<uint64_t> &Tv,
                                      int *agg_ids )
{
    using lidx_t = typename Config::lidx_t;

    // information about A and unpack diag block
    const auto A_nrows                            = static_cast<lidx_t>( A->numLocalRows() );
    const auto begin_row                          = A->beginRow();
    auto A_diag                                   = A->getDiagMatrix();
    auto [Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // the packed representation uses minimal number of bits for ID part
    // of tuple, get log_2 of (num_gbl + 2)
    const auto num_gbl  = static_cast<uint64_t>( A->numGlobalRows() ); // cast to make unsigned
    const auto id_shift = []( uint64_t ng ) -> uint8_t {
        // log2 from stackoverflow. If only bit_width was c++17...
        uint8_t s = 0;
        while ( ng >>= 1 )
            ++s;
        return s;
    }( num_gbl );

    // hash is xorshift* as given on wikipedia
    auto hash = []( uint64_t x ) -> uint64_t {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        return x * 0x2545F4914F6CDD1D;
    };

    std::vector<uint64_t> Mv( A_nrows, OUT );

    // copy input worklist to wl2
    std::vector<lidx_t> wl2( wl1 );

    // now loop until worklists are empty
    const lidx_t max_iters = A_nrows;
    int num_iters          = 0;
    while ( wl1.size() > 0 ) {
        const auto iter_hash = hash( num_iters );

        // first update Tv entries from items in first worklist
        for ( const auto n : wl1 ) {
            const auto n_hash = hash( iter_hash ^ hash( n ) );
            Tv[n]             = ( n_hash << id_shift ) | static_cast<uint64_t>( begin_row + n + 1 );
            AMP_DEBUG_ASSERT( Tv[n] != IN && Tv[n] != OUT );
        }

        // update all Mv entries from items in second worklist
        // this is refresh column from paper
        for ( const auto n : wl2 ) {
            // set to smallest value in neighborhood
            Mv[n] = OUT;
            for ( lidx_t k = Ad_rs[n]; k < Ad_rs[n + 1]; ++k ) {
                const auto c = Ad_cols_loc[k];
                if ( agg_ids[c] < 0 ) {
                    Mv[n] = Tv[c] < Mv[n] ? Tv[c] : Mv[n];
                }
            }
            // if smallest is marked IN mark this as OUT
            if ( Mv[n] == IN ) {
                Mv[n] = OUT;
            }
        }

        // mark undecided as IN or OUT if possible and build new worklists
        std::vector<lidx_t> wl1_new;
        for ( const auto n : wl1 ) {
            bool mark_out = false, mark_in = true;
            for ( lidx_t k = Ad_rs[n]; k < Ad_rs[n + 1]; ++k ) {
                const auto c = Ad_cols_loc[k];
                if ( agg_ids[c] >= 0 ) {
                    continue;
                }
                if ( Mv[c] == OUT ) {
                    mark_out = true;
                    break;
                }
                mark_in = mark_in && ( Tv[n] == Mv[c] );
            }

            if ( mark_out ) {
                Tv[n] = OUT;
            } else if ( mark_in ) {
                Tv[n] = IN;
            }

            // update first worklist
            if ( Tv[n] != IN && Tv[n] != OUT ) {
                wl1_new.push_back( n );
            }
        }

        // update second work list
        std::vector<lidx_t> wl2_new;
        for ( lidx_t n = 0; n < A_nrows; ++n ) {
            if ( Mv[n] != OUT ) {
                wl2_new.push_back( n );
            }
        }

        // swap updated worklists in and loop around
        wl1.swap( wl1_new );
        wl2.swap( wl2_new );

        ++num_iters;

        if ( num_iters == max_iters ) {
            break;
        }
    }

    if ( num_iters == max_iters ) {
        AMP_ERROR( "MIS2Aggregator::classifyVertices failed to terminate" );
    }

    return num_iters;
}

template<typename Config>
int MIS2Aggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                           int *agg_ids )
{
    using lidx_t       = typename Config::lidx_t;
    using matrix_t     = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t = typename matrix_t::matrixdata_t;

    // information about A and unpack diag block
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();
    auto [Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // initially un-aggregated
    AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_ids, A_nrows, -1 );

    // Create temporary storage for aggregate sizes
    std::vector<lidx_t> agg_size;

    // label each vertex as in or out of MIS-2
    std::vector<uint64_t> labels( A_nrows, OUT );

    // Classify vertices, first pass considers all rows
    std::vector<lidx_t> wl1( A_nrows );
    std::iota( wl1.begin(), wl1.end(), 0 );
    auto niter = classifyVertices<Config>( A_data, wl1, labels, agg_ids );

    // initilize aggregates from nodes flagged as in and all of their neighbors
    lidx_t num_agg = 0, num_unagg = A_nrows;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] == OUT ) {
            continue;
        }
        agg_size.push_back( 0 );
        for ( lidx_t c = Ad_rs[row]; c < Ad_rs[row + 1]; ++c ) {
            agg_ids[Ad_cols_loc[c]] = num_agg;
            agg_size[num_agg]++;
            --num_unagg;
        }
        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // do a second pass of classification and aggregation
    // reset worklist to be all vertices that are not part of an aggregate
    wl1.resize( num_unagg );
    lidx_t n = 0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( agg_ids[row] < 0 ) {
            wl1[n] = row;
            ++n;
        }
    }
    AMP_DEBUG_ASSERT( n == num_unagg );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( labels.data(), A_nrows, OUT );
    niter += classifyVertices<Config>( A_data, wl1, labels, agg_ids );

    // on second pass only allow IN vertex to be root of agg if it has
    // at least 2 un-agg nbrs
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] == OUT || agg_ids[row] >= 0 ) {
            continue;
        }
        int n_nbrs = 0;
        for ( lidx_t c = Ad_rs[row] + 1; c < Ad_rs[row + 1]; ++c ) {
            if ( agg_ids[Ad_cols_loc[c]] < 0 ) {
                ++n_nbrs;
            }
        }
        if ( n_nbrs < 2 ) {
            continue;
        }
        agg_size.push_back( 0 );
        for ( lidx_t c = Ad_rs[row]; c < Ad_rs[row + 1]; ++c ) {
            if ( agg_ids[Ad_cols_loc[c]] < 0 ) {
                agg_ids[Ad_cols_loc[c]] = num_agg;
                agg_size[num_agg]++;
            }
        }
        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // final pass adds unmarked entries to the smallest aggregate they are nbrs with
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( agg_ids[row] >= 0 ) {
            // this row already assigned, skip ahead
            continue;
        }
        // find smallest neighboring aggregate
        lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
        for ( lidx_t k = Ad_rs[row]; k < Ad_rs[row + 1]; ++k ) {
            const auto c  = Ad_cols_loc[k];
            const auto id = agg_ids[c];
            if ( id >= 0 && ( agg_size[id] < small_agg_size ) ) {
                small_agg_size = agg_size[id];
                small_agg_id   = id;
            }
        }
        AMP_DEBUG_ASSERT( small_agg_id >= 0 );
        agg_ids[row] = small_agg_id;
        agg_size[small_agg_id]++;
    }

    return num_agg;
}

} // namespace AMP::Solver::AMG
