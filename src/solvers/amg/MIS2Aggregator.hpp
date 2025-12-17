#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/MIS2Aggregator.h"
#include "AMP/solvers/amg/Strength.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/CommunicationList.h"

#include "ProfilerApp.h"

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

    std::shared_ptr<localmatrixdata_t> A_masked;
    if ( d_strength_measure == "evolution" ) {
        auto S   = compute_soc<evolution_strength>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    } else if ( d_strength_measure == "classical_abs" ) {
        auto S = compute_soc<classical_strength<norm::abs>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    } else {
        if ( d_strength_measure != "classical_min" ) {
            AMP_WARN_ONCE( "Unrecognized strength measure, reverting to classical_min" );
        }
        auto S = compute_soc<classical_strength<norm::min>>( csr_view( *A ), d_strength_threshold );
        A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    }

    // pull out data fields from A_masked
    // only care about row starts and local cols
    auto [Am_rs, Am_cols, Am_cols_loc, Am_coeffs] = A_masked->getDataFields();

    // initially un-aggregated
    AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_ids, A_nrows, -1 );

    // Create temporary storage for aggregate sizes
    std::vector<lidx_t> agg_size;

    // label each vertex as in or out of MIS-2
    std::vector<uint64_t> labels( A_nrows, OUT );

    // Classify vertices, first pass considers all rows with length >= 2
    lidx_t num_isolated = 0;
    std::vector<lidx_t> wl1;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        if ( re - rs > 1 ) {
            wl1.push_back( row );
        } else {
            num_isolated++;
        }
    }
    classifyVerticesHost<Config>(
        A_masked, wl1, labels, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids );

    // initialize aggregates from nodes flagged as in and all of their neighbors
    lidx_t num_agg = 0, num_unagg = A_nrows;
    double total_agg = 0.0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] != IN ) {
            continue;
        }
        agg_size.push_back( 0 );
        for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
            agg_ids[Am_cols_loc[c]] = num_agg;
            agg_size[num_agg]++;
            --num_unagg;
            if ( c > Am_rs[row] ) {
                AMP_DEBUG_ASSERT( labels[Am_cols_loc[c]] != IN );
            }
        }
        total_agg += agg_size[num_agg];
        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // do a second pass of classification and aggregation
    // reset worklist to be all vertices that are not part of an aggregate
    wl1.clear();
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        if ( agg_ids[row] < 0 && re - rs > 1 ) {
            wl1.push_back( row );
        }
    }
    AMP::Utilities::Algorithms<uint64_t>::fill_n( labels.data(), A_nrows, OUT );
    classifyVerticesHost<Config>(
        A_masked, wl1, labels, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids );

    // on second pass only allow IN vertex to be root of agg if it has
    // at least 2 un-agg nbrs
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( labels[row] != IN || agg_ids[row] >= 0 ) {
            // not a prospective root or already aggregated
            continue;
        }
        if ( Am_rs[row + 1] - Am_rs[row] <= 1 ) {
            // row is isolated, ignore it
            continue;
        }
        int n_nbrs = 0;
        for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
            if ( agg_ids[Am_cols_loc[c]] < 0 ) {
                ++n_nbrs;
            }
        }
        if ( n_nbrs < 2 ) {
            // too small, skip
            continue;
        }
        agg_size.push_back( 0 );
        for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
            if ( agg_ids[Am_cols_loc[c]] < 0 ) {
                agg_ids[Am_cols_loc[c]] = num_agg;
                agg_size[num_agg]++;
                if ( c > Am_rs[row] ) {
                    AMP_DEBUG_ASSERT( labels[Am_cols_loc[c]] != IN );
                }
            }
        }
        total_agg += agg_size[num_agg];
        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // Add unmarked entries to the smallest aggregate they are nbrs with
    bool grew_agg;
    do {
        grew_agg = false;
        for ( lidx_t row = 0; row < A_nrows; ++row ) {
            const auto rs = Am_rs[row], re = Am_rs[row + 1];
            if ( agg_ids[row] >= 0 ) {
                continue;
            }

            // find smallest neighboring aggregate
            lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
            for ( lidx_t c = rs; c < re; ++c ) {
                const auto agg = agg_ids[Am_cols_loc[c]];
                // only consider nbrs that are aggregated
                if ( agg >= 0 && ( agg_size[agg] < small_agg_size ) ) {
                    small_agg_size = agg_size[agg];
                    small_agg_id   = agg;
                }
            }

            // add to aggregate
            if ( small_agg_id >= 0 ) {
                agg_ids[row] = small_agg_id;
                agg_size[small_agg_id]++;
                grew_agg = true;
            }
        }
    } while ( grew_agg );

    // check if aggregated points neighbor any isolated points
    // and add them to their aggregate if so. These mostly come from BCs
    // where connections might not be symmetric.
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        const auto curr_agg = agg_ids[row];

        if ( curr_agg < 0 ) {
            continue;
        }

        for ( lidx_t c = rs; c < re; ++c ) {
            const auto nid = Am_cols_loc[c];
            if ( Am_rs[nid + 1] - Am_rs[nid] <= 1 ) {
                agg_ids[nid] = curr_agg;
                agg_size[curr_agg]++;
            }
        }
    }

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
                  << ", and max/min " << largest_agg << "/" << smallest_agg << std::endl;
    }

    return num_agg;
}

template<typename Config>
int MIS2Aggregator::classifyVerticesHost(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    std::vector<typename Config::lidx_t> &wl1,
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
    const lidx_t max_iters = 20;
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
            const auto rs = Ad_rs[n], re = Ad_rs[n + 1], row_len = re - rs;
            if ( row_len <= 1 || agg_ids[n] >= 0 ) {
                AMP_ERROR( "This is impossible right?" );
                // point is isolated or already aggregated, skip
                continue;
            }

            // default to IN and check if conditions hold
            bool mark_out = false, mark_in = true;
            for ( lidx_t k = rs; k < re; ++k ) {
                const auto c = Ad_cols_loc[k];
                if ( agg_ids[c] >= 0 ) {
                    // neighbor is aggregated from previous vertex classification pass
                    // ignore on this pass
                    continue;
                }
                if ( Mv[c] == OUT ) {
                    mark_out = true;
                    break;
                }
                if ( Mv[c] != Tv[n] ) {
                    mark_in = false;
                }
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
        const bool wl1_stag = wl1.size() == wl1_new.size();
        const bool wl2_stag = wl2.size() == wl2_new.size();
        wl1.swap( wl1_new );
        wl2.swap( wl2_new );

        ++num_iters;

        if ( num_iters == max_iters ) {
            AMP_WARNING( "MIS2Aggregator::classifyVertices failed to terminate" );
            break;
        }

        if ( wl1_stag && wl2_stag ) {
            AMP_WARNING( "MIS2Aggregator::classifyVertices worklists stagnated" );
            AMP::pout << "wl1.size() = " << wl1.size() << ", wl2.size() = " << wl2.size()
                      << std::endl;

            for ( const auto n : wl1 ) {
                Tv[n] = IN;
            }

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

    // // Get diag block from A and mask it using SoC
    // const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    // auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    // auto A_diag        = A_data->getDiagMatrix();

    // std::shared_ptr<localmatrixdata_t> A_masked;
    // if ( d_strength_measure == "evolution" ) {
    //     auto S   = compute_soc<evolution_strength>( csr_view( *A ), d_strength_threshold );
    //     A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    // } else if ( d_strength_measure == "classical_abs" ) {
    //     auto S = compute_soc<classical_strength<norm::abs>>( csr_view( *A ), d_strength_threshold
    //     ); A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    // } else {
    //     if ( d_strength_measure != "classical_min" ) {
    //         AMP_WARN_ONCE( "Unrecognized strength measure, reverting to classical_min" );
    //     }
    //     auto S = compute_soc<classical_strength<norm::min>>( csr_view( *A ), d_strength_threshold
    //     ); A_masked = A_diag->maskMatrixData( S.diag_mask_data(), true );
    // }

    // // pull out data fields from A_masked
    // // only care about row starts and local cols
    // auto [Am_rs, Am_cols, Am_cols_loc, Am_coeffs] = A_masked->getDataFields();

    // // initially un-aggregated
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_ids, A_nrows, -1 );

    // // Create temporary storage for aggregate sizes
    // auto agg_size = localmatrixdata_t::makeLidxArray( A_nrows );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( agg_size.get(), A_nrows, 0 );

    // // label each vertex as in or out of MIS-2
    // u64Allocator_t u64_alloc;
    // auto labels = u64_alloc.allocate( A_nrows );
    // auto minima = u64_alloc.allocate( A_nrows );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( labels, A_nrows, OUT );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( minima, A_nrows, OUT );

    // // Classify vertices, first pass considers all rows with length >= 2
    // auto wl1 = localmatrixdata_t::makeLidxArray( A_nrows );
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( wl1, A_nrows, 0 );
    // for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //     const auto rs = Am_rs[row], re = Am_rs[row + 1];
    //     if ( re - rs > 1 ) {
    //         wl1[row] = 1;
    //     }
    // }
    // classifyVerticesDevice<Config>(
    //     A_masked, wl1, wl2, labels, minima, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids
    //     );

    // // initialize aggregates from nodes flagged as in and all of their neighbors
    // lidx_t num_agg   = 0;
    // double total_agg = 0.0;
    // for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //     if ( labels[row] != IN ) {
    //         continue;
    //     }
    //     for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
    //         agg_ids[Am_cols_loc[c]] = num_agg;
    //         agg_size[num_agg]++;
    //     }
    //     total_agg += agg_size[num_agg];
    //     // increment current id to start working on next aggregate
    //     ++num_agg;
    // }

    // // do a second pass of classification and aggregation
    // // reset worklist to be all vertices that are not part of an aggregate
    // AMP::Utilities::Algorithms<lidx_t>::fill_n( wl1, A_nrows, 0 );
    // for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //     const auto rs = Am_rs[row], re = Am_rs[row + 1];
    //     if ( agg_ids[row] < 0 && re - rs > 1 ) {
    //         wl1[row] = 1;
    //     }
    // }
    // AMP::Utilities::Algorithms<uint64_t>::fill_n( labels, A_nrows, OUT );
    // AMP::Utilities::Algorithms<uint64_t>::fill_n( minima, A_nrows, OUT );
    // classifyVerticesDevice<Config>(
    //     A_masked, wl1, wl2, labels, minima, static_cast<uint64_t>( A->numGlobalRows() ), agg_ids
    //     );

    // // on second pass only allow IN vertex to be root of agg if it has
    // // at least 2 un-agg nbrs
    // for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //     if ( labels[row] != IN || agg_ids[row] >= 0 ) {
    //         // not a prospective root or already aggregated
    //         continue;
    //     }
    //     if ( Am_rs[row + 1] - Am_rs[row] <= 1 ) {
    //         // row is isolated, ignore it
    //         continue;
    //     }
    //     int n_nbrs = 0;
    //     for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
    //         if ( agg_ids[Am_cols_loc[c]] < 0 ) {
    //             ++n_nbrs;
    //         }
    //     }
    //     if ( n_nbrs < 2 ) {
    //         // too small, skip
    //         continue;
    //     }
    //     for ( lidx_t c = Am_rs[row]; c < Am_rs[row + 1]; ++c ) {
    //         if ( agg_ids[Am_cols_loc[c]] < 0 ) {
    //             agg_ids[Am_cols_loc[c]] = num_agg;
    //             agg_size[num_agg]++;
    //         }
    //     }
    //     total_agg += agg_size[num_agg];
    //     // increment current id to start working on next aggregate
    //     ++num_agg;
    // }

    // // Add unmarked entries to the smallest aggregate they are nbrs with
    // bool grew_agg;
    // do {
    //     grew_agg = false;
    //     for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //         const auto rs = Am_rs[row], re = Am_rs[row + 1];
    //         if ( agg_ids[row] >= 0 ) {
    //             continue;
    //         }

    //         // find smallest neighboring aggregate
    //         lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
    //         for ( lidx_t c = rs; c < re; ++c ) {
    //             const auto agg = agg_ids[Am_cols_loc[c]];
    //             // only consider nbrs that are aggregated
    //             if ( agg >= 0 && ( agg_size[agg] < small_agg_size ) ) {
    //                 small_agg_size = agg_size[agg];
    //                 small_agg_id   = agg;
    //             }
    //         }

    //         // add to aggregate
    //         if ( small_agg_id >= 0 ) {
    //             agg_ids[row] = small_agg_id;
    //             agg_size[small_agg_id]++;
    //             grew_agg = true;
    //         }
    //     }
    // } while ( grew_agg );

    // // check if aggregated points neighbor any isolated points
    // // and add them to their aggregate if so. These mostly come from BCs
    // // where connections might not be symmetric.
    // for ( lidx_t row = 0; row < A_nrows; ++row ) {
    //     const auto rs = Am_rs[row], re = Am_rs[row + 1];
    //     const auto curr_agg = agg_ids[row];

    //     if ( curr_agg < 0 ) {
    //         continue;
    //     }

    //     for ( lidx_t c = rs; c < re; ++c ) {
    //         const auto nid = Am_cols_loc[c];
    //         if ( Am_rs[nid + 1] - Am_rs[nid] <= 1 ) {
    //             agg_ids[nid] = curr_agg;
    //             agg_size[curr_agg]++;
    //         }
    //     }
    // }

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

    // using lidx_t = typename Config::lidx_t;

    // // unpack diag block
    // const auto A_nrows                            = static_cast<lidx_t>( A_diag->numLocalRows()
    // ); const auto begin_row                          = A_diag->beginRow(); auto [Ad_rs, Ad_cols,
    // Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // // the packed representation uses minimal number of bits for ID part
    // // of tuple, get log_2 of (num_gbl + 2)
    // const auto id_shift = __device__[]( uint64_t ng )->uint8_t
    // {
    //     // log2 from stackoverflow. If only bit_width was c++17...
    //     uint8_t s = 0;
    //     while ( ng >>= 1 )
    //         ++s;
    //     return s;
    // }
    // ( num_gbl );

    // // hash is xorshift* as given on wikipedia
    // auto hash = __device__[]( uint64_t x )->uint64_t
    // {
    //     x ^= x >> 12;
    //     x ^= x << 25;
    //     x ^= x >> 27;
    //     return x * 0x2545F4914F6CDD1D;
    // };

    // // copy input worklist to wl2
    // AMP::Utilities::copy( A_nrows, wl1, wl2 );

    // // now loop until worklists are empty
    // const lidx_t max_iters = 20;
    // int num_iters          = 0;
    // while ( wl1.size() > 0 ) {
    //     const auto iter_hash = hash( num_iters );

    //     // first update Tv entries from items in first worklist
    //     for ( const auto n : wl1 ) {
    //         const auto n_hash = hash( iter_hash ^ hash( n ) );
    //         Tv[n]             = ( n_hash << id_shift ) | static_cast<uint64_t>( begin_row + n + 1
    //         ); AMP_DEBUG_ASSERT( Tv[n] != IN && Tv[n] != OUT );
    //     }

    //     // update all Mv entries from items in second worklist
    //     // this is refresh column from paper
    //     for ( const auto n : wl2 ) {
    //         // set to smallest value in neighborhood
    //         Mv[n] = OUT;
    //         for ( lidx_t k = Ad_rs[n]; k < Ad_rs[n + 1]; ++k ) {
    //             const auto c = Ad_cols_loc[k];
    //             if ( agg_ids[c] < 0 ) {
    //                 Mv[n] = Tv[c] < Mv[n] ? Tv[c] : Mv[n];
    //             }
    //         }
    //         // if smallest is marked IN mark this as OUT
    //         if ( Mv[n] == IN ) {
    //             Mv[n] = OUT;
    //         }
    //     }

    //     // mark undecided as IN or OUT if possible and build new worklists
    //     std::vector<lidx_t> wl1_new;
    //     for ( const auto n : wl1 ) {
    //         const auto rs = Ad_rs[n], re = Ad_rs[n + 1], row_len = re - rs;
    //         // default to IN and check if conditions hold
    //         bool mark_out = false, mark_in = true;
    //         for ( lidx_t k = rs; k < re; ++k ) {
    //             const auto c = Ad_cols_loc[k];
    //             if ( agg_ids[c] >= 0 ) {
    //                 // neighbor is aggregated from previous vertex classification pass
    //                 // ignore on this pass
    //                 continue;
    //             }
    //             if ( Mv[c] == OUT ) {
    //                 mark_out = true;
    //                 break;
    //             }
    //             if ( Mv[c] != Tv[n] ) {
    //                 mark_in = false;
    //             }
    //         }

    //         if ( mark_out ) {
    //             Tv[n] = OUT;
    //         } else if ( mark_in ) {
    //             Tv[n] = IN;
    //         }

    //         // update first worklist
    //         if ( Tv[n] != IN && Tv[n] != OUT ) {
    //             wl1_new.push_back( n );
    //         }
    //     }

    //     // update second work list
    //     std::vector<lidx_t> wl2_new;
    //     for ( lidx_t n = 0; n < A_nrows; ++n ) {
    //         if ( Mv[n] != OUT ) {
    //             wl2_new.push_back( n );
    //         }
    //     }

    //     // swap updated worklists in and loop around
    //     const bool wl1_stag = wl1.size() == wl1_new.size();
    //     const bool wl2_stag = wl2.size() == wl2_new.size();
    //     wl1.swap( wl1_new );
    //     wl2.swap( wl2_new );

    //     ++num_iters;

    //     if ( num_iters == max_iters ) {
    //         AMP_WARNING( "MIS2Aggregator::classifyVertices failed to terminate" );
    //         break;
    //     }

    //     if ( wl1_stag && wl2_stag ) {
    //         AMP_WARNING( "MIS2Aggregator::classifyVertices worklists stagnated" );
    //         AMP::pout << "wl1.size() = " << wl1.size() << ", wl2.size() = " << wl2.size()
    //                   << std::endl;

    //         for ( const auto n : wl1 ) {
    //             Tv[n] = IN;
    //         }

    //         break;
    //     }
    // }

    // return num_iters;
}
#endif

} // namespace AMP::Solver::AMG
