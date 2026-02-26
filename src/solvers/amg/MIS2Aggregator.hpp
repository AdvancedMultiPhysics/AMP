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
    #define AMP_FUNCTION_HD __host__ __device__
    #define AMP_FUNCTION_G __global__
#else
    #define AMP_FUNCTION_HD
    #define AMP_FUNCTION_G
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
int MIS2Aggregator::classifyVertices(
    std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A_diag,
    const uint64_t num_gbl,
    typename Config::lidx_t *worklist,
    typename Config::lidx_t worklist_len,
    uint64_t *Tv,
    uint64_t *Tv_hat )
{
    PROFILE( "MIS2Aggregator::classifyVertices" );

    using lidx_t   = typename Config::lidx_t;
    using gidx_t   = typename Config::gidx_t;
    using scalar_t = typename Config::scalar_t;

    constexpr bool host_exec =
        std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>>;

    // unpack diag block
    const auto begin_row = A_diag->beginRow();
    lidx_t *Ad_rs = nullptr, *Ad_cols_loc = nullptr;
    gidx_t *Ad_cols                                    = nullptr;
    scalar_t *Ad_coeffs                                = nullptr;
    std::tie( Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs ) = A_diag->getDataFields();

    // hash is xorshift* as given on wikipedia
    auto hash = [] AMP_FUNCTION_HD( uint64_t x ) -> uint64_t {
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
    auto pack_tuple = [begin_row, hash_mask, hash] AMP_FUNCTION_HD(
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
    auto idx_from_tuple = [begin_row, id_mask] AMP_FUNCTION_HD( uint64_t tpl ) -> lidx_t {
        const auto v = tpl & id_mask;
        return static_cast<lidx_t>( v - begin_row );
    };

    const lidx_t max_iters = 20;
    int num_iters          = 0;
    while ( worklist_len > 0 && num_iters < max_iters ) {
        const auto iter_hash = hash( num_iters + 1 );

        // first update Tv entries from items in worklist
        // this is "refresh row" from paper
        {
            auto ref_row = [Ad_rs, iter_hash, pack_tuple, Tv] AMP_FUNCTION_HD( lidx_t n ) -> void {
                const uint8_t conn = Ad_rs[n + 1] - Ad_rs[n] - 1; // ignore self connection
                Tv[n]              = pack_tuple( n, conn, iter_hash );
            };
            if constexpr ( host_exec ) {
                std::for_each_n( worklist, worklist_len, ref_row );
            } else {
#ifdef AMP_USE_DEVICE
                thrust::for_each( thrust::device, worklist, worklist + worklist_len, ref_row );
#endif
            }
        }

        // Store largest Tv entry from each neighborhood into Tv_hat,
        // then swap Tv and Tv_hat, repeat, leaving max 2-nbr entry
        // in Tv
        for ( int nrep = 0; nrep < 2; ++nrep ) {
            auto nbr_max = [Ad_rs, Ad_cols_loc, Tv, Tv_hat] AMP_FUNCTION_HD( lidx_t n ) -> void {
                const auto rs = Ad_rs[n], re = Ad_rs[n + 1];
                auto lmax = Tv[n];
                for ( lidx_t k = rs; k < re; ++k ) {
                    const auto c = Ad_cols_loc[k];
                    lmax         = lmax < Tv[c] ? Tv[c] : lmax;
                }
                Tv_hat[n] = lmax;
            };
            if constexpr ( host_exec ) {
                std::for_each_n( worklist, worklist_len, nbr_max );
            } else {
#ifdef AMP_USE_DEVICE
                thrust::for_each( thrust::device, worklist, worklist + worklist_len, nbr_max );
#endif
            }
            std::swap( Tv, Tv_hat );
        }

        // mark undecided as IN or OUT if possible
        {
            auto in_out = [idx_from_tuple, Tv] AMP_FUNCTION_HD( const lidx_t n ) -> void {
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
            };
            if constexpr ( host_exec ) {
                std::for_each_n( worklist, worklist_len, in_out );
            } else {
#ifdef AMP_USE_DEVICE
                thrust::for_each( thrust::device, worklist, worklist + worklist_len, in_out );
#endif
            }
        }

        // immediately mark nbrs of IN nodes as OUT
        {
            auto set_out = [Ad_rs, Ad_cols_loc, Tv] AMP_FUNCTION_HD( const lidx_t n ) -> void {
                if ( Tv[n] == IN ) {
                    return;
                }
                const auto rs = Ad_rs[n], re = Ad_rs[n + 1];
                bool nbr_in = false;
                for ( lidx_t k = rs; k < re; ++k ) {
                    const auto c = Ad_cols_loc[k];
                    nbr_in       = nbr_in || Tv[c] == IN;
                }
                Tv[n] = nbr_in ? OUT : Tv[n];
            };
            if constexpr ( host_exec ) {
                std::for_each_n( worklist, worklist_len, set_out );
            } else {
#ifdef AMP_USE_DEVICE
                thrust::for_each( thrust::device, worklist, worklist + worklist_len, set_out );
#endif
            }
        }

        // test if we are done
        {
            auto in_out = [Tv] AMP_FUNCTION_HD( const lidx_t n ) -> bool {
                return ( Tv[n] == IN || Tv[n] == OUT );
            };
            if constexpr ( host_exec ) {
                lidx_t *new_end = std::remove_if( worklist, worklist + worklist_len, in_out );
                worklist_len    = static_cast<lidx_t>( new_end - worklist );
            } else {
#ifdef AMP_USE_DEVICE
                lidx_t *new_end =
                    thrust::remove_if( thrust::device, worklist, worklist + worklist_len, in_out );
                worklist_len = static_cast<lidx_t>( new_end - worklist );
#endif
            }
        }

        ++num_iters;
    }

    return num_iters;
}

// Helper functions, decorated for device use if supported
template<typename lidx_t>
AMP_FUNCTION_HD void agg_from_row( const lidx_t row,
                                   const lidx_t *row_start,
                                   const lidx_t *cols_loc,
                                   uint64_t *labels,
                                   lidx_t *agg_size,
                                   lidx_t *agg_ids )
{
    if ( labels[row] != MIS2Aggregator::IN || agg_ids[row] != MIS2Aggregator::UNASSIGNED ) {
        // not (valid) root node, nothing to do
        return;
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
AMP_FUNCTION_HD void grow_agg( const lidx_t row,
                               const lidx_t num_rows,
                               const lidx_t *row_start,
                               const lidx_t *cols_loc,
                               lidx_t *agg_size,
                               lidx_t *agg_ids )
{
    if ( agg_ids[row] != MIS2Aggregator::UNASSIGNED ) {
        // already aggregated or invalid, nothing to do
        return;
    }

    // find smallest neighboring aggregate
    const auto rs = row_start[row], re = row_start[row + 1];
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

template<typename Config>
int MIS2Aggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                           int *agg_ids )
{
    PROFILE( "MIS2Aggregator::assignLocalAggregatesDevice" );

    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;
    using matrix_t          = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t      = typename matrix_t::matrixdata_t;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    constexpr bool host_exec =
        std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>>;

    // Get diag block from A and mask it using SoC
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );

    // get fields from A and use to make diagonal-dominance checker
    auto A_diag   = A_data->getDiagMatrix();
    lidx_t *Ad_rs = nullptr, *Ad_cols_loc = nullptr;
    gidx_t *Ad_cols     = nullptr;
    scalar_t *Ad_coeffs = nullptr;
    lidx_t *Ao_rs = nullptr, *Ao_cols_loc = nullptr;
    gidx_t *Ao_cols                                    = nullptr;
    scalar_t *Ao_coeffs                                = nullptr;
    std::tie( Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs ) = A_diag->getDataFields();
    const bool have_offd                               = A_data->hasOffDiag();
    if ( have_offd ) {
        auto A_offd                                        = A_data->getOffdMatrix();
        std::tie( Ao_rs, Ao_cols, Ao_cols_loc, Ao_coeffs ) = A_offd->getDataFields();
    }

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

    // get temporary storage for aggregate sizes and MIS2 labels
    auto Tv     = localmatrixdata_t::template sharedArrayBuilder<uint64_t>( A_nrows );
    auto Tv_hat = localmatrixdata_t::template sharedArrayBuilder<uint64_t>( A_nrows );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv.get(), A_nrows, OUT );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv_hat.get(), A_nrows, OUT );
    auto agg_size       = localmatrixdata_t::template sharedArrayBuilder<int>( A_nrows );
    auto agg_root_ids   = localmatrixdata_t::template sharedArrayBuilder<int>( A_nrows );
    auto worklist       = localmatrixdata_t::makeLidxArray( A_nrows );
    lidx_t worklist_len = A_nrows;
    AMP::Utilities::Algorithms<int>::fill_n( agg_size.get(), A_nrows, -1 );
    AMP::Utilities::Algorithms<int>::fill_n( agg_root_ids.get(), A_nrows, UNASSIGNED );

    // Initialize ids to either unassigned (default) or invalid (isolated)
    {
        auto agg_root_ids_ptr = agg_root_ids.get();
        const bool checkdd    = d_checkdd;
        auto not_diag_dom =
            [checkdd, Ad_rs, Ad_coeffs, have_offd, Ao_rs, Ao_coeffs] AMP_FUNCTION_HD(
                const lidx_t row ) -> bool {
            if ( !checkdd ) {
                return true;
            }
            scalar_t od_sum = 0.0;
            for ( lidx_t k = Ad_rs[row] + 1; k < Ad_rs[row + 1]; ++k ) {
                const auto c = Ad_coeffs[k];
                od_sum += ( c > -c ? c : -c );
            }
            if ( have_offd ) {
                for ( lidx_t k = Ao_rs[row]; k < Ao_rs[row + 1]; ++k ) {
                    const auto c = Ao_coeffs[k];
                    od_sum += ( c > -c ? c : -c );
                }
            }
            return Ad_coeffs[Ad_rs[row]] <= scalar_t{ 5.0 } * od_sum;
        };
        auto mark_undec_inv =
            [Am_rs, not_diag_dom, agg_root_ids_ptr] AMP_FUNCTION_HD( const lidx_t row ) -> void {
            const auto rs = Am_rs[row], re = Am_rs[row + 1];
            if ( re - rs > 1 && not_diag_dom( row ) ) {
                agg_root_ids_ptr[row] = MIS2Aggregator::UNASSIGNED;
            } else {
                agg_root_ids_ptr[row] = MIS2Aggregator::INVALID;
            }
        };
        if constexpr ( host_exec ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                mark_undec_inv( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( A_nrows ),
                              mark_undec_inv );
#endif
        }
    }

    // initialize worklist to all UNASSIGNED rows
    {
        if constexpr ( host_exec ) {
            worklist_len = 0;
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                if ( agg_root_ids[row] == UNASSIGNED ) {
                    worklist[worklist_len++] = row;
                }
            }
        } else {
#ifdef AMP_USE_DEVICE
            auto agg_root_ids_ptr = agg_root_ids.get();
            lidx_t *new_end =
                thrust::copy_if( thrust::device,
                                 thrust::make_counting_iterator( 0 ),
                                 thrust::make_counting_iterator( A_nrows ),
                                 worklist.get(),
                                 [agg_root_ids_ptr] __device__( const lidx_t n ) -> bool {
                                     return ( agg_root_ids_ptr[n] == UNASSIGNED );
                                 } );
            worklist_len = static_cast<lidx_t>( new_end - worklist.get() );
#endif
        }
    }

    // First pass of MIS2 classification, ignores INVALID nodes
    classifyVertices<Config>( A_masked,
                              A->numGlobalRows(),
                              worklist.get(),
                              worklist_len,
                              Tv.get(),
                              Tv_hat.get() );

    // initialize aggregates from nodes flagged as IN and all of their neighbors
    {
        auto Tv_ptr           = Tv.get();
        auto agg_size_ptr     = agg_size.get();
        auto agg_root_ids_ptr = agg_root_ids.get();
        auto build_agg =
            [Am_rs, Am_cols_loc, Tv_ptr, agg_size_ptr, agg_root_ids_ptr] AMP_FUNCTION_HD(
                const lidx_t row ) -> void {
            agg_from_row( row, Am_rs, Am_cols_loc, Tv_ptr, agg_size_ptr, agg_root_ids_ptr );
        };
        if constexpr ( host_exec ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                build_agg( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( A_nrows ),
                              build_agg );
#endif
        }
    }

    // re-initialize worklist to all UNASSIGNED rows
    {
        if constexpr ( host_exec ) {
            worklist_len = 0;
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                if ( agg_root_ids[row] == UNASSIGNED ) {
                    worklist[worklist_len++] = row;
                }
            }
        } else {
#ifdef AMP_USE_DEVICE
            auto agg_root_ids_ptr = agg_root_ids.get();
            lidx_t *new_end =
                thrust::copy_if( thrust::device,
                                 thrust::make_counting_iterator( 0 ),
                                 thrust::make_counting_iterator( A_nrows ),
                                 worklist.get(),
                                 [agg_root_ids_ptr] __device__( const lidx_t n ) -> bool {
                                     return ( agg_root_ids_ptr[n] == UNASSIGNED );
                                 } );
            worklist_len = static_cast<lidx_t>( new_end - worklist.get() );
#endif
        }
    }

    // do a second pass of classification and aggregation
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv.get(), A_nrows, OUT );
    AMP::Utilities::Algorithms<uint64_t>::fill_n( Tv_hat.get(), A_nrows, OUT );
    classifyVertices<Config>( A_masked,
                              A->numGlobalRows(),
                              worklist.get(),
                              worklist_len,
                              Tv.get(),
                              Tv_hat.get() );

    // on second pass only allow IN vertex to be root of aggregate if it has
    // at least 2 un-aggregated nbrs
    {
        auto Tv_ptr           = Tv.get();
        auto agg_size_ptr     = agg_size.get();
        auto agg_root_ids_ptr = agg_root_ids.get();
        auto build_agg =
            [Am_rs, Am_cols_loc, Tv_ptr, agg_size_ptr, agg_root_ids_ptr] AMP_FUNCTION_HD(
                const lidx_t row ) -> void {
            agg_from_row( row, Am_rs, Am_cols_loc, Tv_ptr, agg_size_ptr, agg_root_ids_ptr );
        };
        if constexpr ( host_exec ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                build_agg( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( A_nrows ),
                              build_agg );
#endif
        }
    }

    // deallocate Tv and Tv_hat, they are no longer needed
    Tv.reset();
    Tv_hat.reset();

    // Add unmarked entries to the smallest aggregate they are nbrs with
    // this differs from host version, here sizes are not updated during
    // growth to avoid race conditions. "smallest aggregate" now means
    // smallest from above steps only. Also, for simplicity it just gets
    // called twice
    for ( int ngrow = 0; ngrow < 3; ++ngrow ) {
        auto agg_size_ptr     = agg_size.get();
        auto agg_root_ids_ptr = agg_root_ids.get();
        auto grow_aggs =
            [A_nrows, Am_rs, Am_cols_loc, agg_size_ptr, agg_root_ids_ptr] AMP_FUNCTION_HD(
                const lidx_t row ) -> void {
            grow_agg( row, A_nrows, Am_rs, Am_cols_loc, agg_size_ptr, agg_root_ids_ptr );
        };
        if constexpr ( host_exec ) {
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                grow_aggs( row );
            }
        } else {
#ifdef AMP_USE_DEVICE
            thrust::for_each( thrust::device,
                              thrust::make_counting_iterator( 0 ),
                              thrust::make_counting_iterator( A_nrows ),
                              grow_aggs );
#endif
        }
    }

    // agg_root_ids currently uses root node of each aggregate as the ID
    // need to count number of unique ones to get total number of aggregates
    // agg_size is no longer needed, so copy ids to there, sort, call unique
    int num_agg = 0, num_dec = 0;
    auto unq_root_ids = agg_size; // rename for clarity
    {
        AMP::Utilities::Algorithms<lidx_t>::copy_n(
            agg_root_ids.get(), A_nrows, unq_root_ids.get() );
        AMP::Utilities::Algorithms<lidx_t>::sort( unq_root_ids.get(), A_nrows );
        const auto nunq = AMP::Utilities::Algorithms<lidx_t>::unique( unq_root_ids.get(), A_nrows );
        // need to check first two entries of unique'd array
        // if we have UNDECIDED or INVALID need to decrement agg count
        lidx_t first_entries[2];
        AMP::Utilities::Algorithms<lidx_t>::copy_n( unq_root_ids.get(), 2, first_entries );
        const int dec_inv = ( first_entries[0] == INVALID || first_entries[1] == INVALID ) ? 1 : 0;
        const int dec_und =
            ( first_entries[0] == UNASSIGNED || first_entries[1] == UNASSIGNED ) ? 1 : 0;
        num_dec = dec_inv + dec_und;
        num_agg = static_cast<int>( nunq ) - num_dec;
    }

    // finally, use compacted list of uniques to convert root node labels
    // to consecutive ascending labels. Do need an additional work vector to
    {
        if constexpr ( host_exec ) {
            // search for root node ids in list of uniques and offset back
            // to remove invalid/undecided entries
            for ( lidx_t row = 0; row < A_nrows; ++row ) {
                const auto unq_agg = std::lower_bound(
                    unq_root_ids.get(), unq_root_ids.get() + num_agg, agg_root_ids[row] );
                agg_ids[row] =
                    static_cast<lidx_t>( std::distance( unq_root_ids.get(), unq_agg ) ) - num_dec;
            }
        } else {
#ifdef AMP_USE_DEVICE
            // search for root node ids in list of uniques to create local ids
            thrust::lower_bound( thrust::device,
                                 unq_root_ids.get(),
                                 unq_root_ids.get() + num_agg,
                                 agg_root_ids.get(),
                                 agg_root_ids.get() + A_nrows,
                                 agg_ids );
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
#endif
        }
    }

    return num_agg;
}

} // namespace AMP::Solver::AMG

#undef AMP_FUNCTION_HD
#undef AMP_FUNCTION_G
