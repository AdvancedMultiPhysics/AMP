#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/Strength.hpp"
#include "AMP/solvers/amg/default/SimpleAggregator.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/vectors/CommunicationList.h"

#include "ProfilerApp.h"

#include <algorithm>

namespace AMP::Solver::AMG {

int SimpleAggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A,
                                             int *agg_ids )
{
    AMP_DEBUG_INSIST( A->numLocalRows() == A->numLocalColumns(),
                      "SimpleAggregator::assignLocalAggregates input matrix must be square" );
    AMP_DEBUG_ASSERT( agg_ids != nullptr );

    return LinearAlgebra::csrVisit( A, [this, agg_ids]( auto csr_ptr ) {
        return this->assignLocalAggregates( csr_ptr, agg_ids );
    } );
}

template<typename Config>
int SimpleAggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                             int *agg_ids )
{
    PROFILE( "SimpleAggregator::assignLocalAggregates" );

    using lidx_t       = typename Config::lidx_t;
    using matrix_t     = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t = typename matrix_t::matrixdata_t;

    // get strength information
    auto S = compute_soc<evolution_strength>( csr_view( *A ), d_strength_threshold );

    // Get diag block from A and mask it using SoC
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();
    auto A_masked      = A_diag->maskMatrixData( S.diag_mask_data(), true );

    // pull out data fields from A_masked
    // only care about row starts and local cols
    auto [Am_rs, Am_cols, Am_cols_loc, Am_coeffs] = A_masked->getDataFields();

    // fill initial ids with -1's to mark as not associated
    AMP::Utilities::Algorithms<int>::fill_n( agg_ids, A_nrows, -1 );

    // Create temporary storage for aggregate sizes
    std::vector<lidx_t> agg_size;

    // flags for isolated points, 0 undecided, 1 marked isolated, -1 marked un-isolated
    std::vector<lidx_t> isolated_pts( A_nrows, 0 );

    // first pass initilizes aggregates from nodes that have no
    // neighbors that are already associated
    int num_agg = 0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1], row_len = re - rs;

        // skip already aggregated rows
        if ( agg_ids[row] >= 0 ) {
            continue;
        }

        // mark single entry or empty rows as isolated and do not aggregate
        if ( row_len <= 1 ) {
            AMP_DEBUG_ASSERT( isolated_pts[row] == 0 ); // should be undecided if not agg'd
            isolated_pts[row] = 1;
            continue;
        }

        // Check if any members of this row are already associated and skip if so.
        bool have_nbrs = true;
        for ( lidx_t c = rs + 1; c < re; ++c ) {
            const auto nid = Am_cols_loc[c];
            have_nbrs      = have_nbrs && agg_ids[nid] < 0;
        }
        if ( !have_nbrs ) {
            // does not have all nbrs available for aggregation, skip
            continue;
        }

        // create new aggregate from row
        agg_size.push_back( 0 );
        for ( lidx_t n = 0; n < row_len; ++n ) {
            const auto col_idx = Am_cols_loc[rs + n];
            agg_ids[col_idx]   = num_agg;
            agg_size[num_agg]++;
            // steal any isolated points that can be lumped into this aggregate
            isolated_pts[col_idx] = 0;
        }

        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // Second pass, add unaggregated points to the smallest aggregate
    // that they neighbor. Does nothing to isolated points
    bool grew_agg;
    lidx_t npasses = 0;
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
                // steal if isolated
                isolated_pts[row] = 0;
                grew_agg          = true;
            }
        }
        ++npasses;
    } while ( grew_agg );
    AMP::pout << "Agg growth took " << npasses << " passes" << std::endl;

    // Third pass, check if aggregated points neighbor any isolated points
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
            if ( isolated_pts[nid] == 1 ) {
                agg_ids[nid] = curr_agg;
                agg_size[curr_agg]++;
                isolated_pts[nid] = 0;
            }
        }
    }

    // DEBUG
    {
        double total_agg   = 0.0;
        lidx_t largest_agg = 0, smallest_agg = A_nrows;
        for ( int n = 0; n < num_agg; ++n ) {
            total_agg += agg_size[n];
            largest_agg  = largest_agg < agg_size[n] ? agg_size[n] : largest_agg;
            smallest_agg = smallest_agg > agg_size[n] ? agg_size[n] : smallest_agg;
        }
        AMP::pout << "SimpleAggregator found " << num_agg << " aggregates over " << A_nrows
                  << " rows, with average size " << total_agg / static_cast<double>( num_agg )
                  << ", and max/min " << largest_agg << "/" << smallest_agg << std::endl;
    }

    return num_agg;
}

} // namespace AMP::Solver::AMG
