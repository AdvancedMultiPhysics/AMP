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

    // second pass
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Am_rs[row], re = Am_rs[row + 1];
        if ( agg_ids[row] >= 0 ) {
            // this row already assigned, check if it contains
            // any isolated points and add them to this aggregate if so
            const auto curr_agg = agg_ids[row];
            for ( lidx_t c = rs + 1; c < re; ++c ) {
                const auto id = Am_cols_loc[c];
                if ( isolated_pts[id] == 1 ) {
                    agg_ids[id] = curr_agg;
                    agg_size[curr_agg]++;
                    isolated_pts[id] = 0;
                }
            }
            continue;
        }

        // find smallest neighboring aggregate
        lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
        for ( lidx_t c = rs + 1; c < re; ++c ) {
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
        }
    }

    // third pass
    bool have_isolated = false;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( isolated_pts[row] != 1 || agg_ids[row] >= 0 ) {
            // point is not isolated or is already aggregated
            continue;
        }
        if ( !have_isolated ) {
            // this is the first isolated point found, make new agg
            agg_size.push_back( 0 );
        }
        have_isolated = true;
        agg_ids[row]  = num_agg;
        agg_size[num_agg]++;
    }

    // account for lumped isolated rows in aggregate count
    if ( have_isolated ) {
        std::cout << "SimpleAggregator had " << agg_size[num_agg] << " isolated points"
                  << std::endl;
        ++num_agg;
    }

    return num_agg;
}

} // namespace AMP::Solver::AMG
