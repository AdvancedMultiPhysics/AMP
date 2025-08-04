#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
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

    return LinearAlgebra::csrVisit(
        A, [=]( auto csr_ptr ) { return assignLocalAggregates( csr_ptr, agg_ids ); } );
}

template<typename Config>
int SimpleAggregator::assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                             int *agg_ids )
{
    PROFILE( "SimpleAggregator::assignLocalAggregates" );

    using lidx_t       = typename Config::lidx_t;
    using scalar_t     = typename Config::scalar_t;
    using matrix_t     = LinearAlgebra::CSRMatrix<Config>;
    using matrixdata_t = typename matrix_t::matrixdata_t;

    // information about A and unpack diag block
    const auto A_nrows = static_cast<lidx_t>( A->numLocalRows() );
    auto A_data        = std::dynamic_pointer_cast<matrixdata_t>( A->getMatrixData() );
    auto A_diag        = A_data->getDiagMatrix();
    auto [Ad_rs, Ad_cols, Ad_cols_loc, Ad_coeffs] = A_diag->getDataFields();

    // fill initial ids with -1's to mark as not associated
    AMP::Utilities::Algorithms<int>::fill_n( agg_ids, A_nrows, -1 );

    // Create temporary storage for aggregate sizes and row strengths
    std::vector<lidx_t> agg_size;
    std::vector<scalar_t> Sij;

    // first pass initilizes aggregates from nodes that have no
    // neighbors that are already associated
    // NOTE: there are several loops through the entries in a row
    //       that could be collapsed into a single loop
    int num_agg = 0;
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        const auto rs = Ad_rs[row], re = Ad_rs[row + 1], row_len = re - rs;

        // do not form aggregates from rows with only one entry
        // or rows contained in a prior aggregate
        if ( row_len == 1 || agg_ids[row] >= 0 ) {
            continue;
        }

        // Check if any members of this row are already associated
        // and skip if so.
        // Also find the strength-of-connection numerator.
        bool have_nbr      = false;
        scalar_t soc_numer = 0.0;
        for ( lidx_t c = rs + 1; c < re; ++c ) {
            have_nbr = have_nbr || agg_ids[Ad_cols_loc[c]] < 0;
            soc_numer += std::fabs( Ad_coeffs[c] );
        }

        // skip if all nbrs are already aggregated, this should be very rare
        if ( !have_nbr ) {
            continue;
        }

        // fill in strength of connection and find minimum value
        Sij.resize( row_len );
        Sij[0]                 = 0.0;
        const scalar_t scl_max = std::numeric_limits<scalar_t>::max();
        scalar_t min_str       = std::sqrt( scl_max );
        for ( lidx_t n = 1; n < row_len; ++n ) {
            const bool do_soc = Ad_coeffs[rs + n] < 0.0 && agg_ids[Ad_cols_loc[rs + n]] < 0;
            Sij[n]            = do_soc ? std::fabs( 1.0 + soc_numer / Ad_coeffs[rs + n] ) : scl_max;
            min_str           = std::min( Sij[n], min_str );
        }
        auto Sij_sort = Sij;
        std::sort( Sij_sort.data(), Sij_sort.data() + row_len );

        // create new aggregate from row
        scalar_t thresh = Sij_sort[std::min( d_max_agg_size - 1, row_len - 1 )];
        if ( d_weak_thresh > 0.0 ) {
            thresh = std::fmin( d_weak_thresh * min_str, thresh );
        }

        agg_size.push_back( 0 );
        for ( lidx_t n = 0; n < row_len; ++n ) {
            if ( Ad_coeffs[rs + n] != 0.0 && Sij[n] <= thresh &&
                 agg_size[num_agg] < d_max_agg_size ) {
                agg_ids[Ad_cols_loc[rs + n]] = num_agg;
                agg_size[num_agg]++;
            }
        }

        // ignore rows with no strong connections, this happens with how
        // Dirichlet conditions get applied in some problems and is not
        // caught by row length check above
        if ( agg_size[num_agg] <= 1 ) {
            agg_size.pop_back();
            agg_ids[Ad_cols_loc[rs]] = -1;
            continue;
        }

        // increment current id to start working on next aggregate
        ++num_agg;
    }

    // second pass adds unmarked entries to the smallest aggregate they are nbrs with
    // entries are unmarked because they neigbored some aggregate in the above,
    // thus every unmarked entry will neighbor some aggregate
    for ( lidx_t row = 0; row < A_nrows; ++row ) {
        if ( agg_ids[row] >= 0 ) {
            // this row already assigned, skip ahead
            continue;
        }

        // find smallest neighboring aggregate
        lidx_t small_agg_id = -1, small_agg_size = A_nrows + 1;
        for ( lidx_t c = Ad_rs[row]; c < Ad_rs[row + 1]; ++c ) {
            const auto id = agg_ids[Ad_cols_loc[c]];
            // only consider nbrs that are aggregated
            if ( id >= 0 && ( agg_size[id] < small_agg_size ) ) {
                small_agg_size = agg_size[id];
                small_agg_id   = id;
            }
        }

        // Possible that no nbr is found
        if ( small_agg_id >= 0 ) {
            agg_ids[row] = small_agg_id;
            agg_size[small_agg_id]++;
        }
    }

    return num_agg;
}

} // namespace AMP::Solver::AMG
