#ifndef included_AMP_MIS2Aggregator_H_
#define included_AMP_MIS2Aggregator_H_

#include "AMP/solvers/amg/AggregationSettings.h"
#include "AMP/solvers/amg/Aggregator.h"

#include <vector>

namespace AMP::Solver::AMG {

// This aggregator is based on an MIS-2 classification of the vertices
// The implementation follows Sandia report SAND2022-2930C titled
// "Parallel, portable algorithms for distance-2 maximal independent
//  set and graph coarsening" by Brian Kelley and Sivasankaran
// Rajamanickam
struct MIS2Aggregator : Aggregator {
    MIS2Aggregator( const CoarsenSettings &settings ) : Aggregator( settings ) {}

    // Necessary overrides from base class
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::Matrix> A, int *agg_ids ) override;

    // type specific aggregator, dispatches to host/device impls
    template<typename Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );

    // host implementation of aggregator
    template<typename Config>
    int assignLocalAggregatesHost( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                   int *agg_ids );

    // classify vertices as in or out of MIS-2
    template<typename Config>
    int classifyVerticesHost( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A,
                              std::vector<typename Config::lidx_t> &wl1,
                              std::vector<uint64_t> &labels,
                              const uint64_t num_gbl,
                              int *agg_ids );

#ifdef AMP_USE_DEVICE
    template<typename Config>
    int assignLocalAggregatesDevice( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                     int *agg_ids );
    template<typename Config>
    int classifyVerticesDevice( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A,
                                typename Config::lidx_t *wl1,
                                typename Config::lidx_t *wl2,
                                uint64_t *Tv,
                                uint64_t *Mv,
                                const uint64_t num_gbl,
                                int *agg_ids );
#endif

    // status labels such that OUT < UNDECIDED < IN
    // there is no ordering within the OUT set so all are marked 0,
    // similarly all IN are marked with max value
    static constexpr uint64_t IN  = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t OUT = 0;

    // Aggregate IDs are signed, where nonegative values are the
    // assigned aggregate determined by this process.
    // Negative values are used as semaphores for two cases,
    // unaggregated points valid for assignment, and invalid
    // points that are not to be aggregated at all
    static constexpr int UNASSIGNED = -1;
    static constexpr int INVALID    = -2;
};

} // namespace AMP::Solver::AMG

#endif
