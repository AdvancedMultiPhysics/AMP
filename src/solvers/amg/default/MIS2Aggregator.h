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

    // type specific aggregator
    template<typename Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );

    // classify vertices as in or out of MIS-2
    template<typename Config>
    int classifyVertices( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A,
                          std::vector<typename Config::lidx_t> &wl1,
                          std::vector<uint64_t> &labels,
                          const uint64_t num_gbl,
                          int *agg_ids );

    // status labels such that IN < UNDECIDED < OUT
    // there is no ordering within the IN set so all are marked 0,
    // similarly all OUT are marked with max value
    static constexpr uint64_t IN  = 0;
    static constexpr uint64_t OUT = std::numeric_limits<uint64_t>::max();
};

} // namespace AMP::Solver::AMG

#endif
