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

    // type specific aggregator, dispatches to host/device versions
    template<typename Config>
    int assignLocalAggregates( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A, int *agg_ids );

    // form aggregated from MIS-2 classification, host only
    template<typename Config>
    int assignLocalAggregatesHost( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                   int *agg_ids );

    // classify vertices as in or out of MIS-2
    template<typename Config>
    int classifyVerticesHost( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A,
                              const uint64_t num_gbl,
                              std::vector<typename Config::lidx_t> &wl1,
                              std::vector<uint64_t> &labels,
                              int *agg_ids );

    // device versions of aggregation/classification
#ifdef AMP_USE_DEVICE
    template<typename Config>
    int assignLocalAggregatesDevice( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                                     int *agg_ids );
    template<typename Config>
    int classifyVerticesDevice( std::shared_ptr<LinearAlgebra::CSRLocalMatrixData<Config>> A,
                                const uint64_t num_gbl,
                                typename Config::lidx_t *worklist,
                                typename Config::lidx_t worklist_len,
                                uint64_t *Tv,
                                uint64_t *Tv_hat,
                                int *agg_ids );
#endif

    // helper function to choose bits for id part of packed tuples
    uint64_t getIdMask( uint64_t num_global ) const
    {
        // the packed representation uses minimal number of bits for ID part
        // of tuple, get log_2 of (num_gbl + 2)
        AMP_ASSERT( num_global < ( std::numeric_limits<uint64_t>::max() - 33 ) );
        const auto id_shift = []( uint64_t ng ) -> uint8_t {
            // log2 from stackoverflow. If only bit_width was c++17...
            uint8_t s = 1;
            while ( ng >>= 1 )
                ++s;
            return s;
        }( num_global );
        return std::numeric_limits<uint64_t>::max() >> ( 64 - id_shift );
    }

    // helper function to choose bits for hash part of packed tuples
    uint64_t getHashMask( uint64_t id_mask ) const
    {
        const uint64_t conn_mask = ( (uint64_t) 31 ) << 59;
        return ~( conn_mask | id_mask );
    }

    // status labels such that IN < UNDECIDED < OUT
    // there is no ordering within the IN set so all are marked 0,
    // similarly all OUT are marked with max value
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
