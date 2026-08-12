#ifndef included_AMP_AMG_AggregationSettings
#define included_AMP_AMG_AggregationSettings

#include <string>

namespace AMP::Solver::AMG {

/**
   Flags used to mark unknowns for aggregation.

   Aggregate IDs are signed, where nonegative values are the
   assigned aggregate determined by this process.
   Negative values are used as semaphores for two cases,
   unaggregated points valid for assignment, and invalid
   points that are not to be aggregated at all
*/
enum AggregationFlags : int {
    /// unknowns eligible, but not yet selected by aggregation.
    eligible = -1,
    /**
     * unknowns not suitable for aggregation (should not be selected).
     *
     * This can be used to mark unknowns that should not be considered for
     * aggregation.  For example, strongly diagonally dominant rows that
     * may be unsuitable for coarse-grid correction.
     */
    ineligible = -2
};

struct CoarsenSettings {
    float strength_threshold;
    std::string strength_measure;
    bool checkdd;
    //! Redistribution coarsening factor for communicator size
    int redist_coarsen_factor;
};

struct PairwiseCoarsenSettings : CoarsenSettings {
    size_t pairwise_passes;
    PairwiseCoarsenSettings &operator=( const CoarsenSettings &other )
    {
        strength_threshold    = other.strength_threshold;
        strength_measure      = other.strength_measure;
        checkdd               = other.checkdd;
        redist_coarsen_factor = other.redist_coarsen_factor;
        return *this;
    }
};

} // namespace AMP::Solver::AMG
#endif
