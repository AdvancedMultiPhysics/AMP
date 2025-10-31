#ifndef included_AMP_AMG_AggregationSettings
#define included_AMP_AMG_AggregationSettings

#include <string>

namespace AMP::Solver::AMG {

struct CoarsenSettings {
    float strength_threshold;
    int min_coarse_local;
    size_t min_coarse;
    std::string strength_measure;
};

struct PairwiseCoarsenSettings : CoarsenSettings {
    size_t pairwise_passes;
    bool checkdd;
    PairwiseCoarsenSettings &operator=( const CoarsenSettings &other )
    {
        strength_threshold = other.strength_threshold;
        min_coarse_local   = other.min_coarse_local;
        min_coarse         = other.min_coarse;
        strength_measure   = other.strength_measure;
        return *this;
    }
};

} // namespace AMP::Solver::AMG
#endif
