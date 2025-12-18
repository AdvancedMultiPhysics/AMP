#ifndef included_AMP_AMG_AggregationSettings
#define included_AMP_AMG_AggregationSettings

#include <string>

namespace AMP::Solver::AMG {

struct CoarsenSettings {
    float strength_threshold;
    std::string strength_measure;
};

struct PairwiseCoarsenSettings : CoarsenSettings {
    size_t pairwise_passes;
    bool checkdd;
    PairwiseCoarsenSettings &operator=( const CoarsenSettings &other )
    {
        strength_threshold = other.strength_threshold;
        strength_measure   = other.strength_measure;
        return *this;
    }
};

} // namespace AMP::Solver::AMG
#endif
