#ifndef included_AMP_AMG_AggregationSettings
#define included_AMP_AMG_AggregationSettings

#include <string>

namespace AMP::Solver::AMG {

struct CoarsenSettings {
    float strength_threshold;
    std::string strength_measure;
    bool checkdd;
};

struct PairwiseCoarsenSettings : CoarsenSettings {
    size_t pairwise_passes;
    PairwiseCoarsenSettings &operator=( const CoarsenSettings &other )
    {
        strength_threshold = other.strength_threshold;
        strength_measure   = other.strength_measure;
        checkdd            = other.checkdd;
        return *this;
    }
};

} // namespace AMP::Solver::AMG
#endif
