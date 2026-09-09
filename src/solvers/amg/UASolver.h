#ifndef included_AMP_UAAMGSolver
#define included_AMP_UAAMGSolver

#include "AMP/solvers/amg/AggregationSolver.h"

#include <limits>

namespace AMP::Solver::AMG {

struct UASolver : AggregationSolver {
    explicit UASolver( std::shared_ptr<SolverStrategyParameters> params )
        : AggregationSolver( params, "UASolver" )
    {
        finishConstruction();
    }

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return createAggregationSolver<UASolver>( params );
    }

private:
    size_t defaultKcycleKappa() const override { return std::numeric_limits<size_t>::max(); }
    bool defaultPropagateNearNull() const override { return false; }
    float defaultStrengthThreshold() const override { return 0.25f; }
    int defaultNumSmoothProl() const override { return 0; }
    std::string defaultAggType() const override { return "MIS2"; }
    bool supportsImplicitRAP() const override { return true; }
    bool usesLevelOptions() const override { return false; }
    bool relaxationUsesSourceOperator() const override { return true; }
    bool shouldRegisterCoarseOperator() const override { return false; }
    std::shared_ptr<AMP::Database> createInternalOperatorDb() const override;
    PairwiseCoarsenSettings getCoarsenSettings( size_t lvl ) const override;
};

} // namespace AMP::Solver::AMG

#endif
