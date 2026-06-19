#ifndef included_AMP_UAAMGSolver
#define included_AMP_UAAMGSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/solvers/amg/Cycle.h"
#include "AMP/solvers/amg/Relaxation.h"
#include "AMP/utils/Flags.h"
#include "AMP/utils/GroupedRedistributionPlan.h"

#include <optional>

namespace AMP::Solver::AMG {

struct UASolver : SolverStrategy {
    explicit UASolver( std::shared_ptr<SolverStrategyParameters> );
    std::string type() const override { return "UASolver"; }

    enum class flags : std::uint8_t {
        none         = 0,
        redistribute = 1 << 0,
        boomer_cg    = 1 << 1,
        implicit_RAP = 1 << 2
    };

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<UASolver>( params );
    }

    void setup();

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> ) override;
    void getFromInput( std::shared_ptr<AMP::Database> );

    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;
    const std::vector<KCycleLevel> &levels() const { return d_levels; }
    const SolverStrategy &getCoarseSolver() const { return *d_coarse_solver; }

private:
    using redist_context = Utilities::GroupedRedistributionPlan;

    coarse_ops_type coarsen( std::shared_ptr<Operator::LinearOperator> A,
                             const PairwiseCoarsenSettings &,
                             std::shared_ptr<Operator::OperatorParameters> );
    static std::unique_ptr<SolverStrategy>
    create_relaxation( size_t lvl,
                       std::shared_ptr<AMP::Operator::LinearOperator> A,
                       std::shared_ptr<RelaxationParameters> params );
    void makeCoarseSolver();
    std::optional<redist_context> redistributeIfNeeded( std::shared_ptr<LinearAlgebra::Matrix> &A );
    size_t d_max_levels;
    int d_min_coarse_local;
    size_t d_min_coarse_global;
    size_t d_num_relax_pre;
    size_t d_num_relax_post;
    KappaKCycle::settings d_cycle_settings;
    Utilities::MemoryType d_mem_loc;
    std::shared_ptr<Aggregator> d_aggregator;
    PairwiseCoarsenSettings d_coarsen_settings;
    std::vector<KCycleLevel> d_levels;
    std::shared_ptr<RelaxationParameters> d_pre_relax_params;
    std::shared_ptr<RelaxationParameters> d_post_relax_params;
    std::shared_ptr<SolverStrategyParameters> d_coarse_solver_params;
    std::unique_ptr<SolverStrategy> d_coarse_solver;
    Utilities::Flags<flags> d_flags;
};

} // namespace AMP::Solver::AMG

#endif
