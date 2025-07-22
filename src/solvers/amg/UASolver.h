#ifndef included_AMP_UAAMGSolver
#define included_AMP_UAAMGSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/Cycle.h"
#include "AMP/solvers/amg/Relaxation.h"

namespace AMP::Solver::AMG {

struct UASolver : SolverStrategy {
    explicit UASolver( std::shared_ptr<SolverStrategyParameters> );
    std::string type() const override { return "UASolver"; }

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

private:
    static std::unique_ptr<SolverStrategy>
    create_relaxation( std::shared_ptr<AMP::Operator::LinearOperator> A,
                       std::shared_ptr<RelaxationParameters> params );
    void makeCoarseSolver();
    size_t d_max_levels;
    size_t d_min_coarse;
    size_t d_num_relax_pre;
    size_t d_num_relax_post;
    bool d_boomer_cg;
    size_t d_kappa;
    float d_kcycle_tol;
    PairwiseCoarsenSettings d_coarsen_settings;
    std::vector<Level> d_levels;
    std::shared_ptr<RelaxationParameters> d_pre_relax_params;
    std::shared_ptr<RelaxationParameters> d_post_relax_params;
    std::shared_ptr<SolverStrategyParameters> d_coarse_solver_params;
    std::unique_ptr<SolverStrategy> d_coarse_solver;
};

} // namespace AMP::Solver::AMG

#endif
