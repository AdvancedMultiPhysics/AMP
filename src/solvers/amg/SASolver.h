#ifndef included_AMP_SASolver_H_
#define included_AMP_SASolver_H_

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/solvers/amg/Cycle.h"
#include "AMP/solvers/amg/Relaxation.h"

#include <memory>

namespace AMP::Solver::AMG {

struct SASolver : SolverStrategy {
public:
    explicit SASolver( std::shared_ptr<SolverStrategyParameters> );

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<SASolver>( params );
    }

    std::string type() const override { return "SASolver"; }

    void registerOperator( std::shared_ptr<Operator::Operator> ) override;

    void getFromInput( std::shared_ptr<Database> );

    void apply( std::shared_ptr<const LinearAlgebra::Vector> f,
                std::shared_ptr<LinearAlgebra::Vector> u ) override;

protected:
    size_t d_max_levels;
    int d_min_coarse_local;
    size_t d_min_coarse_global;
    int d_num_smooth_prol;
    int d_num_relax_pre;
    int d_num_relax_post;
    int d_kappa;
    float d_kcycle_tol;
    float d_prol_trunc;
    Utilities::MemoryType d_mem_loc;


    std::shared_ptr<AMG::Aggregator> d_aggregator;
    std::vector<AMG::KCycleLevel> d_levels;
    std::shared_ptr<AMG::RelaxationParameters> d_pre_relax_params;
    std::shared_ptr<AMG::RelaxationParameters> d_post_relax_params;
    std::shared_ptr<SolverStrategyParameters> d_coarse_solver_params;
    std::unique_ptr<SolverStrategy> d_coarse_solver;

    void setup();

    void makeCoarseSolver();

    std::unique_ptr<SolverStrategy>
    createRelaxation( std::shared_ptr<Operator::Operator> A,
                      std::shared_ptr<AMG::RelaxationParameters> params );

    void smoothP_JacobiL1( std::shared_ptr<LinearAlgebra::Matrix> A,
                           std::shared_ptr<LinearAlgebra::Matrix> &P ) const;
};

} // namespace AMP::Solver::AMG

#endif
