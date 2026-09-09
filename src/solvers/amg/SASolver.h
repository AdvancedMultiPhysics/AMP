#ifndef included_AMP_SASolver_H_
#define included_AMP_SASolver_H_

#include "AMP/solvers/amg/AggregationSolver.h"

namespace AMP::Solver::AMG {

struct SASolver : AggregationSolver {
    explicit SASolver( std::shared_ptr<SolverStrategyParameters> params )
        : AggregationSolver( params, "SASolver" )
    {
        finishConstruction();
    }

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return createAggregationSolver<SASolver>( params );
    }
};

} // namespace AMP::Solver::AMG

#endif
