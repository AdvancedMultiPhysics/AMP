#ifndef included_AMP_ConstraintsEliminationSolver
#define included_AMP_ConstraintsEliminationSolver

#include "AMP/solvers/SolverStrategy.h"

namespace AMP::Solver {

typedef SolverStrategyParameters ConstraintsEliminationSolverParameters;

class ConstraintsEliminationSolver : public SolverStrategy
{
public:
    explicit ConstraintsEliminationSolver(
        std::shared_ptr<ConstraintsEliminationSolverParameters> params );

    std::string type() const override { return "ConstraintsEliminationSolver"; }


    virtual void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;
};
} // namespace AMP::Solver

#endif
