#ifndef included_AMP_AMG_Cycle
#define included_AMP_AMG_Cycle

#include <memory>
#include <vector>

#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverStrategy.h"

namespace AMP::Solver::AMG {

struct Level {
    std::shared_ptr<AMP::Operator::LinearOperator> A;
    std::shared_ptr<AMP::Operator::Operator> R, P;
    std::unique_ptr<AMP::Solver::SolverStrategy> pre_relaxation, post_relaxation;
    std::shared_ptr<LinearAlgebra::Vector> x, b;
    mutable std::size_t nrelax = 0;
};

void kappa_kcycle( size_t lvl,
                   std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<Level> &levels,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp );

void kappa_kcycle( std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<Level> &levels,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp = false );
} // namespace AMP::Solver::AMG

#endif
