
#ifndef included_AMP_ConstraintsEliminationSolver
#define included_AMP_ConstraintsEliminationSolver

#include <solvers/SolverStrategy.h>

namespace AMP {
  namespace Solver {

    typedef SolverStrategyParameters ConstraintsEliminationSolverParameters;

    class ConstraintsEliminationSolver : public SolverStrategy {
      public:
        ConstraintsEliminationSolver(boost::shared_ptr<ConstraintsEliminationSolverParameters> params);
        void solve(boost::shared_ptr<AMP::LinearAlgebra::Vector> f, boost::shared_ptr<AMP::LinearAlgebra::Vector> u);
    };

  }
}

#endif

