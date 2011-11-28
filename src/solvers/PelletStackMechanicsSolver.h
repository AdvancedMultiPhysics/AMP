
#ifndef included_AMP_PelletStackMechanicsSolver
#define included_AMP_PelletStackMechanicsSolver

#include "solvers/SolverStrategy.h"
#include "solvers/ColumnSolver.h"
#include "solvers/PelletStackMechanicsSolverParameters.h"
#include "operators/PelletStackOperator.h"

namespace AMP {
  namespace Solver {

    class PelletStackMechanicsSolver: public SolverStrategy {
      public:
        PelletStackMechanicsSolver(boost::shared_ptr<PelletStackMechanicsSolverParameters> params);

        ~PelletStackMechanicsSolver() { }

        void resetOperator(const boost::shared_ptr<AMP::Operator::OperatorParameters> params);

        void solve(boost::shared_ptr<AMP::LinearAlgebra::Vector> f, boost::shared_ptr<AMP::LinearAlgebra::Vector> u);

      protected:
        void solveSerial(boost::shared_ptr<AMP::LinearAlgebra::Vector> f, boost::shared_ptr<AMP::LinearAlgebra::Vector> u);

        void solveScan(boost::shared_ptr<AMP::LinearAlgebra::Vector> f, boost::shared_ptr<AMP::LinearAlgebra::Vector> u);

        boost::shared_ptr<AMP::Operator::PelletStackOperator> d_pelletStackOp;
        boost::shared_ptr<AMP::Solver::ColumnSolver> d_columnSolver;
        bool d_useSerial;
    };

  }
}

#endif



