#ifndef included_AMP_PelletStackMechanicsSolver
#define included_AMP_PelletStackMechanicsSolver

#include "AMP/operators/libmesh/PelletStackOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/libmesh/PelletStackMechanicsSolverParameters.h"

namespace AMP {
namespace Solver {


class PelletStackMechanicsSolver : public SolverStrategy
{
public:
    explicit PelletStackMechanicsSolver(
        AMP::shared_ptr<PelletStackMechanicsSolverParameters> params );

    virtual ~PelletStackMechanicsSolver() {}

    virtual void
    resetOperator( const AMP::shared_ptr<AMP::Operator::OperatorParameters> params ) override;

    virtual void solve( AMP::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                        AMP::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

protected:
    void solveSerial( AMP::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                      AMP::shared_ptr<AMP::LinearAlgebra::Vector> u );

    void solveScan( AMP::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                    AMP::shared_ptr<AMP::LinearAlgebra::Vector> u );

    AMP::shared_ptr<AMP::Operator::PelletStackOperator> d_pelletStackOp;
    AMP::shared_ptr<AMP::Solver::ColumnSolver> d_columnSolver;
    AMP::shared_ptr<AMP::LinearAlgebra::Vector> d_fbuffer1;
    AMP::shared_ptr<AMP::LinearAlgebra::Vector> d_fbuffer2;
};
} // namespace Solver
} // namespace AMP

#endif
