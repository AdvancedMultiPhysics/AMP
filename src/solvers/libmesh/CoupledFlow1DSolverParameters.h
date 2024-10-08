#ifndef included_AMP_CoupledFlowFrapconParameters
#define included_AMP_CoupledFlowFrapconParameters

#include "AMP/operators/subchannel/CoupledFlowFrapconOperator.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/Database.h"
#include <memory>

namespace AMP::Solver {

class CoupledFlow1DSolverParameters : public SolverStrategyParameters
{
public:
    CoupledFlow1DSolverParameters() {}
    explicit CoupledFlow1DSolverParameters( std::shared_ptr<AMP::Database> db )
        : SolverStrategyParameters( db )
    {
    }
    virtual ~CoupledFlow1DSolverParameters() {}

    std::shared_ptr<AMP::Solver::SolverStrategy> d_flow1DSolver;
    using SolverStrategyParameters::d_pOperator;

protected:
private:
};
} // namespace AMP::Solver

#endif
