#ifndef included_AMP_CoupledFlowFrapconParameters
#define included_AMP_CoupledFlowFrapconParameters

#include "operators/subchannel/CoupledFlowFrapconOperator.h"
#include "solvers/SolverStrategy.h"
#include "solvers/SolverStrategyParameters.h"
#include "utils/Database.h"
#include "utils/shared_ptr.h"

namespace AMP {
namespace Solver {

class CoupledFlow1DSolverParameters : public SolverStrategyParameters
{
public:
    CoupledFlow1DSolverParameters() {}
    explicit CoupledFlow1DSolverParameters( const AMP::shared_ptr<AMP::Database> &db )
        : SolverStrategyParameters( db )
    {
    }
    virtual ~CoupledFlow1DSolverParameters() {}

    AMP::shared_ptr<AMP::Solver::SolverStrategy> d_flow1DSolver;
    using SolverStrategyParameters::d_pOperator;

protected:
private:
};
} // namespace Solver
} // namespace AMP

#endif
