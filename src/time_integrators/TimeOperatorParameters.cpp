#include "TimeOperatorParameters.h"

namespace AMP {
namespace TimeIntegrator {

TimeOperatorParameters::TimeOperatorParameters( std::shared_ptr<AMP::Database> db )
    : OperatorParameters( db )
{
    d_pRhsOperator.reset();
    d_pMassOperator.reset();
    d_pPreviousTimeSolution.reset();
    d_pSourceTerm.reset();
    d_pRhsOperatorParameters.reset();
    d_pMassOperatorParameters.reset();
}

TimeOperatorParameters::~TimeOperatorParameters() = default;
} // namespace TimeIntegrator
} // namespace AMP