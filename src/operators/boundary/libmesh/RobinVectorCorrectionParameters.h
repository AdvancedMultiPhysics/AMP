
#ifndef included_AMP_RobinVectorCorrectionParameters
#define included_AMP_RobinVectorCorrectionParameters

#include "AMP/operators/OperatorParameters.h"

namespace AMP::Operator {

class RobinVectorCorrectionParameters : public OperatorParameters
{
public:
    explicit RobinVectorCorrectionParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db )
    {
    }

    std::string type() const { return "RobinVectorCorrectionParameters"; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;
};
} // namespace AMP::Operator

#endif
