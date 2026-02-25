#ifndef included_AMP_RobinVectorCorrectionParameters
#define included_AMP_RobinVectorCorrectionParameters

#include "AMP/operators/boundary/BoundaryOperatorParameters.h"

namespace AMP::Operator {

class RobinVectorCorrectionParameters : public BoundaryOperatorParameters
{
public:
    explicit RobinVectorCorrectionParameters( std::shared_ptr<AMP::Database> db )
        : BoundaryOperatorParameters( db )
    {
    }

    std::string type() const { return "RobinVectorCorrectionParameters"; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;
};
} // namespace AMP::Operator

#endif
