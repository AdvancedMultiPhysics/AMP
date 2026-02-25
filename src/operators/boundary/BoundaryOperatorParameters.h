#ifndef included_AMP_BoundaryOperatorParameters
#define included_AMP_BoundaryOperatorParameters

#include "AMP/matrices/Matrix.h"
#include "AMP/operators/OperatorParameters.h"

namespace AMP::Operator {

class BoundaryOperatorParameters : public OperatorParameters
{
public:
    explicit BoundaryOperatorParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db )
    {
    }

    virtual ~BoundaryOperatorParameters() {}

    std::shared_ptr<AMP::Operator::Operator> d_volumeOperator; // Optional Volume operator
};

} // namespace AMP::Operator

#endif
