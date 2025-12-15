#ifndef included_AMP_LinearBoundaryOperatorParameters
#define included_AMP_LinearBoundaryOperatorParameters

#include "AMP/matrices/Matrix.h"
#include "AMP/operators/boundary/BoundaryOperatorParameters.h"

namespace AMP::Operator {

class LinearBoundaryOperatorParameters : public BoundaryOperatorParameters
{
public:
    explicit LinearBoundaryOperatorParameters( std::shared_ptr<AMP::Database> db )
        : BoundaryOperatorParameters( db )
    {
    }

    virtual ~LinearBoundaryOperatorParameters() {}

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_inputMatrix;
};
} // namespace AMP::Operator

#endif
