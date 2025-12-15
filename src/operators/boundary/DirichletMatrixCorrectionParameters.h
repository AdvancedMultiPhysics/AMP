#ifndef included_AMP_DirichletMatrixCorrectionParameters
#define included_AMP_DirichletMatrixCorrectionParameters

#include "AMP/operators/boundary/LinearBoundaryOperatorParameters.h"

namespace AMP::Operator {

class DirichletMatrixCorrectionParameters : public LinearBoundaryOperatorParameters
{
public:
    explicit DirichletMatrixCorrectionParameters( std::shared_ptr<AMP::Database> db )
        : LinearBoundaryOperatorParameters( db )
    {
    }

    virtual ~DirichletMatrixCorrectionParameters() {}

    // This must be a simple variable not a dual or multivariable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;
};
} // namespace AMP::Operator

#endif
