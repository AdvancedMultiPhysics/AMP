#ifndef included_AMP_RobinMatrixCorrectionParameters
#define included_AMP_RobinMatrixCorrectionParameters

#include "AMP/operators/boundary/LinearBoundaryOperatorParameters.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinPhysicsModel.h"

namespace AMP::Operator {

class RobinMatrixCorrectionParameters : public LinearBoundaryOperatorParameters
{
public:
    explicit RobinMatrixCorrectionParameters( std::shared_ptr<AMP::Database> db )
        : LinearBoundaryOperatorParameters( db )
    {
    }

    virtual ~RobinMatrixCorrectionParameters() {}

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;

    AMP::LinearAlgebra::Vector::shared_ptr d_variableFlux;

    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> d_elementInputVec;

    std::shared_ptr<RobinPhysicsModel> d_robinPhysicsModel;

    std::shared_ptr<AMP::Discretization::DOFManager> d_DofMap;
};
} // namespace AMP::Operator

#endif
