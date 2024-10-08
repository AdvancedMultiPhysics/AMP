
#ifndef included_AMP_NeumannVectorCorrectionParameters
#define included_AMP_NeumannVectorCorrectionParameters

#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/boundary/LinearBoundaryOperatorParameters.h"
#include "AMP/operators/boundary/libmesh/RobinPhysicsModel.h"

namespace AMP::Operator {

class NeumannVectorCorrectionParameters : public OperatorParameters
{
public:
    explicit NeumannVectorCorrectionParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db )
    {
    }

    virtual ~NeumannVectorCorrectionParameters() {}

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_variable;

    AMP::LinearAlgebra::Vector::shared_ptr d_variableFlux;

    std::shared_ptr<RobinPhysicsModel> d_robinPhysicsModel;
};
} // namespace AMP::Operator

#endif
