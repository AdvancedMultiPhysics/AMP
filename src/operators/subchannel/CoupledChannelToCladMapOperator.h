#ifndef included_AMP_CoupledChannelToCladMapOperator
#define included_AMP_CoupledChannelToCladMapOperator

#include "AMP/operators/Operator.h"
#include "AMP/operators/subchannel/CoupledChannelToCladMapOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/vectors/Vector.h"

#include <vector>


namespace AMP::Operator {

class CoupledChannelToCladMapOperator : public Operator
{
public:
    explicit CoupledChannelToCladMapOperator(
        std::shared_ptr<const CoupledChannelToCladMapOperatorParameters> params );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_flowVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_thermalMapOperator->getOutputVariable();
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    virtual ~CoupledChannelToCladMapOperator() {}

    std::string type() const override { return "CoupledChannelToCladMapOperator"; }

protected:
private:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_flowVariable;

    AMP::LinearAlgebra::Vector::shared_ptr d_subchannelTemperature;
    AMP::LinearAlgebra::Vector::shared_ptr d_subchannelDensity;

    std::shared_ptr<AMP::Operator::Operator> d_thermalMapOperator;
    std::shared_ptr<AMP::Operator::Operator> d_densityMapOperator;

    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;
};
} // namespace AMP::Operator

#endif
