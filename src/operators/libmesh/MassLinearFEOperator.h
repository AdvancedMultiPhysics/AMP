
#ifndef included_AMP_MassLinearFEOperator
#define included_AMP_MassLinearFEOperator

// AMP files
#include "AMP/operators/libmesh/LinearFEOperator.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/libmesh/MassLinearFEOperatorParameters.h"

#include <memory>
#include <vector>

namespace AMP::Operator {

class MassLinearFEOperator : public LinearFEOperator
{
public:
    explicit MassLinearFEOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~MassLinearFEOperator() {}

    std::string type() const override { return "MassLinearFEOperator"; }

    void preAssembly( std::shared_ptr<const AMP::Operator::OperatorParameters> ) override;

    void postAssembly() override;

    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    void postElementOperation() override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    std::shared_ptr<MassDensityModel> getDensityModel() { return d_densityModel; };

protected:
    explicit MassLinearFEOperator( std::shared_ptr<const MassLinearFEOperatorParameters>, bool );

protected:
    bool d_useConstantTemperature;

    bool d_useConstantConcentration;

    bool d_useConstantBurnup;

    double d_constantTemperatureValue;

    double d_constantConcentrationValue;

    double d_constantBurnupValue;

    AMP::LinearAlgebra::Vector::shared_ptr d_temperature;

    AMP::LinearAlgebra::Vector::shared_ptr d_concentration;

    AMP::LinearAlgebra::Vector::shared_ptr d_burnup;

    std::vector<std::vector<double>> d_elementMassMatrix;

    std::shared_ptr<MassLinearElement> d_massLinElem;

    std::shared_ptr<MassDensityModel> d_densityModel;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

private:
};
} // namespace AMP::Operator

#endif
