#ifndef included_AMP_DiffusionLinearFEOperator
#define included_AMP_DiffusionLinearFEOperator

#include "AMP/mesh/MeshElement.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "AMP/operators/libmesh/LinearFEOperator.h"

#include <memory>
#include <vector>


namespace AMP::Operator {


class DiffusionLinearFEOperator : public LinearFEOperator
{
public:
    explicit DiffusionLinearFEOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~DiffusionLinearFEOperator() {}

    void preAssembly( std::shared_ptr<const OperatorParameters> params ) override;

    void postAssembly() override;

    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    void postElementOperation() override;

    std::shared_ptr<DiffusionTransportModel> getTransportModel();

protected:
    explicit DiffusionLinearFEOperator(
        std::shared_ptr<const DiffusionLinearFEOperatorParameters> params, bool );

protected:
    std::set<std::string> d_constantVecs;
    std::map<std::string, std::shared_ptr<const AMP::LinearAlgebra::Vector>> d_inputVecs;

    std::vector<std::vector<double>> d_elementStiffnessMatrix;

    std::shared_ptr<DiffusionLinearElement> d_diffLinElem;

    std::shared_ptr<DiffusionTransportModel> d_transportModel;
};
} // namespace AMP::Operator

#endif
