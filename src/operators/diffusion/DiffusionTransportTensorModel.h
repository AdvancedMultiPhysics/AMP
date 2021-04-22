#ifndef included_AMP_DiffusionTransportTensorModel
#define included_AMP_DiffusionTransportTensorModel

#include "AMP/materials/TensorProperty.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/utils/Utilities.h"

namespace AMP {
namespace Operator {

typedef ElementPhysicsModelParameters DiffusionTransportTensorModelParameters;

class DiffusionTransportTensorModel : public DiffusionTransportModel
{
public:
    explicit DiffusionTransportTensorModel(
        const std::shared_ptr<DiffusionTransportTensorModelParameters> params );

    /**
     * \brief transport model returning a vector of tensors
     * \param result result[i] is a tensor of diffusion coefficients.
     * \param args args[j][i] is j-th material evalv argument
     * \param Coordinates coordinates on the mesh that may be needed by the model.
     */
    virtual void
    getTensorTransport( std::vector<std::vector<std::shared_ptr<std::vector<double>>>> &result,
                        std::map<std::string, std::shared_ptr<std::vector<double>>> &args,
                        const std::vector<libMesh::Point> &Coordinates = d_DummyCoords );

private:
    std::shared_ptr<AMP::Materials::TensorProperty> d_tensorProperty; /// tensor property pointer
};
} // namespace Operator
} // namespace AMP

#endif
