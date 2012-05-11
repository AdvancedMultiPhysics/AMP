#ifndef included_AMP_DiffusionTransportTensorModel
#define included_AMP_DiffusionTransportTensorModel

#include "operators/diffusion/DiffusionTransportModel.h"
#include "materials/TensorProperty.h"
#include "utils/Utilities.h"

namespace AMP {
namespace Operator {

typedef ElementPhysicsModelParameters DiffusionTransportTensorModelParameters;

class DiffusionTransportTensorModel  : public DiffusionTransportModel
{
public:
    DiffusionTransportTensorModel(const boost::shared_ptr<DiffusionTransportTensorModelParameters> params);

    /**
     * \brief transport model returning a vector of tensors
     * \param result result[i] is a tensor of diffusion coefficients.
     * \param args args[j][i] is j-th material evalv argument
     * \param Coordinates coordinates on the mesh that may be needed by the model.
     */
    virtual void getTensorTransport(std::vector< std::vector< boost::shared_ptr<std::vector<double> > > >& result,
             std::map<std::string, boost::shared_ptr<std::vector<double> > >& args,
             const std::vector<Point>& Coordinates=d_DummyCoords);

private:
    boost::shared_ptr<AMP::Materials::TensorProperty<double> > d_tensorProperty; /// tensor property pointer
};

}
}

#endif
