#ifndef DIFFUSIONCYLINDRICALTRANSPORTMODEL_H_
#define DIFFUSIONCYLINDRICALTRANSPORTMODEL_H_

#include "AMP/operators/diffusion/DiffusionTransportTensorModel.h"

#include <string>


namespace AMP::Operator {
typedef ElementPhysicsModelParameters DiffusionCylindricalTransportModelParameters;

class DiffusionCylindricalTransportModel : public DiffusionTransportTensorModel
{
public:
    explicit DiffusionCylindricalTransportModel(
        std::shared_ptr<const DiffusionTransportTensorModelParameters> params );

    /**
     * \brief transport model returning a vector of tensors for cylindrical symmetry
     * \param result result[i] is a tensor of diffusion coefficients.
     * \param args args[j][i] is j-th material evalv argument
     * \param Coordinates vector of points that define the spatial location
     */
    virtual void
    getTensorTransport( AMP::Array<std::shared_ptr<std::vector<double>>> &result,
                        std::map<std::string, std::shared_ptr<std::vector<double>>> &args,
                        const std::vector<libMesh::Point> &Coordinates = d_DummyCoords ) override;

private:
    std::string d_RadiusArgument;
};
} // namespace AMP::Operator

#endif /* DIFFUSIONCYLINDRICALTRANSPORTMODEL_H_ */
