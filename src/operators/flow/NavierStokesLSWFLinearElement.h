
#ifndef included_AMP_NavierStokesLSWFLinearElement
#define included_AMP_NavierStokesLSWFLinearElement

#include <vector>

#include <memory>

/* AMP files */
#include "AMP/operators/flow/FlowElement.h"
#include "AMP/operators/flow/NavierStokesConstants.h"

namespace AMP::Operator {


class NavierStokesLSWFLinearElement : public FlowElement
{
public:
    explicit NavierStokesLSWFLinearElement(
        std::shared_ptr<const ElementOperationParameters> params )
        : FlowElement( params ), d_elementStiffnessMatrix( nullptr )
    {
        d_JxW     = &( d_fe->get_JxW() );
        d_dphi    = &( d_fe->get_dphi() );
        d_phi     = &( d_fe->get_phi() );
        d_xyz     = &( d_fe->get_xyz() );
        d_density = 0.;
        d_fmu     = 0.;
        d_Re      = 0.;
    }

    virtual ~NavierStokesLSWFLinearElement() {}

    void setElementStiffnessMatrix( std::vector<std::vector<double>> &elementStiffnessMatrix )
    {
        d_elementStiffnessMatrix = &( elementStiffnessMatrix );
    }

    void setElementVectors( const std::vector<double> &elementInputVectors )
    {
        d_elementInputVectors = elementInputVectors;
    }

    void apply() override;

protected:
    double d_density;

    double d_fmu;

    double d_Re;

    const std::vector<libMesh::Real> *d_JxW;

    const std::vector<std::vector<libMesh::RealGradient>> *d_dphi;

    const std::vector<std::vector<libMesh::Real>> *d_phi;

    const std::vector<libMesh::Point> *d_xyz;

    std::vector<double> d_elementInputVectors;

    std::vector<std::vector<double>> *d_elementStiffnessMatrix;

private:
};
} // namespace AMP::Operator

#endif
