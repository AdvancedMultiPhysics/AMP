#ifndef included_AMP_DiffusionNonlinearElement
#define included_AMP_DiffusionNonlinearElement

#include <vector>

#include "AMP/operators/diffusion/DiffusionConstants.h"
#include "AMP/operators/diffusion/DiffusionElement.h"
#include <memory>


namespace AMP::Operator {

class DiffusionNonlinearElement : public DiffusionElement
{
public:
    explicit DiffusionNonlinearElement( std::shared_ptr<const ElementOperationParameters> params )
        : DiffusionElement( params ),
          d_elementOutputVector( nullptr ),
          d_transportOutputVector( nullptr )
    {
        d_JxW = &( d_fe->get_JxW() );

        d_dphi = &( d_fe->get_dphi() );

        d_transportAtGauss = params->d_db->getWithDefault<bool>( "TransportAtGaussPoints", true );
    }

    virtual ~DiffusionNonlinearElement() {}

    void setElementInputVector( const std::vector<std::vector<double>> &elementInputVectors )
    {
        d_elementInputVectors = elementInputVectors;
    }

    void setElementVectors( const std::vector<std::vector<double>> &elementInputVectors,
                            std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors = elementInputVectors;
        d_elementOutputVector = &( elementOutputVector );
    }

    void setElementTransport( const std::vector<std::vector<double>> &elementInputVectors,
                              std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors   = elementInputVectors;
        d_transportOutputVector = &( elementOutputVector );
    }

    void apply() override;

    void initTransportModel();

    void setPrincipalVariable( const std::string &var ) { d_PrincipalVariable = var; }

    bool getTransportAtGauss() { return d_transportAtGauss; }

protected:
    std::vector<std::vector<double>> d_elementInputVectors;

    std::vector<double> *d_elementOutputVector;

    std::vector<double> *d_transportOutputVector;

    std::vector<std::vector<double>> d_elementOtherVectors;

    bool d_transportAtGauss;

    std::string d_PrincipalVariable;

private:
};
} // namespace AMP::Operator

#endif
