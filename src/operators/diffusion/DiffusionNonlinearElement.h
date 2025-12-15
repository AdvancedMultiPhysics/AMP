#ifndef included_AMP_DiffusionNonlinearElement
#define included_AMP_DiffusionNonlinearElement

#include "AMP/operators/diffusion/DiffusionElement.h"

#include <memory>
#include <vector>


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

    void setElementInputVector( std::map<std::string, std::vector<double>> elementInputVectors )
    {
        d_elementInputVectors = std::move( elementInputVectors );
    }

    void setElementVectors( std::map<std::string, std::vector<double>> elementInputVectors,
                            std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors = std::move( elementInputVectors );
        d_elementOutputVector = &elementOutputVector;
    }

    void setElementTransport( std::map<std::string, std::vector<double>> elementInputVectors,
                              std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors   = std::move( elementInputVectors );
        d_transportOutputVector = &elementOutputVector;
    }

    void apply() override;

    void initTransportModel();

    void setPrincipalVariable( const std::string &var ) { d_PrincipalVariable = var; }

    bool getTransportAtGauss() { return d_transportAtGauss; }

protected:
    void applyScalar();
    void applyTensor();

protected:
    std::map<std::string, std::vector<double>> d_elementInputVectors;

    std::vector<double> *d_elementOutputVector;

    std::vector<double> *d_transportOutputVector;

    std::vector<std::vector<double>> d_elementOtherVectors;

    bool d_transportAtGauss;

    std::string d_PrincipalVariable;

private:
};
} // namespace AMP::Operator

#endif
