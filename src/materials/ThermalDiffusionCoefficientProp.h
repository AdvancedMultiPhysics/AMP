#ifndef included_AMP_ThermalDiffusionCoefficientProp
#define included_AMP_ThermalDiffusionCoefficientProp

#include "AMP/materials/Property.h"


namespace AMP::Materials {


/**
 * These context-dependent classes handle the thermal diffusion coefficient in a generic way.
 * They depend on previous definitions of Fick and Soret coefficients and optionally their
 * derivatives.
 * Parameter variables thermalDiffusionParams and numberThDiffParams must be defined.
 * Argument names list is thermDiffArgs, number is numberThermDiffArgs, ranges are thermDiffRanges.
 * Length of parameter array must be the same as the combined Fick and Soret parameter arrays.
 * The thermal diffusion coefficient calculated herein is the FickCoefficient * SoretCoefficient.
 * Simply include this file in the scope of your material namespace in your material header file.
 * Specializations may be handled within specific material header files in lieu of including this
 * one.
 */


class ThermalDiffusionCoefficientProp : public AMP::Materials::Property
{
public:
    ThermalDiffusionCoefficientProp( const std::string &name,
                                     std::shared_ptr<Property> FickProp,
                                     std::shared_ptr<Property> SoretProp,
                                     std::vector<double> params,
                                     std::vector<std::string> args,
                                     std::vector<std::array<double, 2>> range )
        : AMP::Materials::Property( name, Units(), "", params, args, range ),
          d_FickProp( FickProp ),
          d_SoretProp( SoretProp )
    {
    }

    virtual void set_parameters( std::vector<double> params )
    {
        std::vector<double> fickParams  = d_FickProp->get_parameters();
        std::vector<double> soretParams = d_SoretProp->get_parameters();
        AMP_ASSERT( params.size() == fickParams.size() + soretParams.size() );
        for ( size_t i = 0; i < fickParams.size(); i++ )
            fickParams[i] = params[i];
        for ( size_t i = 0; i < soretParams.size(); i++ )
            soretParams[i] = params[fickParams.size() + i];
        d_FickProp->set_parameters( std::move( fickParams ) );
        d_SoretProp->set_parameters( std::move( soretParams ) );
    }

    double eval( const std::vector<double> &args ) override
    {
        double fick  = d_FickProp->evalDirect( args );
        double soret = d_SoretProp->evalDirect( args );
        return fick * soret;
    }


private:
    std::shared_ptr<Property> d_FickProp;
    std::shared_ptr<Property> d_SoretProp;
};


class DxThermalDiffusionCoefficientProp : public AMP::Materials::Property
{
public:
    DxThermalDiffusionCoefficientProp( const std::string &name,
                                       std::shared_ptr<Property> FickProp,
                                       std::shared_ptr<Property> SoretProp,
                                       std::shared_ptr<Property> DxFickProp,
                                       std::shared_ptr<Property> DxSoretProp,
                                       std::vector<double> params,
                                       std::vector<std::string> args,
                                       std::vector<std::array<double, 2>> range )
        : AMP::Materials::Property( name, Units(), "", params, args, range ),
          d_FickProp( FickProp ),
          d_SoretProp( SoretProp ),
          d_DxFickProp( DxFickProp ),
          d_DxSoretProp( DxSoretProp )
    {
    }

    virtual void set_parameters( const double *params, const unsigned int nparams )
    {
        std::vector<double> fickParams  = d_FickProp->get_parameters();
        std::vector<double> soretParams = d_SoretProp->get_parameters();
        AMP_ASSERT( nparams == fickParams.size() + soretParams.size() );
        for ( size_t i = 0; i < fickParams.size(); i++ )
            fickParams[i] = params[i];
        for ( size_t i = 0; i < soretParams.size(); i++ )
            soretParams[i] = params[fickParams.size() + i];
        d_FickProp->set_parameters( std::move( fickParams ) );
        d_SoretProp->set_parameters( std::move( soretParams ) );
    }

    double eval( const std::vector<double> &args ) override
    {
        double fick    = d_FickProp->evalDirect( args );
        double soret   = d_SoretProp->evalDirect( args );
        double fickDx  = d_DxFickProp->evalDirect( args );
        double soretDx = d_DxSoretProp->evalDirect( args );
        return fickDx * soret + fick * soretDx;
    }

private:
    std::shared_ptr<Property> d_FickProp;
    std::shared_ptr<Property> d_SoretProp;
    std::shared_ptr<Property> d_DxFickProp;
    std::shared_ptr<Property> d_DxSoretProp;
};

class DTThermalDiffusionCoefficientProp : public AMP::Materials::Property
{
public:
    DTThermalDiffusionCoefficientProp( const std::string &name,
                                       std::shared_ptr<Property> FickProp,
                                       std::shared_ptr<Property> SoretProp,
                                       std::shared_ptr<Property> DTFickProp,
                                       std::shared_ptr<Property> DTSoretProp,
                                       std::vector<double> params,
                                       std::vector<std::string> args,
                                       std::vector<std::array<double, 2>> range )
        : AMP::Materials::Property( name, Units(), "", params, args, range ),
          d_FickProp( FickProp ),
          d_SoretProp( SoretProp ),
          d_DTFickProp( DTFickProp ),
          d_DTSoretProp( DTSoretProp )
    {
    }

    virtual void set_parameters( std::vector<double> params )
    {
        std::vector<double> fickParams  = d_FickProp->get_parameters();
        std::vector<double> soretParams = d_SoretProp->get_parameters();
        AMP_ASSERT( params.size() == fickParams.size() + soretParams.size() );
        for ( size_t i = 0; i < fickParams.size(); i++ )
            fickParams[i] = params[i];
        for ( size_t i = 0; i < soretParams.size(); i++ )
            soretParams[i] = params[fickParams.size() + i];
        d_FickProp->set_parameters( std::move( fickParams ) );
        d_SoretProp->set_parameters( std::move( soretParams ) );
    }

    double eval( const std::vector<double> &args ) override
    {
        double fick    = d_FickProp->evalDirect( args );
        double soret   = d_SoretProp->evalDirect( args );
        double fickDT  = d_DTFickProp->evalDirect( args );
        double soretDT = d_DTSoretProp->evalDirect( args );
        return fickDT * soret + fick * soretDT;
    }

private:
    std::shared_ptr<Property> d_FickProp;
    std::shared_ptr<Property> d_SoretProp;
    std::shared_ptr<Property> d_DTFickProp;
    std::shared_ptr<Property> d_DTSoretProp;
};


} // namespace AMP::Materials

#endif
