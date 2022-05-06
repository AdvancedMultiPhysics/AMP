//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   materials/FixedClad.cc
 * \author Aaron Phillippe
 * \brief  Implementation file for constant cladding properties
 */
//---------------------------------------------------------------------------//

#include "FixedClad.h"

#include "Material.h"
#include "Property.h"
#include "TensorProperty.h"
#include "VectorProperty.h"

#include <string>

namespace AMP::Materials {

namespace FixedClad_NS {

//  =================== Constants =====================================================

const char *name_base = "FixedClad";
const char *source    = "average values from matpro; as defined by Phillippe";

static const double thermalval = 16.05;
static const double fickval    = 1.;
static const double soretval   = 1.;

static const double densval   = 1.;
static const double alphaval  = 1.;
static const double heatcpval = 1.;

static const double youngsval = 1.;
static const double pratioval = 0.290;

//  =================== Classes =======================================================

class ThermalConductivityProp : public Property
{
public:
    ThermalConductivityProp()
        : Property( "FixedClad_ThermalConductivity", // Name string
                    source,                          // Reference source
                    { thermalval }                   // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class FickCoefficientProp : public Property
{
public:
    FickCoefficientProp()
        : Property( "FixedClad_FickCoefficient", // Name string
                    source,                      // Reference source
                    { fickval }                  // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class SoretCoefficientProp : public Property
{
public:
    SoretCoefficientProp()
        : Property( "FixedClad_SoretCoefficient", // Name string
                    source,                       // Reference source
                    { fickval }                   // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DensityProp : public Property
{
public:
    DensityProp()
        : Property( "FixedClad_Density", // Name string
                    source,              // Reference source
                    { densval }          // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class ThermalExpansionProp : public Property
{
public:
    ThermalExpansionProp()
        : Property( "FixedClad_ThermalExpansion", // Name string
                    source,                       // Reference source
                    { alphaval }                  // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class HeatCapacityPressureProp : public Property
{
public:
    HeatCapacityPressureProp()
        : Property( "FixedClad_HeatCapacityPressure", // Name string
                    source,                           // Reference source
                    { heatcpval }                     // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class YoungsModulusProp : public Property
{
public:
    YoungsModulusProp()
        : Property( "FixedClad_YoungsModulus", // Name string
                    source,                    // Reference source
                    { youngsval }              // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class PoissonRatioProp : public Property
{
public:
    PoissonRatioProp()
        : Property( "FixedClad_PoissonRatio", // Name string
                    source,                   // Reference source
                    { pratioval }             // Property parameters
          )
    {
    } // Number of arguments

    double eval( const std::vector<double> &args ) override;
};

class DTThermalConductivityProp : public Property
{
public:
    DTThermalConductivityProp()
        : Property( "FixedClad_DTThermalConductivity", // Name string
                    source,                            // Reference source
                    { thermalval }                     // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DTFickCoefficientProp : public Property
{
public:
    DTFickCoefficientProp()
        : Property( "FixedClad_DTFickCoefficient", // Name string
                    source,                        // Reference source
                    { fickval },                   // Property parameters
                    {}                             // Names of arguments
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DTSoretCoefficientProp : public Property
{
public:
    DTSoretCoefficientProp()
        : Property( "FixedClad_DTSoretCoefficient", // Name string
                    source,                         // Reference source
                    { soretval }                    // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DxThermalConductivityProp : public Property
{
public:
    DxThermalConductivityProp()
        : Property( "FixedClad_DxThermalConductivity", // Name string
                    source,                            // Reference source
                    { thermalval }                     // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DxFickCoefficientProp : public Property
{
public:
    DxFickCoefficientProp()
        : Property( "FixedClad_DxFickCoefficient", // Name string
                    source,                        // Reference source
                    { fickval },                   // Property parameters
                    {}                             // Names of arguments
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class DxSoretCoefficientProp : public Property
{
public:
    DxSoretCoefficientProp()
        : Property( "FixedClad_DxSoretCoefficient", // Name string
                    source,                         // Reference source
                    { soretval }                    // Property parameters
          )
    {
    }

    double eval( const std::vector<double> &args ) override;
};

class VectorFickCoefficientProp : public VectorProperty
{
public:
    explicit VectorFickCoefficientProp( const size_t dim = 1 )
        : VectorProperty( "FixedClad_VectorFickCoefficient", // Name string
                          source,                            // Reference source
                          { 1. },                            // Property parameters
                          {},                                // Names of arguments
                          {},                                // ranges
                          dim )                              // dimension
    {
        AMP_INSIST( d_params.size() == dim, "dimensions and number of parameters don't match" );
        d_variableNumberParameters = true;
        d_variableDimension        = true;
    }

    // NOTE: must change dimension first before changing number of parameters
    void set_parameters_and_number( std::vector<double> params ) override
    {
        AMP_INSIST( d_dimension == params.size(),
                    "number of new parameters must be same as dimension" );
        Property::set_parameters_and_number( params );
    }

    std::vector<double> evalVector( const std::vector<double> &args ) override;
};

class TensorFickCoefficientProp : public TensorProperty
{
public:
    explicit TensorFickCoefficientProp( const std::vector<size_t> &dims = std::vector<size_t>( 2,
                                                                                               1 ) )
        : TensorProperty( "FixedClad_TensorFickCoefficient", // Name string
                          source,                            // Reference source
                          { 1. },                            // Property parameters
                          {},                                // Names of arguments
                          {},                                // ranges
                          dims )                             // dimensions
    {
        AMP_INSIST( d_params.size() == dims[0] * dims[1],
                    "dimensions and number of parameters don't match" );
        d_variableNumberParameters = true;
        d_variableDimensions       = true;
    }

    // NOTE: must change dimension first before changing number of parameters
    void set_parameters_and_number( std::vector<double> params ) override
    {
        AMP_INSIST( d_dimensions[0] * d_dimensions[1] == params.size(),
                    "number of new parameters must be product of dimensions" );
        Property::set_parameters_and_number( params );
    }

    std::vector<std::vector<double>> evalTensor( const std::vector<double> &args ) override;
};

static std::initializer_list<double> thermalDiffusionParams     = { 1., 1. };
static const std::vector<std::string> thermDiffArgs             = {};
static const std::vector<std::array<double, 2>> thermDiffRanges = {};

#define THERMAL_DIFFUSION_DERIVATIVE
#include "ThermalDiffusionCoefficientProp.h"
#undef THERMAL_DIFFUSION_DERIVATIVE

//  =================== Functions =====================================================

inline double ThermalConductivityProp::eval( const std::vector<double> & )
{
    return get_parameters()[0];
}

inline double FickCoefficientProp::eval( const std::vector<double> & )
{
    return get_parameters()[0];
}

inline double SoretCoefficientProp::eval( const std::vector<double> & )
{
    return get_parameters()[0];
}

inline double DTThermalConductivityProp::eval( const std::vector<double> & ) { return 0.; }

inline double DxThermalConductivityProp::eval( const std::vector<double> & ) { return 0.; }

inline double DTFickCoefficientProp::eval( const std::vector<double> & ) { return 0.; }

inline double DxFickCoefficientProp::eval( const std::vector<double> & ) { return 0.; }

inline double DTSoretCoefficientProp::eval( const std::vector<double> & ) { return 0.; }

inline double DxSoretCoefficientProp::eval( const std::vector<double> & ) { return 0.; }

inline double DensityProp::eval( const std::vector<double> & ) { return get_parameters()[0]; }

inline double ThermalExpansionProp::eval( const std::vector<double> & )
{
    return get_parameters()[0];
}

inline double HeatCapacityPressureProp::eval( const std::vector<double> & )
{
    return get_parameters()[0];
}

inline double YoungsModulusProp::eval( const std::vector<double> & ) { return get_parameters()[0]; }

inline double PoissonRatioProp::eval( const std::vector<double> & ) { return get_parameters()[0]; }

std::vector<double> VectorFickCoefficientProp::evalVector( const std::vector<double> & )
{
    std::vector<double> result( d_dimension );
    for ( size_t i = 0; i < d_dimension; i++ )
        result[i] = d_params[i];
    return result;
}

std::vector<std::vector<double>>
TensorFickCoefficientProp::evalTensor( const std::vector<double> & )
{
    std::vector<std::vector<double>> result( d_dimensions[0],
                                             std::vector<double>( d_dimensions[1] ) );
    for ( size_t i = 0; i < d_dimensions[0]; i++ )
        for ( size_t j = 0; j < d_dimensions[1]; j++ )
            result[i][j] = d_params[i * d_dimensions[1] + j];
    return result;
}
} // namespace FixedClad_NS

//  =================== Materials =====================================================
// clang-format off
FixedClad::FixedClad()
{
    d_propertyMap["ThermalConductivity"]         = std::make_shared<FixedClad_NS::ThermalConductivityProp>();
    d_propertyMap["FickCoefficient"]             = std::make_shared<FixedClad_NS::FickCoefficientProp>();
    d_propertyMap["SoretCoefficient"]            = std::make_shared<FixedClad_NS::SoretCoefficientProp>();
    d_propertyMap["DTThermalConductivity"]       = std::make_shared<FixedClad_NS::DTThermalConductivityProp>();
    d_propertyMap["DTFickCoefficient"]           = std::make_shared<FixedClad_NS::DTFickCoefficientProp>();
    d_propertyMap["DTSoretCoefficient"]          = std::make_shared<FixedClad_NS::DTSoretCoefficientProp>();
    d_propertyMap["DxThermalConductivity"]       = std::make_shared<FixedClad_NS::DxThermalConductivityProp>();
    d_propertyMap["DxFickCoefficient"]           = std::make_shared<FixedClad_NS::DxFickCoefficientProp>();
    d_propertyMap["DxSoretCoefficient"]          = std::make_shared<FixedClad_NS::DxSoretCoefficientProp>();
    d_propertyMap["Density"]                     = std::make_shared<FixedClad_NS::DensityProp>();
    d_propertyMap["HeatCapacityPressure"]        = std::make_shared<FixedClad_NS::HeatCapacityPressureProp>();
    d_propertyMap["ThermalExpansion"]            = std::make_shared<FixedClad_NS::ThermalExpansionProp>();
    d_propertyMap["YoungsModulus"]               = std::make_shared<FixedClad_NS::YoungsModulusProp>();
    d_propertyMap["PoissonRatio"]                = std::make_shared<FixedClad_NS::PoissonRatioProp>();
    d_propertyMap["ThermalDiffusionCoefficient"] = std::make_shared<FixedClad_NS::ThermalDiffusionCoefficientProp>();
    d_propertyMap["VectorFickCoefficient"]       = std::make_shared<FixedClad_NS::VectorFickCoefficientProp>();
    d_propertyMap["TensorFickCoefficient"]       = std::make_shared<FixedClad_NS::TensorFickCoefficientProp>();
}
// clang-format on

} // namespace AMP::Materials
