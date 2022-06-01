/*
 *  Created on: Mar 11, 2010
 *	  Author: bm, gad
 */

#include "AMP/materials/Material.h"
#include "AMP/materials/MaterialList.h"
#include "AMP/materials/Property.h"

#include <string>
#include <vector>

namespace AMP::Materials {

namespace UO2_MSRZC_09_NS {

//=================== Constants =====================================================

static const char *source = "Bogdan Mihaila, Marius Stan, Juan Ramirez, Alek Zubelewicz, Petrica "
                            "Cristea, Journal of Nuclear Materials 394 (2009) 182--189";

static std::initializer_list<double> TCparams = { 3.24e-2, 2.51e-4, 3.67, -4.73e-4, 5.95e-11 };
static std::initializer_list<double> DEparams = { 0.99672,    1.179e-5,  -2.429e-9,
                                                  1.219e-12,  0.99734,   9.082e-6,
                                                  -2.705e-10, 4.391e-13, 10960 };
static std::initializer_list<double> TEparams = { 1.1833e-5, -5.013e-9,  3.756e-12, -6.125e-17,
                                                  9.828e-6,  -6.390e-10, 1.33e-12,  -1.757e-17 };
static std::initializer_list<double> HCparams = { 52.174,      45.806,     87.951e-3,
                                                  -7.3461e-2,  -84.241e-6, 31.542e-9,
                                                  -2.6334e-12, -713910,    -295090 };

static std::initializer_list<double> YMparams = { 2.334e11, 1.095e-4, -1.34 };
static const double PRatio                    = 0.316;

static const std::initializer_list<std::string> arguments = { "temperature", "concentration" };

static const double TminVal = 299.9;
static const double TmaxVal = 1400;
static const double uminVal = 0.0;
static const double umaxVal = 0.2;

static std::initializer_list<std::array<double, 2>> ranges = { { TminVal, TmaxVal },
                                                               { uminVal, umaxVal } };

//=================== Classes =======================================================

class ThermalConductivityProp : public Property
{
public:
    ThermalConductivityProp()
        : Property( "UO2_MSRZC_09_ThermalConductivity", Units(), source, arguments, ranges ),
          d_params( TCparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class DensityProp : public Property
{
public:
    DensityProp()
        : Property( "UO2_MSRZC_09_Density", Units(), source, arguments, ranges ),
          d_params( DEparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class ThermalExpansionProp : public Property
{
public:
    ThermalExpansionProp()
        : Property( "UO2_MSRZC_09_ThermalExpansion", Units(), source, arguments, ranges ),
          d_params( TEparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class HeatCapacityPressureProp : public Property
{
public:
    HeatCapacityPressureProp()
        : Property( "UO2_MSRZC_09_HeatCapacityPressure", Units(), source, arguments, ranges ),
          d_params( HCparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class YoungsModulusProp : public Property
{
public:
    YoungsModulusProp()
        : Property( "UO2_MSRZC_09_YoungsModulus", Units(), source, arguments, ranges ),
          d_params( YMparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class PoissonRatioProp : public Property
{
public:
    PoissonRatioProp()
        : Property( "UO2_MSRZC_09_PoissonRatio", Units(), source, {}, {} ), d_params( { PRatio } )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class DxThermalConductivityProp : public Property
{
public:
    DxThermalConductivityProp()
        : Property( "UO2_MSRZC_09_DxThermalConductivity", Units(), source, arguments, ranges ),
          d_params( TCparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};

class DTThermalConductivityProp : public Property
{
public:
    DTThermalConductivityProp()
        : Property( "UO2_MSRZC_09_DTThermalConductivity", Units(), source, arguments, ranges ),
          d_params( TCparams )
    {
    }

    double eval( const std::vector<double> &args ) const override;

private:
    std::vector<double> d_params;
};


//=================== Functions =====================================================

double ThermalConductivityProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double x = u;
    if ( x < 0.001 )
        x = 0.001;

    double lambda0 = 1. / ( p[0] + p[1] * T );
    double Dtherm  = p[2] * exp( p[3] * T );
    double theta   = Dtherm * sqrt( 2 * x * lambda0 );
    double thcond  = p[4] * T * T * T + lambda0 * atan( theta ) / theta;
    return thcond;
}

double DensityProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double T2 = T * T;
    double expDe;
    if ( T > 923 ) {
        expDe = p[0] + p[1] * T + p[2] * T2 + p[3] * T2 * T;
    } else {
        expDe = p[4] + p[5] * T + p[6] * T2 + p[7] * T2 * T;
    }
    double dens = p[8] * exp( -3 * log( expDe ) );
    return dens;
}

double ThermalExpansionProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double T2 = T * T;
    double alpha;
    if ( T > 923 ) {
        alpha = p[0] + p[1] * T + p[2] * T2 + p[3] * T2 * T;
    } else {
        alpha = p[4] + p[5] * T + p[6] * T2 + p[7] * T2 * T;
    }
    return alpha;
}

double HeatCapacityPressureProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double x = u;
    if ( x < 0.001 )
        x = 0.001;

    double T2     = T * T;
    double heatcp = p[0] + p[1] * x + ( p[2] + p[3] * x ) * T +
                    ( 1 - x ) * ( p[4] * T2 + p[5] * T2 * T + p[6] * T2 * T2 ) +
                    ( p[7] + p[8] * x ) / T2;
    return heatcp;
}

double YoungsModulusProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double x = u;
    if ( x < 0.001 )
        x = 0.001;

    double youngs = p[0] * ( 1. + p[1] * T ) * exp( p[2] * x );
    return youngs;
}

double PoissonRatioProp::eval( const std::vector<double> & ) const { return PRatio; }

double DxThermalConductivityProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double x = u;
    if ( x < 0.001 )
        x = 0.001;

    double DxThCond =
        ( ( -0.35355339059327373 *
            atan( 1.4142135623730951 * exp( T * p[3] ) * sqrt( x / ( p[0] + T * p[1] ) ) * p[2] ) *
            sqrt( x / ( p[0] + T * p[1] ) ) ) /
              ( exp( T * p[3] ) * p[2] ) +
          ( 0.5 * x ) / ( p[0] + T * p[1] + 2.0 * exp( 2 * T * p[3] ) * x * p[2] * p[2] ) ) /
        x * x;

    return DxThCond;
}

double DTThermalConductivityProp::eval( const std::vector<double> &args ) const
{
    double T = args[0];
    double u = args[1];

    const auto &p = d_params;
    AMP_ASSERT( T > TminVal && T < TmaxVal );
    AMP_ASSERT( u >= uminVal && u <= umaxVal );

    double x = u;
    if ( x < 0.001 )
        x = 0.001;

    double DTThCond =
        ( -0.5 * p[1] ) / ( ( p[0] + T * p[1] ) *
                            ( p[0] + T * p[1] + 2.0 * exp( 2 * T * p[3] ) * x * p[2] * p[2] ) ) +
        ( 1. * p[3] ) / ( p[0] + T * p[1] + 2.0 * exp( 2 * T * p[3] ) * x * p[2] * p[2] ) +
        ( atan( 1.4142135623730951 * exp( T * p[3] ) * sqrt( x / ( p[0] + T * p[1] ) ) * p[2] ) *
          ( -0.7071067811865475 * p[0] * p[3] +
            p[1] * ( -0.35355339059327373 - 0.7071067811865475 * T * p[3] ) ) ) /
            ( exp( T * p[3] ) * sqrt( x / ( p[0] + T * p[1] ) ) * ( p[0] + T * p[1] ) *
              ( p[0] + T * p[1] ) * p[2] ) +
        3 * T * T * p[4];

    return DTThCond;
}
} // namespace UO2_MSRZC_09_NS

//=================== Materials =====================================================
// clang-format off
UO2_MSRZC_09::UO2_MSRZC_09()
{
    d_propertyMap["ThermalConductivity"]   = std::make_shared<UO2_MSRZC_09_NS::ThermalConductivityProp>();
    d_propertyMap["Density"]               = std::make_shared<UO2_MSRZC_09_NS::DensityProp>();
    d_propertyMap["HeatCapacityPressure"]  = std::make_shared<UO2_MSRZC_09_NS::HeatCapacityPressureProp>();
    d_propertyMap["ThermalExpansion"]      = std::make_shared<UO2_MSRZC_09_NS::ThermalExpansionProp>();
    d_propertyMap["YoungsModulus"]         = std::make_shared<UO2_MSRZC_09_NS::YoungsModulusProp>();
    d_propertyMap["PoissonRatio"]          = std::make_shared<UO2_MSRZC_09_NS::PoissonRatioProp>();
    d_propertyMap["DxThermalConductivity"] = std::make_shared<UO2_MSRZC_09_NS::DxThermalConductivityProp>();
    d_propertyMap["DTThermalConductivity"] = std::make_shared<UO2_MSRZC_09_NS::DTThermalConductivityProp>();
}
// clang-format on

} // namespace AMP::Materials
