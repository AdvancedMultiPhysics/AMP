#include "AMP/operators/libmesh/MassDensityModel.h"
#include "AMP/materials/CylindricallySymmetric.h"
#include "AMP/materials/ScalarProperty.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"

#include <algorithm>
#include <limits>


template<class TYPE>
static inline bool is( std::shared_ptr<AMP::Materials::Property> prop )
{
    return std::dynamic_pointer_cast<TYPE>( prop ).get() != nullptr;
}


namespace AMP::Operator {

MassDensityModel::MassDensityModel( std::shared_ptr<const MassDensityModelParameters> params )
    : ElementPhysicsModel( params )
{
    AMP_INSIST( ( params->d_db->keyExists( "Material" ) ), "Mass Key ''Material'' is missing!" );
    auto matname = params->d_db->getWithDefault<std::string>( "Material", "Independent" );
    d_material   = AMP::Materials::getMaterial( matname );

    if ( params->d_db->keyExists( "Property" ) ) {
        d_PropertyName = params->d_db->getString( "Property" );
        if ( params->d_db->keyExists( "Parameters" ) ) {
            d_Parameters = params->d_db->getVector<double>( "Parameters" );
        }
    } else {
        d_PropertyName = "unspecified";
    }

    AMP_INSIST( ( params->d_db->keyExists( "Equation" ) ), "Mass Key ''Equation'' is missing!" );
    std::string eqnname = params->d_db->getString( "Equation" );
    if ( eqnname == "Mechanics" )
        d_equation = MassEquation::Mechanics;
    // The mechanics mass matrix is multiplied by the density of the material.
    else if ( eqnname == "ThermalSource" )
        d_equation = MassEquation::Mechanics;
    // Because the specific power (Watts/gram) are defined from the NeutronicsSource,
    // The mass matrix for the right-hand-side of the thermal equation must include
    // the density (grams/cubic-centimeter) to the get units correct (Watts/cc)
    else if ( eqnname == "Thermal" )
        d_equation = MassEquation::Thermal;
    else if ( eqnname == "Chemical" )
        d_equation = MassEquation::Chemical;
    else if ( eqnname == "ManufacturedSource" )
        d_equation = MassEquation::Manufactured;
    // used for manufactured solution testing rhs
    else
        AMP_INSIST( false, "Mass Equation name is invalid" );

    d_UseBilogScaling = params->d_db->getWithDefault<bool>( "UseBilogScaling", false );
    if ( d_UseBilogScaling ) {
        AMP_INSIST( params->d_db->keyExists( "BilogVariable" ), "must specify BilogVariable" );
        d_BilogVariable = params->d_db->getWithDefault<std::string>( "BilogVariable", "NONE" );

        if ( d_equation == MassEquation::Thermal ) {
            d_BilogRange =
                d_material->property( "ThermalConductivity" )->get_arg_range( "temperature" );
        } else if ( d_equation == MassEquation::Chemical ) {
            d_BilogRange =
                d_material->property( "FickCoefficient" )->get_arg_range( "concentration" );
        }
        AMP_INSIST( d_BilogRange[1] > d_BilogRange[0],
                    "material argument upper bound == lower bound" );

        std::vector<std::string> names;
        if ( d_equation == MassEquation::Thermal ) {
            names = d_material->property( "ThermalConductivity" )->get_arguments();
        } else if ( d_equation == MassEquation::Chemical ) {
            names = d_material->property( "FickCoefficient" )->get_arguments();
        }
        d_BilogIndex = 999999;
        for ( size_t i = 0; i < names.size(); i++ ) {
            if ( names[i] == d_BilogVariable ) {
                d_BilogIndex = i;
                break;
            }
        }
        AMP_INSIST( d_BilogIndex < 999999,
                    "Did not find " + d_BilogVariable + " in list of material argument names" );

        if ( eqnname == "Thermal" ) {
            AMP_INSIST( d_BilogVariable == "temperature",
                        "thermal equation requires bilog scaling of temperature" );
        }
        if ( eqnname == "Chemical" ) {
            AMP_INSIST( d_BilogVariable == "concentration",
                        "chemical equation requires bilog scaling of concentration" );
        }
    }

    if ( d_equation == MassEquation::Manufactured ) {
        AMP_INSIST( params->d_db->keyExists( "ManufacturedSourceEquation" ),
                    "ManufacturedSourceEquation is missing" );
        std::string mfgeqn = params->d_db->getString( "ManufacturedSourceEquation" );

        if ( mfgeqn == "Thermal" )
            d_ManufacturedEquation = ManufacturedEquation::ThermalSrc;
        else if ( mfgeqn == "Fick" )
            d_ManufacturedEquation = ManufacturedEquation::FickSrc;
        else if ( mfgeqn == "Soret" )
            d_ManufacturedEquation = ManufacturedEquation::SoretSrc;
        else if ( mfgeqn == "FickSoret" )
            d_ManufacturedEquation = ManufacturedEquation::FickSoretSrc;
        else
            AMP_INSIST( false, "invalid value for ManufacturedSourceEquation" );

        AMP_INSIST( params->d_db->keyExists( "ManufacturedVariable" ),
                    "must specify ManufacturedVariable" );
        d_ManufacturedVariable = params->d_db->getString( "ManufacturedVariable" );
        AMP_INSIST( d_ManufacturedVariable == "Temperature" or
                        d_ManufacturedVariable == "Concentration",
                    "ManufacturedVariable must have the values Temperature or Concentration" );
        d_ManufacturedUseTemp = d_ManufacturedVariable == "Temperature";
        d_ManufacturedUseConc = d_ManufacturedVariable == "Concentration";

        std::shared_ptr<Database> mfg_db = params->d_db->getDatabase( "ManufacturedSolution" );
        d_ManufacturedSolution.reset( new ManufacturedSolution( mfg_db ) );
    }

    if ( d_equation == MassEquation::Mechanics ) {
        auto property = d_material->property( "Density" );

        // load and check defaults
        // initially set them to the minimum of the range plus a bit
        auto defaults = property->get_defaults();
        if ( params->d_db->keyExists( "Defaults" ) ) {
            // check for correct names
            auto defaults_db = params->d_db->getDatabase( "Defaults" );
            auto defaultkeys = defaults_db->getAllKeys();
            // if the defaults block is the right size, use it, else ignor it.
            if ( defaultkeys.size() == property->get_number_arguments() ) {
                std::vector<std::string> argnames = property->get_arguments();
                for ( auto &defaultkey : defaultkeys ) {
                    auto hit = std::find( argnames.begin(), argnames.end(), defaultkey );
                    AMP_INSIST( hit != argnames.end(),
                                std::string( "Argument name " ) + defaultkey +
                                    std::string( " is invalid" ) );
                }

                // load defaults into the material property, checking range validity
                for ( size_t i = 0; i < argnames.size(); ++i ) {
                    defaults[i] = defaults_db->getScalar<double>( argnames[i] );
                    AMP_INSIST( property->in_range( argnames[i], defaults[i] ),
                                std::string( "Default for argument " ) + argnames[i] +
                                    std::string( " is out of range" ) );
                }
            }
        }
        property->set_defaults( defaults );
    }
}

void MassDensityModel::getDensityMechanics( std::vector<double> &result,
                                            const std::vector<double> &T,
                                            const std::vector<double> &U,
                                            const std::vector<double> &B )
{
    AMP_ASSERT( ( T.size() == U.size() ) && ( U.size() == result.size() ) &&
                ( B.size() == U.size() ) );

    d_material->property( "Density" )
        ->evalv( result, {}, "temperature", T, "concentration", U, "burnup", B );
}

void MassDensityModel::getDensityThermal( std::vector<double> &result,
                                          const std::vector<double> &T,
                                          const std::vector<double> &U,
                                          const std::vector<double> &B )
{
    AMP_ASSERT( ( T.size() == U.size() ) && ( U.size() == result.size() ) &&
                ( B.size() == U.size() ) );
    unsigned int n = result.size();
    std::vector<double> density( n ), specificheat( n );
    auto densityProp = d_material->property( "Density" );
    auto heatCapProp = d_material->property( "HeatCapacityPressure" );
    densityProp->evalv( density, {}, "temperature", T, "concentration", U, "burnup", B );
    heatCapProp->evalv( specificheat, {}, "temperature", T, "concentration", U, "burnup", B );
    for ( unsigned int i = 0; i < n; i++ )
        result[i] = density[i] * specificheat[i];

    if ( d_UseBilogScaling ) {
        DiffusionTransportModel::bilogScale( result, d_BilogRange[0], d_BilogRange[1] );
    }
}

void MassDensityModel::getDensityChemical( std::vector<double> &result,
                                           const std::vector<double> &T,
                                           const std::vector<double> &U,
                                           const std::vector<double> &B )
{
    AMP_ASSERT( ( T.size() == U.size() ) && ( U.size() == result.size() ) &&
                ( B.size() == U.size() ) );

    for ( auto &elem : result )
        elem = 1.;

    if ( d_UseBilogScaling ) {
        DiffusionTransportModel::bilogScale( result, d_BilogRange[0], d_BilogRange[1] );
    }
}

void MassDensityModel::getDensityManufactured( std::vector<double> &result,
                                               const std::vector<double> &T,
                                               const std::vector<double> &U,
                                               const std::vector<double> &B,
                                               const std::vector<libMesh::Point> &xyz )
{

    AMP_ASSERT( ( T.size() == U.size() ) && ( U.size() == result.size() ) &&
                ( B.size() == U.size() ) );
    AMP_ASSERT( xyz.size() == result.size() );

    size_t neval = result.size();

    std::shared_ptr<AMP::Materials::Property> sourceProp;
    std::shared_ptr<AMP::Materials::Property> dSourceProp;
    bool needD = false;

    if ( d_PropertyName == "unspecified" ) {
        if ( d_ManufacturedEquation == ManufacturedEquation::ThermalSrc ) {
            sourceProp = d_material->property( "ThermalConductivity" );
            if ( d_ManufacturedUseConc ) {
                dSourceProp = d_material->property( "DxThermalConductivity" );
                needD       = true;
            }
            if ( d_ManufacturedUseTemp ) {
                dSourceProp = d_material->property( "DTThermalConductivity" );
                needD       = true;
            }
        } else if ( d_ManufacturedEquation == ManufacturedEquation::FickSrc ) {
            sourceProp = d_material->property( "FickCoefficient" );
            if ( d_ManufacturedUseConc ) {
                dSourceProp = d_material->property( "DxFickCoefficient" );
                needD       = true;
            }
            if ( d_ManufacturedUseTemp ) {
                dSourceProp = d_material->property( "DTFickCoefficient" );
                needD       = true;
            }
        } else if ( d_ManufacturedEquation == ManufacturedEquation::SoretSrc ) {
            sourceProp = d_material->property( "ThermalDiffusionCoefficient" );
            if ( d_ManufacturedUseConc ) {
                dSourceProp = d_material->property( "DxThermalDiffusionCoefficient" );
                needD       = true;
            }
            if ( d_ManufacturedUseTemp ) {
                dSourceProp = d_material->property( "DTThermalDiffusionCoefficient" );
                needD       = true;
            }
        } else if ( d_ManufacturedEquation == ManufacturedEquation::FickSoretSrc ) {
            AMP_INSIST( false, "cannot do Fick-Soret yet" );
        }
    } else if ( !d_Parameters.empty() ) {
        sourceProp    = d_material->property( d_PropertyName );
        auto name     = sourceProp->get_name();
        auto units    = sourceProp->get_units();
        auto args     = sourceProp->get_arguments();
        auto ranges   = sourceProp->get_arg_ranges();
        auto argUnits = sourceProp->get_arg_units();
        if ( is<AMP::Materials::CylindricallySymmetricTensor>( sourceProp ) ) {
            sourceProp = std::make_shared<AMP::Materials::CylindricallySymmetricTensor>(
                name, d_Parameters );
        } else if ( is<AMP::Materials::PolynomialProperty>( sourceProp ) ) {
            sourceProp = std::make_shared<AMP::Materials::PolynomialProperty>(
                name, "", units, d_Parameters, args, ranges, argUnits );
        } else if ( sourceProp->isTensor() ) {
            auto dims = sourceProp->size();
            AMP_INSIST( dims.size() == 2, "only two dimensions allowed for tensor property" );
            if ( d_Parameters.size() == 4 )
                dims = { 2, 2 };
            else if ( d_Parameters.size() == 9 )
                dims = { 3, 3 };
            AMP_ASSERT( d_Parameters.size() == dims[0] * dims[1] );
            sourceProp = std::make_shared<AMP::Materials::ScalarProperty>(
                name, Array<double>( dims, d_Parameters.data() ) );
        } else if ( d_Parameters.size() == 1 ) {
            sourceProp = std::make_shared<AMP::Materials::ScalarProperty>( name, d_Parameters[0] );
        } else {
            sourceProp = std::make_shared<AMP::Materials::PolynomialProperty>(
                d_PropertyName, "", units, d_Parameters, args, ranges, argUnits );
        }
    } else {
        sourceProp = d_material->property( d_PropertyName );
    }

    std::map<std::string, const std::vector<double> &> args = { { "temperature", T },
                                                                { "concentration", U },
                                                                { "burnup", B } };

    if ( sourceProp->isScalar() ) {
        std::vector<double> coeff( neval ), dCoeff( neval, 0. );
        sourceProp->evalv( coeff, {}, args );
        if ( needD ) {
            dSourceProp->evalv( dCoeff, {}, args );
        }

        for ( size_t i = 0; i < neval; i++ ) {
            auto soln = d_ManufacturedSolution->evaluate( xyz[i]( 0 ), xyz[i]( 1 ), xyz[i]( 2 ) );

            result[i] = coeff[i] * ( soln[4] + soln[7] + soln[9] ) +
                        dCoeff[i] * ( soln[1] * soln[1] + soln[2] * soln[2] + soln[3] * soln[3] );
        }
    }

    std::string propname = sourceProp->get_name();
    std::string solnname = d_ManufacturedSolution->get_name();
    bool isCylindrical   = propname.find( "CylindricallySymmetric" ) < propname.size() and
                         solnname.find( "Cylindrical" ) < solnname.size();

    if ( sourceProp->isTensor() && !isCylindrical ) {
        auto dimensions = sourceProp->size();
        AMP::Array<std::shared_ptr<std::vector<double>>> coeff( dimensions );
        for ( size_t i = 0; i < dimensions.length(); i++ )
            coeff( i ) = std::make_shared<std::vector<double>>( neval, 0 );
        sourceProp->evalv( coeff, {}, args );

        // 4 + xx xy xz yy yz zz =
        //      4  5  6  7  8  9 =
        // xx xy xz
        // yx yy yz
        // zx zy zz
        size_t xlate[3][3] = { { 4, 5, 6 }, { 5, 7, 8 }, { 6, 8, 9 } };

        for ( size_t k = 0; k < neval; k++ ) {
            auto soln = d_ManufacturedSolution->evaluate( xyz[k]( 0 ), xyz[k]( 1 ), xyz[k]( 2 ) );

            result[k] = 0.;
            for ( size_t i = 0; i < dimensions[0]; i++ )
                for ( size_t j = 0; j < dimensions[1]; j++ ) {
                    result[k] += ( *coeff( i, j ) )[k] * soln[xlate[i][j]];
                }
        }
    } else if ( sourceProp->isTensor() ) {
        // check dimensions, set up temporary storage
        auto dimensions = sourceProp->size();
        AMP_ASSERT( ( dimensions[0] == 3 ) && ( dimensions[1] == 3 ) );
        AMP::Array<std::shared_ptr<std::vector<double>>> coeff( dimensions );
        AMP::Array<std::shared_ptr<std::vector<double>>> coeffr( dimensions );
        AMP::Array<std::shared_ptr<std::vector<double>>> coeffz( dimensions );

        for ( size_t i = 0; i < dimensions.length(); i++ ) {
            coeff( i )  = std::make_shared<std::vector<double>>( neval, 0 );
            coeffr( i ) = std::make_shared<std::vector<double>>( neval, 0 );
            coeffz( i ) = std::make_shared<std::vector<double>>( neval, 0 );
        }

        // check that material property has expected argument names
        std::vector<std::string> argnames = sourceProp->get_arguments();
        AMP_ASSERT( std::find( argnames.begin(), argnames.end(), "radius" ) != argnames.end() );
        AMP_ASSERT( std::find( argnames.begin(), argnames.end(), "theta" ) != argnames.end() );
        AMP_ASSERT( std::find( argnames.begin(), argnames.end(), "zee" ) != argnames.end() );

        // get argument vectors
        std::vector<double> radius( neval );
        std::vector<double> theta( neval );
        std::vector<double> zee( neval );
        args.insert( std::pair<std::string, const std::vector<double> &>( "radius", radius ) );
        args.insert( std::pair<std::string, const std::vector<double> &>( "theta", theta ) );
        args.insert( std::pair<std::string, const std::vector<double> &>( "zee", zee ) );

        // fill in cylindrical coordinates
        double Pi = 3.1415926535898;
        for ( size_t k = 0; k < neval; k++ ) {
            double x = xyz[k]( 0 ), y = xyz[k]( 1 ), z = xyz[k]( 2 );
            double r = std::sqrt( x * x + y * y ), th = acos( x / r );
            if ( y < 0 )
                th = 2 * Pi - th;
            radius[k] = r;
            theta[k]  = th;
            zee[k]    = z;
        }

        // evaluate various derivatives of diffusion coefficient tensor
        sourceProp->setAuxiliaryData( "derivative", 0 );
        sourceProp->evalv( coeff, {}, args );

        sourceProp->setAuxiliaryData( "derivative", 1 );
        sourceProp->evalv( coeffr, {}, args );

        sourceProp->setAuxiliaryData( "derivative", 2 );
        sourceProp->evalv( coeffz, {}, args );

        // compute div (K . grad u) = div K . grad u + K : grad grad u
        for ( size_t k = 0; k < neval; k++ ) {
            double x = xyz[k]( 0 ), y = xyz[k]( 1 ), z = xyz[k]( 2 );
            double r = std::sqrt( x * x + y * y );
            if ( r == 0. ) {
                r = std::numeric_limits<double>::min();
            }
            double th = acos( x / r );
            if ( y < 0 ) {
                th = 2 * Pi - th;
            }
            // soln is the set of all derivatives wrto r, th, and z of order <= 2
            auto soln = d_ManufacturedSolution->evaluate( r, th, z );

            std::vector<double> Kr( 2 ), Kz( 2 );
            Kr[0] = ( *coeff( 0, 0 ) )[k] + ( *coeff( 1, 1 ) )[k];
            Kz[0] = ( *coeff( 2, 2 ) )[k];
            Kr[1] = ( *coeffr( 0, 0 ) )[k] + ( *coeffr( 1, 1 ) )[k];
            Kz[1] = ( *coeffz( 2, 2 ) )[k];

            result[k] = Kz[1] * soln[3] + Kz[0] * soln[9] + Kr[0] * soln[1] / r + Kr[1] * soln[1] +
                        Kr[0] * soln[4];
        }
    }
}
} // namespace AMP::Operator
