/*
 * DiffusionTransportModel.cc
 *
 *  Created on: Jul 8, 2010
 *      Author: gad
 */

#include "DiffusionTransportModel.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "DiffusionTransportTensorModel.h"
#include "ProfilerApp.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace AMP {
namespace Operator {

const std::vector<libMesh::Point> DiffusionTransportModel::d_DummyCoords =
    std::vector<libMesh::Point>( 0 );

DiffusionTransportModel::DiffusionTransportModel(
    const std::shared_ptr<DiffusionTransportModelParameters> &params )
    : ElementPhysicsModel( params ),
      d_defaults( Diffusion::NUMBER_VARIABLES ),
      d_MaterialParameters( 0 ),
      d_IsTensor( false )
{
    AMP_INSIST( ( params->d_db->keyExists( "Material" ) ),
                "Diffusion Key ''Material'' is missing!" );
    std::string matname = params->d_db->getString( "Material" );

    d_material = AMP::voodoo::Factory<AMP::Materials::Material>::instance().create( matname );

    AMP_INSIST( ( params->d_db->keyExists( "Property" ) ),
                "Diffusion Key ''Property'' is missing!" );
    std::string propname = params->d_db->getString( "Property" );
    d_property           = d_material->property( propname );

    // load and check defaults
    // initially set them to the minimum of the range plus a bit
    d_defaults.resize( d_property->get_number_arguments() );
    std::vector<std::vector<double>> ranges = d_property->get_arg_ranges();
    for ( size_t i = 0; i < d_defaults.size(); ++i ) {
        d_defaults[i] = ranges[i][0] * ( 1.0000001 );
    }
    if ( params->d_db->keyExists( "Defaults" ) ) {
        // check for correct names
        std::shared_ptr<Database> defaults_db = params->d_db->getDatabase( "Defaults" );
        std::vector<std::string> defaultkeys  = defaults_db->getAllKeys();
        AMP_INSIST( defaultkeys.size() == d_property->get_number_arguments(),
                    "Incorrect number of defaults supplied." );
        std::vector<std::string> argnames = d_property->get_arguments();
        for ( auto &defaultkey : defaultkeys ) {
            auto hit = std::find( argnames.begin(), argnames.end(), defaultkey );
            AMP_INSIST( hit != argnames.end(),
                        std::string( "Argument name " ) + defaultkey +
                            std::string( " is invalid" ) );
        }

        // load defaults into the material property, checking range validity
        for ( size_t i = 0; i < argnames.size(); ++i ) {
            d_defaults[i] = defaults_db->getScalar<double>( argnames[i] );
            AMP_INSIST( d_property->in_range( argnames[i], d_defaults[i] ),
                        std::string( "Default for argument " ) + argnames[i] +
                            std::string( " is out of range" ) );
        }
    }
    d_property->set_defaults( d_defaults );

    // process bilog scaling details
    d_UseBilogScaling = params->d_db->getWithDefault( "UseBilogScaling", false );
    if ( d_UseBilogScaling ) {
        AMP_INSIST( params->d_db->keyExists( "BilogVariable" ), "must specify BilogVariable" );
        d_BilogVariable = params->d_db->getWithDefault<std::string>( "BilogVariable", "NONE" );

        d_BilogRange = d_property->get_arg_range( d_BilogVariable );
        AMP_INSIST( d_BilogRange[1] > d_BilogRange[0],
                    "material argument upper bound <= lower bound" );

        d_BilogScaleCoefficient = params->d_db->getWithDefault( "BilogScaleCoefficient", true );
        d_BilogEpsilonRangeLimit =
            params->d_db->getWithDefault<double>( "BilogEpsilonRangeLimit", 1.0e-06 );
    }

    // for tensor properties, set or change dimension
    if ( d_property->isTensor() ) {
        std::shared_ptr<AMP::Materials::TensorProperty<double>> tensprop =
            std::dynamic_pointer_cast<AMP::Materials::TensorProperty<double>>( d_property );
        if ( tensprop->variable_dimensions() ) {
            if ( params->d_db->keyExists( "Dimensions" ) ) {
                std::vector<int> dims = params->d_db->getVector<int>( "Dimensions" );
                AMP_INSIST( dims.size() == 2, "only two dimensions allowed for tensor property" );
                std::vector<size_t> dims2( dims.size() );
                dims2[0] = dims[0];
                dims2[1] = dims[1];
                tensprop->set_dimensions( dims2 );
            }
        }
    }

    // set property parameters
    if ( params->d_db->keyExists( "Parameters" ) ) {
        d_MaterialParameters = params->d_db->getVector<double>( "Parameters" );
        size_t nparams       = d_MaterialParameters.size();
        size_t ndefparams    = d_property->get_parameters().size();
        if ( d_property->variable_number_parameters() ) {
            d_property->set_parameters_and_number( &d_MaterialParameters[0],
                                                   d_MaterialParameters.size() );
        } else {
            AMP_INSIST( nparams == ndefparams,
                        "attempted to set incorrect number of parameters for this property" );
            d_property->set_parameters( &d_MaterialParameters[0], d_MaterialParameters.size() );
        }
    }
}

std::shared_ptr<std::vector<double>> DiffusionTransportModel::bilogTransform(
    const std::vector<double> &U, const double a, const double b )
{
    std::shared_ptr<std::vector<double>> up( new std::vector<double>( U.size() ) );
    std::vector<double> &u = *up;

    for ( size_t i = 0; i < U.size(); i++ ) {
        double eU = std::exp( U[i] );
        u[i]      = ( a + b * eU ) / ( 1.0 + eU );
    }

    return up;
}

void DiffusionTransportModel::bilogScale( std::vector<double> &v, const double a, const double b )
{
    for ( auto &elem : v ) {
        double ev     = std::exp( elem );
        double temp   = ( 1.0 + ev );
        double factor = ( b - a ) * ev / ( temp * temp );
        elem *= factor;
    }
}

void DiffusionTransportModel::getTransport(
    std::vector<double> &result,
    std::map<std::string, std::shared_ptr<std::vector<double>>> &args,
    const std::vector<libMesh::Point> & )
{
    PROFILE_START( "getTransport", 7 );
    std::shared_ptr<std::vector<double>> scaledp;
    double lower, upper;

    if ( d_UseBilogScaling ) {
        // do the transform
        lower                       = d_BilogRange[0] + d_BilogEpsilonRangeLimit;
        upper                       = d_BilogRange[1] - d_BilogEpsilonRangeLimit;
        scaledp                     = bilogTransform( ( *args[d_BilogVariable] ), lower, upper );
        std::vector<double> &scaled = *scaledp;

        // save untransformed argument value
        for ( size_t i = 0; i < ( *args[d_BilogVariable] ).size(); i++ ) {
            double temp                   = ( *args[d_BilogVariable] )[i];
            ( *args[d_BilogVariable] )[i] = scaled[i];
            scaled[i]                     = temp;
        }
    }

    // evaluate material property
    d_property->evalv( result, args );

    if ( d_UseBilogScaling ) {
        // restore untransformed argument value
        std::vector<double> &scaled = *scaledp;
        lower                       = d_BilogRange[0] + d_BilogEpsilonRangeLimit;
        upper                       = d_BilogRange[1] - d_BilogEpsilonRangeLimit;
        for ( size_t i = 0; i < args[d_BilogVariable]->size(); i++ ) {
            ( *args[d_BilogVariable] )[i] = scaled[i];
        }

        if ( d_BilogScaleCoefficient )
            bilogScale( result, lower, upper );
    }
    PROFILE_STOP( "getTransport", 7 );
}
} // namespace Operator
} // namespace AMP