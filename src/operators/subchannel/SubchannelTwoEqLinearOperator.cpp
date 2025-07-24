#include "AMP/operators/subchannel/SubchannelTwoEqLinearOperator.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"

#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "ProfilerApp.h"

#include <string>


namespace AMP::Operator {


// Constructor
SubchannelTwoEqLinearOperator::SubchannelTwoEqLinearOperator(
    std::shared_ptr<const OperatorParameters> inparams )
    : LinearOperator( inparams ),
      d_Pout( 0 ),
      d_Tin( 0 ),
      d_mass( 0 ),
      d_gamma( 0 ),
      d_theta( 0 ),
      d_Q( 0 ),
      d_reynolds( 0 ),
      d_prandtl( 0 ),
      d_friction( 0 ),
      d_roughness( 0 ),
      d_NGrid( 0 ),
      d_machinePrecision( 1.0e-15 ),
      d_numSubchannels( 0 )
{
    auto params = Subchannel::convert( inparams );
    AMP_INSIST( params->d_db->keyExists( "InputVariable" ), "Key 'InputVariable' does not exist" );
    std::string inpVar = params->d_db->getString( "InputVariable" );
    d_inputVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );

    AMP_INSIST( params->d_db->keyExists( "OutputVariable" ),
                "Key 'OutputVariable' does not exist" );
    std::string outVar = params->d_db->getString( "OutputVariable" );
    d_outputVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );

    d_params      = params;
    d_initialized = false;

    reset( params );
}


// reset
void SubchannelTwoEqLinearOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    if ( params ) {

        PROFILE( "reset" );
        d_initialized = true;
        auto myparams = std::dynamic_pointer_cast<const SubchannelOperatorParameters>( params );

        AMP_INSIST( ( ( myparams.get() ) != nullptr ), "NULL parameters" );
        AMP_INSIST( ( ( ( myparams->d_db ).get() ) != nullptr ), "NULL database" );

        d_params = myparams;

        // We require that subchannel is on an AMP structured

        // Get the subchannel mesh coordinates
        AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( d_Mesh, d_x, d_y, d_z );
        d_numSubchannels = ( d_x.size() - 1 ) * ( d_y.size() - 1 );

        // Get the properties from the database
        d_Pout  = getDoubleParameter( myparams, "Exit_Pressure", 15.5132e6 );
        d_Tin   = getDoubleParameter( myparams, "Inlet_Temperature", 569.26 );
        d_mass  = getDoubleParameter( myparams, "Inlet_Mass_Flow_Rate", 0.3522 * d_numSubchannels );
        d_gamma = getDoubleParameter( myparams, "Fission_Heating_Coefficient", 0.0 );
        d_theta = getDoubleParameter( myparams, "Channel_Angle", 0.0 );
        d_reynolds      = getDoubleParameter( myparams, "Reynolds", 0.0 );
        d_prandtl       = getDoubleParameter( myparams, "Prandtl", 0.0 );
        d_friction      = getDoubleParameter( myparams, "Friction_Factor", 0.001 );
        d_source        = getStringParameter( myparams, "Heat_Source_Type", "totalHeatGeneration" );
        d_frictionModel = getStringParameter( myparams, "Friction_Model", "Constant" );
        d_NGrid         = getIntegerParameter( myparams, "Number_GridSpacers", 0 );

        // Check for obsolete properites
        if ( myparams->d_db->keyExists( "Rod_Diameter" ) )
            AMP_WARNING( "Field 'Rod_Diameter' is obsolete and should be removed from database" );
        if ( myparams->d_db->keyExists( "Channel_Diameter" ) )
            AMP_WARNING(
                "Field 'Channel_Diameter' is obsolete and should be removed from database" );
        if ( myparams->d_db->keyExists( "attice_Pitch" ) )
            AMP_WARNING( "Field 'attice_Pitch' is obsolete and should be removed from database" );
        if ( myparams->d_db->keyExists( "ChannelFractions" ) )
            AMP_WARNING(
                "Field 'ChannelFractions' is obsolete and should be removed from database" );
        if ( myparams->d_db->keyExists( "Mass_Flow_Rate" ) )
            AMP_WARNING( "Field 'Mass_Flow_Rate' is obsolete and should be removed from database" );

        // Get the subchannel properties from the mesh
        std::vector<double> x, y;
        Subchannel::getSubchannelProperties( d_Mesh,
                                             myparams->clad_x,
                                             myparams->clad_y,
                                             myparams->clad_d,
                                             x,
                                             y,
                                             d_channelArea,
                                             d_channelDiam,
                                             d_rodDiameter,
                                             d_rodFraction );
        AMP_ASSERT( d_channelArea.size() == d_numSubchannels );
        double total_area = 0.0;
        for ( size_t i = 0; i < d_numSubchannels; i++ )
            total_area += d_channelArea[i];
        d_channelMass.resize( d_numSubchannels, 0.0 );
        for ( size_t i = 0; i < d_numSubchannels; i++ )
            d_channelMass[i] = d_mass * d_channelArea[i] / total_area;

        // get additional parameters based on heat source type
        if ( d_source == "totalHeatGeneration" ) {
            d_Q         = getDoubleParameter( myparams, "Rod_Power", 66.81e3 );
            d_heatShape = getStringParameter( myparams, "Heat_Shape", "Sinusoidal" );
        }

        // get additional parameters based on friction model
        if ( d_frictionModel == "Constant" ) {
            d_friction = getDoubleParameter( myparams, "Friction_Factor", 0.001 );
        } else if ( d_frictionModel == "Selander" ) {
            d_roughness = getDoubleParameter( myparams, "Surface_Roughness", 0.0015e-3 );
        }

        // get form loss parameters if there are grid spacers
        if ( d_NGrid > 0 ) {
            d_zMinGrid = myparams->d_db->getVector<double>( "zMin_GridSpacers" );
            d_zMaxGrid = myparams->d_db->getVector<double>( "zMax_GridSpacers" );
            d_lossGrid = myparams->d_db->getVector<double>( "LossCoefficient_GridSpacers" );
            // check that sizes of grid spacer loss vectors are consistent with the provided number
            // of grid spacers
            if ( !( d_NGrid == d_zMinGrid.size() && d_NGrid == d_zMaxGrid.size() &&
                    d_NGrid == d_lossGrid.size() ) )
                AMP_ERROR(
                    "The size of a grid spacer loss vector is inconsistent with the provided "
                    "number of grid spacers" );
        }

        // get subchannel physics model
        d_subchannelPhysicsModel = myparams->d_subchannelPhysicsModel;

        // Get the subchannel elements
        d_ownSubChannel  = std::vector<bool>( d_numSubchannels, false );
        d_subchannelElem = std::vector<std::vector<AMP::Mesh::MeshElement>>(
            d_numSubchannels, std::vector<AMP::Mesh::MeshElement>( 0 ) );
        auto el = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        for ( size_t i = 0; i < el.size(); i++ ) {
            auto center = el->centroid();
            int index   = getSubchannelIndex( center[0], center[1] );
            if ( index >= 0 ) {
                d_ownSubChannel[index] = true;
                d_subchannelElem[index].push_back( *el );
            }
            ++el;
        }
        d_subchannelFace = std::vector<std::vector<AMP::Mesh::MeshElement>>(
            d_numSubchannels, std::vector<AMP::Mesh::MeshElement>( 0 ) );
        for ( size_t i = 0; i < d_numSubchannels; i++ ) {
            if ( !d_ownSubChannel[i] )
                continue;
            std::shared_ptr<std::vector<AMP::Mesh::MeshElement>> elemPtr( &d_subchannelElem[i],
                                                                          []( auto ) {} );
            auto localSubchannelIt = AMP::Mesh::MeshElementVectorIterator( elemPtr );
            auto localSubchannel   = d_Mesh->Subset( localSubchannelIt, false );
            auto face = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( localSubchannel, 0 );
            for ( size_t j = 0; j < face.size(); j++ ) {
                d_subchannelFace[i].push_back( *face );
                ++face;
            }
        }

        // check to ensure frozen vector isn't null
        d_frozenVec = myparams->d_frozenSolution;
        AMP_INSIST( d_frozenVec, "Null Frozen Vector inside Jacobian" );
        std::shared_ptr<AMP::Discretization::DOFManager> dofMap =
            myparams->d_frozenSolution->getDOFManager();

        // Create the matrix
        d_matrix = AMP::LinearAlgebra::createMatrix( d_frozenVec, d_frozenVec );

        if ( !myparams->d_initialize ) {
            // We are done with the reset
            d_matrix->setIdentity();
            return;
        }

        // calculate extra parameters
        // Constants
        const double g = 9.805; // acceleration due to gravity [m/s2]
        const double h_scale =
            1.0 /
            Subchannel::scaleEnthalpy; // Scale to change the input vector back to correct units
        const double P_scale =
            1.0 /
            Subchannel::scalePressure; // Scale to change the input vector back to correct units
        for ( size_t isub = 0; isub < d_numSubchannels; ++isub ) {
            if ( !d_ownSubChannel[isub] )
                continue;

            // Get the iterator over the faces in the local subchannel
            std::shared_ptr<std::vector<AMP::Mesh::MeshElement>> elemPtr( &d_subchannelFace[isub],
                                                                          []( auto ) {} );
            auto localSubchannelIt = AMP::Mesh::MeshElementVectorIterator( elemPtr );
            AMP_ASSERT( localSubchannelIt.size() == d_z.size() );

            std::vector<size_t> dofs_minus;
            std::vector<size_t> dofs;
            std::vector<size_t> dofs_plus;

            // calculate residual for axial momentum equations
            double A      = d_channelArea[isub]; // Channel area
            double D      = d_channelDiam[isub]; // Channel hydraulic diameter
            double mass   = d_channelMass[isub]; // Mass flow rate in the current subchannel
            int j         = 1;
            auto face     = localSubchannelIt.begin();
            auto end_face = localSubchannelIt.end();
            for ( size_t iface = 0; iface < localSubchannelIt.size(); ++iface, ++j ) {
                dofMap->getDOFs( face->globalID(), dofs );
                // ======================================================
                // energy residual
                // ======================================================
                if ( face == localSubchannelIt.begin() ) {
                    double p_in = P_scale * d_frozenVec->getValueByGlobalID( dofs[1] );
                    d_matrix->setValueByGlobalID( dofs[0], dofs[0], 1.0 );
                    d_matrix->setValueByGlobalID( dofs[0], dofs[1], -1.0 * dhdp( d_Tin, p_in ) );
                } else {
                    // residual at face corresponds to cell below
                    double z_plus = ( face->centroid() )[2];
                    --face;
                    dofMap->getDOFs( face->globalID(), dofs_minus );
                    double z_minus = ( face->centroid() )[2];
                    ++face;
                    double dz = z_plus - z_minus;

                    d_matrix->setValueByGlobalID( dofs[0], dofs_minus[0], -( mass / dz ) );
                    d_matrix->setValueByGlobalID( dofs[0], dofs[0], ( mass / dz ) );
                }

                // ======================================================
                // axial momentum residual
                // ======================================================
                // residual at face corresponds to cell above
                dofMap->getDOFs( face->globalID(), dofs );
                double h_minus = h_scale * d_frozenVec->getValueByGlobalID(
                                               dofs[0] ); // enthalpy evaluated at lower face
                double p_minus = P_scale * d_frozenVec->getValueByGlobalID(
                                               dofs[1] ); // pressure evaluated at lower face
                auto minusFaceCentroid = face->centroid();
                double z_minus         = minusFaceCentroid[2]; // z-coordinate of lower face
                if ( face == end_face - 1 ) {
                    dofMap->getDOFs( face->globalID(), dofs );
                    d_matrix->setValueByGlobalID( dofs[1], dofs[1], 1.0 );
                } else {
                    ++face;
                    dofMap->getDOFs( face->globalID(), dofs_plus );
                    auto plusFaceCentroid = face->centroid();
                    double z_plus         = plusFaceCentroid[2]; // z-coordinate of lower face
                    --face;
                    double h_plus =
                        h_scale * d_frozenVec->getValueByGlobalID(
                                      dofs_plus[0] ); // enthalpy evaluated at upper face
                    double p_plus =
                        P_scale * d_frozenVec->getValueByGlobalID(
                                      dofs_plus[1] ); // pressure evaluated at upper face

                    // evaluate specific volume at upper face
                    std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_plus;
                    volumeArgMap_plus.insert(
                        std::make_pair( std::string( "enthalpy" ),
                                        std::make_shared<std::vector<double>>( 1, h_plus ) ) );
                    volumeArgMap_plus.insert(
                        std::make_pair( std::string( "pressure" ),
                                        std::make_shared<std::vector<double>>( 1, p_plus ) ) );
                    std::vector<double> volumeResult_plus( 1 );
                    d_subchannelPhysicsModel->getProperty(
                        "SpecificVolume", volumeResult_plus, volumeArgMap_plus );
                    double v_plus = volumeResult_plus[0];

                    // evaluate specific volume at lower face
                    std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_minus;
                    volumeArgMap_minus.insert(
                        std::make_pair( std::string( "enthalpy" ),
                                        std::make_shared<std::vector<double>>( 1, h_minus ) ) );
                    volumeArgMap_minus.insert(
                        std::make_pair( std::string( "pressure" ),
                                        std::make_shared<std::vector<double>>( 1, p_minus ) ) );
                    std::vector<double> volumeResult_minus( 1 );
                    d_subchannelPhysicsModel->getProperty(
                        "SpecificVolume", volumeResult_minus, volumeArgMap_minus );
                    double v_minus = volumeResult_minus[0];

                    // evaluate friction factor
                    double fric = friction( h_minus, p_minus, h_plus, p_plus, mass, A, D );

                    // evaluate derivatives of specific volume
                    double dvdh_plus  = dvdh( h_plus, p_plus );
                    double dvdh_minus = dvdh( h_minus, p_minus );
                    double dvdp_plus  = dvdp( h_plus, p_plus );
                    double dvdp_minus = dvdp( h_minus, p_minus );

                    // evaluate derivatives of friction
                    double dfdh_minus = dfdh_lower( h_minus, p_minus, h_plus, p_plus, mass, A, D );
                    double dfdh_plus  = dfdh_upper( h_minus, p_minus, h_plus, p_plus, mass, A, D );
                    double dfdp_minus = dfdp_lower( h_minus, p_minus, h_plus, p_plus, mass, A, D );
                    double dfdp_plus  = dfdp_upper( h_minus, p_minus, h_plus, p_plus, mass, A, D );

                    // compute form loss coefficient
                    double K = 0.0;
                    for ( size_t igrid = 0; igrid < d_lossGrid.size(); igrid++ ) {
                        double zMin_grid = d_zMinGrid[igrid];
                        double zMax_grid = d_zMaxGrid[igrid];
                        AMP_INSIST( ( zMax_grid > zMin_grid ), "Grid spacer zMin > zMax" );
                        double K_grid      = d_lossGrid[igrid];
                        double K_perLength = K_grid / ( zMax_grid - zMin_grid );
                        if ( zMax_grid >= z_plus ) {
                            double overlap = 0.0;
                            if ( zMin_grid >= z_plus ) {
                                overlap = 0.0;
                            } else if ( zMin_grid > z_minus && zMin_grid < z_plus ) {
                                overlap = z_plus - zMin_grid;
                            } else if ( zMin_grid <= z_minus ) {
                                overlap = z_plus - z_minus;
                            } else {
                                AMP_ERROR( "Unexpected position comparison for zMin_grid" );
                            }
                            K += overlap * K_perLength;
                        } else if ( zMax_grid < z_plus && zMax_grid > z_minus ) {
                            double overlap = 0.0;
                            if ( zMin_grid > z_minus ) {
                                overlap = zMax_grid - zMin_grid;
                            } else {
                                overlap = zMax_grid - z_minus;
                            }
                            K += overlap * K_perLength;
                        }
                    }

                    // compute Jacobian entries
                    double dz  = d_z[j] - d_z[j - 1];
                    double A_j = -1.0 * std::pow( mass / A, 2 ) * dvdh_minus -
                                 2.0 * g * dz * std::cos( d_theta ) * dvdh_minus /
                                     std::pow( v_plus + v_minus, 2 ) +
                                 ( 1.0 / 4.0 ) * std::pow( mass / A, 2 ) *
                                     ( ( dz * fric / D + K ) * dvdh_minus +
                                       dz / D * dfdh_minus * ( v_plus + v_minus ) );
                    double B_j = -1.0 * std::pow( mass / A, 2 ) * dvdp_minus -
                                 2.0 * g * dz * std::cos( d_theta ) * dvdp_minus /
                                     std::pow( v_plus + v_minus, 2 ) +
                                 ( 1.0 / 4.0 ) * std::pow( mass / A, 2 ) *
                                     ( ( dz * fric / D + K ) * dvdp_minus +
                                       dz / D * dfdp_minus * ( v_plus + v_minus ) ) -
                                 1;
                    double C_j = std::pow( mass / A, 2 ) * dvdh_plus -
                                 2.0 * g * dz * std::cos( d_theta ) * dvdh_plus /
                                     std::pow( v_plus + v_minus, 2 ) +
                                 ( 1.0 / 4.0 ) * std::pow( mass / A, 2 ) *
                                     ( ( dz * fric / D + K ) * dvdh_plus +
                                       dz / D * dfdh_plus * ( v_plus + v_minus ) );
                    double D_j = std::pow( mass / A, 2 ) * dvdp_plus -
                                 2.0 * g * dz * std::cos( d_theta ) * dvdp_plus /
                                     std::pow( v_plus + v_minus, 2 ) +
                                 ( 1.0 / 4.0 ) * std::pow( mass / A, 2 ) *
                                     ( ( dz * fric / D + K ) * dvdp_plus +
                                       dz / D * dfdp_plus * ( v_plus + v_minus ) ) +
                                 1;

                    d_matrix->setValueByGlobalID( dofs[1], dofs[0], A * A_j );
                    d_matrix->setValueByGlobalID( dofs[1], dofs[1], A * B_j );
                    d_matrix->setValueByGlobalID( dofs[1], dofs_plus[0], A * C_j );
                    d_matrix->setValueByGlobalID( dofs[1], dofs_plus[1], A * D_j );
                }
                ++face;
            }

        } // end of isub
        d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    }
}

// function used in reset to get double parameter or set default if missing
double SubchannelTwoEqLinearOperator::getDoubleParameter(
    std::shared_ptr<const SubchannelOperatorParameters> myparams,
    std::string paramString,
    double defaultValue )
{
    bool keyExists = myparams->d_db->keyExists( paramString );
    if ( keyExists ) {
        return myparams->d_db->getScalar<double>( paramString );
    } else {
        AMP_WARNING( "Key '" + paramString + "' was not provided. Using default value: "
                     << defaultValue << "\n" );
        return defaultValue;
    }
}

// function used in reset to get integer parameter or set default if missing
int SubchannelTwoEqLinearOperator::getIntegerParameter(
    std::shared_ptr<const SubchannelOperatorParameters> myparams,
    std::string paramString,
    int defaultValue )
{
    bool keyExists = myparams->d_db->keyExists( paramString );
    if ( keyExists ) {
        return myparams->d_db->getScalar<int>( paramString );
    } else {
        AMP::pout << "Key '" + paramString + "' was not provided. Using default value: "
                  << defaultValue << "\n";
        return defaultValue;
    }
}

// function used in reset to get string parameter or set default if missing
std::string SubchannelTwoEqLinearOperator::getStringParameter(
    std::shared_ptr<const SubchannelOperatorParameters> myparams,
    std::string paramString,
    std::string defaultValue )
{
    bool keyExists = myparams->d_db->keyExists( paramString );
    if ( keyExists ) {
        return myparams->d_db->getString( paramString );
    } else {
        AMP::pout << "Key '" + paramString + "' was not provided. Using default value: "
                  << defaultValue << "\n";
        return defaultValue;
    }
}


int SubchannelTwoEqLinearOperator::getSubchannelIndex( double x, double y )
{
    size_t i = Utilities::findfirst( d_x, x );
    size_t j = Utilities::findfirst( d_y, y );
    if ( i > 0 && i < d_x.size() && j > 0 && j < d_y.size() )
        return ( i - 1 ) + ( j - 1 ) * ( d_x.size() - 1 );
    return -1;
}

// derivative of enthalpy with respect to pressure
double SubchannelTwoEqLinearOperator::dhdp( double T, double p )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + p ) * b; // perturbation

    // calculate perturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap_pert;
    enthalpyArgMap_pert.insert( std::make_pair( std::string( "temperature" ),
                                                std::make_shared<std::vector<double>>( 1, T ) ) );
    enthalpyArgMap_pert.insert( std::make_pair(
        std::string( "pressure" ), std::make_shared<std::vector<double>>( 1, p + pert ) ) );
    std::vector<double> enthalpyResult_pert( 1 );
    d_subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult_pert, enthalpyArgMap_pert );
    double h_pert = enthalpyResult_pert[0];

    // calculate unperturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
    enthalpyArgMap.insert( std::make_pair( std::string( "temperature" ),
                                           std::make_shared<std::vector<double>>( 1, T ) ) );
    enthalpyArgMap.insert( std::make_pair( std::string( "pressure" ),
                                           std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> enthalpyResult( 1 );
    d_subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
    double h = enthalpyResult[0];

    // calculate derivative
    return ( h_pert - h ) / pert;
}

// derivative of specific volume with respect to enthalpy
double SubchannelTwoEqLinearOperator::dvdh( double h, double p )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + h ) * b; // perturbation

    // calculate perturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> specificVolumeArgMap_pert;
    specificVolumeArgMap_pert.insert( std::make_pair(
        std::string( "enthalpy" ), std::make_shared<std::vector<double>>( 1, h + pert ) ) );
    specificVolumeArgMap_pert.insert( std::make_pair(
        std::string( "pressure" ), std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> specificVolumeResult_pert( 1 );
    d_subchannelPhysicsModel->getProperty(
        "SpecificVolume", specificVolumeResult_pert, specificVolumeArgMap_pert );
    double v_pert = specificVolumeResult_pert[0];

    // calculate unperturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> specificVolumeArgMap;
    specificVolumeArgMap.insert( std::make_pair( std::string( "enthalpy" ),
                                                 std::make_shared<std::vector<double>>( 1, h ) ) );
    specificVolumeArgMap.insert( std::make_pair( std::string( "pressure" ),
                                                 std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> specificVolumeResult( 1 );
    d_subchannelPhysicsModel->getProperty(
        "SpecificVolume", specificVolumeResult, specificVolumeArgMap );
    double v = specificVolumeResult[0];

    // calculate derivative
    return ( v_pert - v ) / pert;
}

// derivative of specific volume with respect to pressure
double SubchannelTwoEqLinearOperator::dvdp( double h, double p )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + p ) * b; // perturbation

    // calculate perturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> specificVolumeArgMap_pert;
    specificVolumeArgMap_pert.insert( std::make_pair(
        std::string( "enthalpy" ), std::make_shared<std::vector<double>>( 1, h ) ) );
    specificVolumeArgMap_pert.insert( std::make_pair(
        std::string( "pressure" ), std::make_shared<std::vector<double>>( 1, p + pert ) ) );
    std::vector<double> specificVolumeResult_pert( 1 );
    d_subchannelPhysicsModel->getProperty(
        "SpecificVolume", specificVolumeResult_pert, specificVolumeArgMap_pert );
    double v_pert = specificVolumeResult_pert[0];

    // calculate unperturbed value
    std::map<std::string, std::shared_ptr<std::vector<double>>> specificVolumeArgMap;
    specificVolumeArgMap.insert( std::make_pair( std::string( "enthalpy" ),
                                                 std::make_shared<std::vector<double>>( 1, h ) ) );
    specificVolumeArgMap.insert( std::make_pair( std::string( "pressure" ),
                                                 std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> specificVolumeResult( 1 );
    d_subchannelPhysicsModel->getProperty(
        "SpecificVolume", specificVolumeResult, specificVolumeArgMap );
    double v = specificVolumeResult[0];

    // calculate derivative
    return ( v_pert - v ) / pert;
}

// friction factor
double SubchannelTwoEqLinearOperator::friction(
    double h_minus, double p_minus, double h_plus, double p_plus, double mass, double A, double D )
{

    if ( d_frictionModel == "Constant" )
        return d_friction;

    std::stringstream ss;
    ss << "Dynamic viscosity calculation may be incorrect" << std::endl
       << "Verify units of the correlation if using non-constant friction model." << std::endl;
    AMP_WARNING( ss.str() );

    double h_avg = ( 1.0 / 2.0 ) * ( h_minus + h_plus ); // enthalpy evaluated at cell center
    double p_avg = ( 1.0 / 2.0 ) * ( p_minus + p_plus ); // pressure evaluated at cell center

    // evaluate specific volume at upper face
    std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_plus;
    volumeArgMap_plus.insert( std::make_pair(
        std::string( "enthalpy" ), std::make_shared<std::vector<double>>( 1, h_plus ) ) );
    volumeArgMap_plus.insert( std::make_pair(
        std::string( "pressure" ), std::make_shared<std::vector<double>>( 1, p_plus ) ) );
    std::vector<double> volumeResult_plus( 1 );
    d_subchannelPhysicsModel->getProperty( "SpecificVolume", volumeResult_plus, volumeArgMap_plus );
    double v_plus = volumeResult_plus[0];

    // evaluate specific volume at lower face
    std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_minus;
    volumeArgMap_minus.insert( std::make_pair(
        std::string( "enthalpy" ), std::make_shared<std::vector<double>>( 1, h_minus ) ) );
    volumeArgMap_minus.insert( std::make_pair(
        std::string( "pressure" ), std::make_shared<std::vector<double>>( 1, p_minus ) ) );
    std::vector<double> volumeResult_minus( 1 );
    d_subchannelPhysicsModel->getProperty(
        "SpecificVolume", volumeResult_minus, volumeArgMap_minus );
    double v_minus = volumeResult_minus[0];

    double v_avg   = 0.5 * ( v_plus + v_minus );
    double rho_avg = 1.0 / v_avg;

    double u_plus  = mass * v_plus / A;                    // velocity evaluated at upper face
    double u_minus = mass * v_minus / A;                   // velocity evaluated at lower face
    double u_avg   = ( 1.0 / 2.0 ) * ( u_minus + u_plus ); // velocity evaluated at cell center

    // evaluate temperature at cell center
    std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
    temperatureArgMap.insert( std::make_pair( std::string( "enthalpy" ),
                                              std::make_shared<std::vector<double>>( 1, h_avg ) ) );
    temperatureArgMap.insert( std::make_pair( std::string( "pressure" ),
                                              std::make_shared<std::vector<double>>( 1, p_avg ) ) );
    std::vector<double> temperatureResult( 1 );
    d_subchannelPhysicsModel->getProperty( "Temperature", temperatureResult, temperatureArgMap );
    double T_avg = temperatureResult[0];

    // evaluate viscosity at cell center
    std::map<std::string, std::shared_ptr<std::vector<double>>> viscosityArgMap;
    viscosityArgMap.insert( std::make_pair( std::string( "temperature" ),
                                            std::make_shared<std::vector<double>>( 1, T_avg ) ) );
    viscosityArgMap.insert( std::make_pair( std::string( "density" ),
                                            std::make_shared<std::vector<double>>( 1, rho_avg ) ) );
    std::vector<double> viscosityResult( 1 );
    d_subchannelPhysicsModel->getProperty( "DynamicViscosity", viscosityResult, viscosityArgMap );
    double visc = viscosityResult[0];

    // evaluate friction factor
    double Re = rho_avg * u_avg * D / visc;
    double fl = 64.0 / Re; // laminar friction factor
    double fric;           // friction factor
    double ft     = 0.;    // turbulent friction factor evaluated from computed Re
    double ft4000 = 0.;    // turbulent friction factor evaluated from Re = 4000
    if ( d_frictionModel == "Blasius" ) {
        ft     = 0.316 * std::pow( Re, -0.25 );
        ft4000 = 0.316 * std::pow( 4000.0, -0.25 );
    } else if ( d_frictionModel == "Drew" ) {
        ft     = 0.0056 + 0.5 * std::pow( Re, -0.32 );
        ft4000 = 0.0056 + 0.5 * std::pow( 4000.0, -0.32 );
    } else if ( d_frictionModel == "Filonenko" ) {
        ft     = std::pow( 1.82 * std::log( Re ) - 1.64, -2 );
        ft4000 = std::pow( 1.82 * std::log( 4000.0 ) - 1.64, -2 );
    } else if ( d_frictionModel == "Selander" ) {
        ft     = 4.0 * std::pow( 3.8 * std::log( 10.0 / Re + 0.2 * d_roughness / D ), -2 );
        ft4000 = 4.0 * std::pow( 3.8 * std::log( 10.0 / 4000.0 + 0.2 * d_roughness / D ), -2 );
    } else {
        AMP_ERROR( "Invalid choice for Friction_Model." );
    }
    if ( Re < 4000.0 )
        fric = std::max( fl, ft4000 );
    else
        fric = ft;

    return fric;
}

// derivative of friction with respect to lower enthalpy
double SubchannelTwoEqLinearOperator::dfdh_lower(
    double h_minus, double p_minus, double h_plus, double p_plus, double mass, double A, double D )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + h_minus ) * b; // perturbation

    double f_pert = friction(
        h_minus + pert, p_minus, h_plus, p_plus, mass, A, D ); // calculate perturbed value
    double f =
        friction( h_minus, p_minus, h_plus, p_plus, mass, A, D ); // calculate unperturbed value

    // calculate derivative
    return ( f_pert - f ) / pert;
}

// derivative of friction with respect to upper enthalpy
double SubchannelTwoEqLinearOperator::dfdh_upper(
    double h_minus, double p_minus, double h_plus, double p_plus, double mass, double A, double D )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + h_plus ) * b; // perturbation

    double f_pert = friction(
        h_minus, p_minus, h_plus + pert, p_plus, mass, A, D ); // calculate perturbed value
    double f =
        friction( h_minus, p_minus, h_plus, p_plus, mass, A, D ); // calculate unperturbed value

    // calculate derivative
    return ( f_pert - f ) / pert;
}

// derivative of friction with respect to lower pressure
double SubchannelTwoEqLinearOperator::dfdp_lower(
    double h_minus, double p_minus, double h_plus, double p_plus, double mass, double A, double D )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + p_minus ) * b; // perturbation

    double f_pert = friction(
        h_minus, p_minus + pert, h_plus, p_plus, mass, A, D ); // calculate perturbed value
    double f =
        friction( h_minus, p_minus, h_plus, p_plus, mass, A, D ); // calculate unperturbed value

    // calculate derivative
    return ( f_pert - f ) / pert;
}

// derivative of friction with respect to upper pressure
double SubchannelTwoEqLinearOperator::dfdp_upper(
    double h_minus, double p_minus, double h_plus, double p_plus, double mass, double A, double D )
{
    // calculate perturbation
    double b    = std::pow( d_machinePrecision, 0.5 );
    double pert = ( 1.0 + p_plus ) * b; // perturbation

    double f_pert = friction(
        h_minus, p_minus, h_plus, p_plus + pert, mass, A, D ); // calculate perturbed value
    double f =
        friction( h_minus, p_minus, h_plus, p_plus, mass, A, D ); // calculate unperturbed value

    // calculate derivative
    return ( f_pert - f ) / pert;
}
} // namespace AMP::Operator
