#include "AMP/operators/subchannel/SubchannelTwoEqNonlinearOperator.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "ProfilerApp.h"

#include <string>


namespace AMP::Operator {


// Constructor
SubchannelTwoEqNonlinearOperator::SubchannelTwoEqNonlinearOperator(
    std::shared_ptr<const OperatorParameters> inparams )
    : Operator( inparams ),
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
      d_numSubchannels( 0 )
{
    auto params = Subchannel::convert( inparams );
    AMP_INSIST( params->d_db->keyExists( "InputVariable" ), "Key 'InputVariable' does not exist" );
    std::string inpVar = params->d_db->getString( "InputVariable" );
    d_inpVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );

    AMP_INSIST( params->d_db->keyExists( "OutputVariable" ),
                "Key 'OutputVariable' does not exist" );
    std::string outVar = params->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );

    d_params      = params;
    d_initialized = false;
}


// reset
void SubchannelTwoEqNonlinearOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    if ( params ) {

        auto myparams = std::dynamic_pointer_cast<const SubchannelOperatorParameters>( params );
        AMP_INSIST( ( ( myparams.get() ) != nullptr ), "NULL parameters" );
        AMP_INSIST( ( ( ( myparams->d_db ).get() ) != nullptr ), "NULL database" );
        d_params = myparams;

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
        if ( myparams->d_db->keyExists( "Lattice_Pitch" ) )
            AMP_WARNING( "Field 'Lattice_Pitch' is obsolete and should be removed from database" );
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
        if ( ( d_source == "totalHeatGeneration" ) ||
             ( d_source == "totalHeatGenerationWithDiscretizationError" ) ) {
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

        d_initialized = true;
    }
}


// apply
void SubchannelTwoEqNonlinearOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                              AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );

    // Check that the operator has been initialized
    if ( !d_initialized )
        reset( d_params );

    // ensure that solution and residual vectors aren't NULL
    AMP_INSIST( ( ( r.get() ) != nullptr ), "NULL Residual Vector" );
    AMP_INSIST( ( ( u.get() ) != nullptr ), "NULL Solution Vector" );
    AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                "Input vector is in an inconsistent state" );

    // Constants
    const double pi = 4.0 * atan( 1.0 ); // pi
    const double g  = 9.805;             // acceleration due to gravity [m/s2]
    const double h_scale =
        1.0 / Subchannel::scaleEnthalpy; // Scale to change the input vector back to correct units
    const double P_scale =
        1.0 / Subchannel::scalePressure; // Scale to change the input vector back to correct units

    // Subset the vectors
    auto inputVec  = subsetInputVector( u );
    auto outputVec = subsetOutputVector( r );

    auto dof_manager = inputVec->getDOFManager();
    std::shared_ptr<AMP::Discretization::DOFManager> cladDofManager;
    if ( d_source == "averageCladdingTemperature" ) {
        cladDofManager = d_cladTemperature->getDOFManager();
    }

    for ( size_t isub = 0; isub < d_numSubchannels; ++isub ) {
        if ( !d_ownSubChannel[isub] )
            continue;
        PROFILE( "apply-subchannel" );

        // Get the iterator over the faces in the local subchannel
        std::shared_ptr<std::vector<AMP::Mesh::MeshElement>> elemPtr( &d_subchannelFace[isub],
                                                                      []( auto ) {} );
        auto localSubchannelIt = AMP::Mesh::MeshElementVectorIterator( elemPtr );
        AMP_ASSERT( localSubchannelIt.size() == d_z.size() );

        // get solution sizes
        const size_t numFaces = d_z.size();
        const size_t numCells = numFaces - 1;

        std::vector<size_t> dofs;
        dof_manager->getDOFs( ( localSubchannelIt.begin() )->globalID(), dofs );

        double h_in = h_scale * inputVec->getValueByGlobalID( dofs[0] );
        double P_in = P_scale * inputVec->getValueByGlobalID( dofs[1] );

        // evaluate enthalpy at inlet
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert( std::make_pair(
            std::string( "temperature" ), std::make_shared<std::vector<double>>( 1, d_Tin ) ) );
        enthalpyArgMap.insert( std::make_pair( std::string( "pressure" ),
                                               std::make_shared<std::vector<double>>( 1, P_in ) ) );
        std::vector<double> enthalpyResult( 1 );
        d_subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        double h_eval = enthalpyResult[0];

        // compute the enthalpy change in each interval
        std::vector<double> flux( numCells );
        if ( d_source == "averageCladdingTemperature" ) {
            auto face = localSubchannelIt.begin();
            std::vector<AMP::Mesh::MeshElementID> face_ids( face.size() );
            for ( size_t j = 0; j < face.size(); j++ ) {
                auto center = face->centroid();
                AMP_ASSERT( Utilities::approx_equal( center[2], d_z[j] ) );
                face_ids[j] = face->globalID();
                ++face;
            }
            flux = Subchannel::getHeatFluxClad( d_z,
                                                face_ids,
                                                d_channelDiam[isub],
                                                d_reynolds,
                                                d_prandtl,
                                                d_rodFraction[isub],
                                                d_subchannelPhysicsModel,
                                                inputVec,
                                                d_cladTemperature );
        } else if ( d_source == "averageHeatFlux" ) {
            AMP_ERROR( "Heat source type 'averageHeatFlux' not yet implemented." );
        } else if ( d_source == "totalHeatGeneration" ) {
            flux = Subchannel::getHeatFluxGeneration( d_heatShape, d_z, d_rodDiameter[isub], d_Q );
        } else if ( d_source == "totalHeatGenerationWithDiscretizationError" ) {
            flux = Subchannel::getHeatFluxGenerationWithDiscretizationError(
                d_heatShape, d_z, d_rodDiameter[isub], d_Q );
        } else {
            AMP_ERROR( "Heat source type '" + d_source + "' is invalid" );
        }
        std::vector<double> dh( numCells );
        for ( size_t j = 0; j < numCells; j++ ) {
            double dz       = d_z[j + 1] - d_z[j];
            double flux_sum = pi * d_rodDiameter[isub] * flux[j];
            double lin_sum  = d_gamma * pi * d_rodDiameter[isub] * flux[j];
            dh[j]           = ( dz / d_channelMass[isub] ) * ( flux_sum + lin_sum );
        }

        // calculate residual for axial momentum equations
        double A    = d_channelArea[isub]; // Channel area
        double D    = d_channelDiam[isub]; // Channel hydraulic diameter
        double mass = d_channelMass[isub]; // Mass flow rate in the current subchannel
        double R_h, R_p;
        int j         = 1;
        auto face     = localSubchannelIt.begin();
        auto end_face = localSubchannelIt.end();
        for ( size_t iface = 0; iface < localSubchannelIt.size(); ++iface, ++j ) {

            // ======================================================
            // energy residual
            // ======================================================
            if ( face == localSubchannelIt.begin() ) {
                // evaluate first residual entry, corresponding to inlet enthalpy:
                //    \f[ R_0 = h_{in} - h(T_{in},p_{1-})\f]
                R_h = h_in - h_eval;
            } else {
                // residual at face corresponds to cell below
                dof_manager->getDOFs( face->globalID(), dofs );
                double h_plus = h_scale * inputVec->getValueByGlobalID(
                                              dofs[0] ); // enthalpy evaluated at lower face
                double z_plus = ( face->centroid() )[2];
                --face;
                dof_manager->getDOFs( face->globalID(), dofs );
                double h_minus = h_scale * inputVec->getValueByGlobalID(
                                               dofs[0] ); // enthalpy evaluated at lower face
                double z_minus = ( face->centroid() )[2];
                ++face;
                double dz = z_plus - z_minus;
                R_h       = mass / dz * ( h_plus - h_minus - dh[j - 2] );
            }

            // ======================================================
            // axial momentum residual
            // ======================================================
            // residual at face corresponds to cell above
            dof_manager->getDOFs( face->globalID(), dofs );
            double h_minus = h_scale * inputVec->getValueByGlobalID(
                                           dofs[0] ); // enthalpy evaluated at lower face
            double p_minus = P_scale * inputVec->getValueByGlobalID(
                                           dofs[1] ); // pressure evaluated at lower face
            auto minusFaceCentroid = face->centroid();
            double z_minus         = minusFaceCentroid[2]; // z-coordinate of lower face
            if ( face == end_face - 1 ) {
                R_p = p_minus - d_Pout;
            } else {
                ++face;
                dof_manager->getDOFs( face->globalID(), dofs );
                double h_plus = h_scale * inputVec->getValueByGlobalID(
                                              dofs[0] ); // enthalpy evaluated at upper face
                double p_plus = P_scale * inputVec->getValueByGlobalID(
                                              dofs[1] ); // pressure evaluated at upper face
                auto plusFaceCentroid = face->centroid();
                double z_plus         = plusFaceCentroid[2]; // z-coordinate of lower face
                --face;

                double h_avg =
                    ( 1.0 / 2.0 ) * ( h_minus + h_plus ); // enthalpy evaluated at cell center
                double p_avg =
                    ( 1.0 / 2.0 ) * ( p_minus + p_plus ); // pressure evaluated at cell center

                // evaluate density at upper face
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
                double rho_plus = 1.0 / volumeResult_plus[0];

                // evaluate density at lower face
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
                double rho_minus = 1.0 / volumeResult_minus[0];

                double u_plus  = mass / ( A * rho_plus );  // velocity evaluated at upper face
                double u_minus = mass / ( A * rho_minus ); // velocity evaluated at lower face
                double u_avg =
                    ( 1.0 / 2.0 ) * ( u_minus + u_plus ); // velocity evaluated at cell center

                // evaluate density at cell center
                std::map<std::string, std::shared_ptr<std::vector<double>>> volumeArgMap_avg;
                volumeArgMap_avg.insert(
                    std::make_pair( std::string( "enthalpy" ),
                                    std::make_shared<std::vector<double>>( 1, h_avg ) ) );
                volumeArgMap_avg.insert(
                    std::make_pair( std::string( "pressure" ),
                                    std::make_shared<std::vector<double>>( 1, p_avg ) ) );
                std::vector<double> volumeResult_avg( 1 );
                d_subchannelPhysicsModel->getProperty(
                    "SpecificVolume", volumeResult_avg, volumeArgMap_avg );
                double rho_avg = 1.0 / volumeResult_avg[0];

                double fric; // friction factor
                if ( d_frictionModel == "Constant" ) {
                    fric = d_friction;
                } else {
                    std::stringstream ss;
                    ss << "Dynamic viscosity calculation may be incorrect" << std::endl
                       << "Verify units of the correlation if using non-constant friction model."
                       << std::endl;
                    AMP_WARNING( ss.str() );

                    // evaluate temperature at cell center
                    std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
                    temperatureArgMap.insert(
                        std::make_pair( std::string( "enthalpy" ),
                                        std::make_shared<std::vector<double>>( 1, h_avg ) ) );
                    temperatureArgMap.insert(
                        std::make_pair( std::string( "pressure" ),
                                        std::make_shared<std::vector<double>>( 1, p_avg ) ) );
                    std::vector<double> temperatureResult( 1 );
                    d_subchannelPhysicsModel->getProperty(
                        "Temperature", temperatureResult, temperatureArgMap );
                    double T_avg = temperatureResult[0];

                    // evaluate viscosity at cell center
                    std::map<std::string, std::shared_ptr<std::vector<double>>> viscosityArgMap;
                    viscosityArgMap.insert(
                        std::make_pair( std::string( "temperature" ),
                                        std::make_shared<std::vector<double>>( 1, T_avg ) ) );
                    viscosityArgMap.insert(
                        std::make_pair( std::string( "density" ),
                                        std::make_shared<std::vector<double>>( 1, rho_avg ) ) );
                    std::vector<double> viscosityResult( 1 );
                    d_subchannelPhysicsModel->getProperty(
                        "DynamicViscosity", viscosityResult, viscosityArgMap );
                    double visc = viscosityResult[0];

                    // evaluate friction factor
                    double Re     = rho_avg * u_avg * D / visc;
                    double fl     = 64.0 / Re; // laminar friction factor
                    double ft     = 0.; // turbulent friction factor evaluated from computed Re
                    double ft4000 = 0.; // turbulent friction factor evaluated from Re = 4000
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
                        ft = 4.0 *
                             std::pow( 3.8 * std::log( 10.0 / Re + 0.2 * d_roughness / D ), -2 );
                        ft4000 =
                            4.0 *
                            std::pow( 3.8 * std::log( 10.0 / 4000.0 + 0.2 * d_roughness / D ), -2 );
                    } else {
                        AMP_ERROR( "Invalid choice for Friction_Model." );
                    }
                    if ( Re < 4000.0 )
                        fric = std::max( fl, ft4000 );
                    else
                        fric = ft;
                }

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

                // evaluate residual: axial momentum equation
                double dz = d_z[j] - d_z[j - 1];
                R_p = mass * ( u_plus - u_minus ) + g * A * dz * rho_avg * std::cos( d_theta ) +
                      0.5 * ( dz * fric / D + K ) * std::abs( mass / ( A * rho_avg ) ) * mass +
                      A * ( p_plus - p_minus );
            }

            // put residual value in residual vector
            dof_manager->getDOFs( face->globalID(), dofs );
            const double v1 = Subchannel::scaleEnthalpy * R_h;
            const double v2 = Subchannel::scalePressure * R_p;
            outputVec->setValuesByGlobalID( 1, &dofs[0], &v1 );
            outputVec->setValuesByGlobalID( 1, &dofs[1], &v2 );
            ++face;
        }
    } // end of isub

    outputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

std::shared_ptr<OperatorParameters> SubchannelTwoEqNonlinearOperator::getJacobianParameters(
    AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{
    std::shared_ptr<AMP::Database> tmp_db = d_params->d_db->cloneDatabase();
    Operator::setMemoryAndBackendParameters( tmp_db );
    tmp_db->putScalar( "name", "SubchannelTwoEqLinearOperator" );
    tmp_db->putScalar( "InputVariable", d_inpVariable->getName() );
    tmp_db->putScalar( "OutputVariable", d_outVariable->getName() );

    auto outParams              = std::make_shared<SubchannelOperatorParameters>( tmp_db );
    auto u                      = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );
    outParams->d_frozenSolution = subsetInputVector( u );
    outParams->d_initialize     = true;
    outParams->d_subchannelPhysicsModel = d_subchannelPhysicsModel;
    outParams->d_Mesh                   = d_params->d_Mesh;
    outParams->clad_x                   = d_params->clad_x;
    outParams->clad_y                   = d_params->clad_y;
    outParams->clad_d                   = d_params->clad_d;

    return outParams;
}


// function used in reset to get double parameter or set default if missing
double SubchannelTwoEqNonlinearOperator::getDoubleParameter(
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
int SubchannelTwoEqNonlinearOperator::getIntegerParameter(
    std::shared_ptr<const SubchannelOperatorParameters> myparams,
    std::string paramString,
    int defaultValue )
{
    bool keyExists = myparams->d_db->keyExists( paramString );
    if ( keyExists ) {
        return myparams->d_db->getScalar<int>( paramString );
    } else {
        AMP_WARNING( "Key '" + paramString + "' was not provided. Using default value: "
                     << defaultValue << "\n" );
        return defaultValue;
    }
}

// function used in reset to get string parameter or set default if missing
std::string SubchannelTwoEqNonlinearOperator::getStringParameter(
    std::shared_ptr<const SubchannelOperatorParameters> myparams,
    std::string paramString,
    std::string defaultValue )
{
    bool keyExists = myparams->d_db->keyExists( paramString );
    if ( keyExists ) {
        return myparams->d_db->getString( paramString );
    } else {
        AMP_WARNING( "Key '" + paramString + "' was not provided. Using default value: "
                     << defaultValue << "\n" );
        return defaultValue;
    }
}


int SubchannelTwoEqNonlinearOperator::getSubchannelIndex( double x, double y )
{
    size_t i = Utilities::findfirst( d_x, x );
    size_t j = Utilities::findfirst( d_y, y );
    if ( i > 0 && i < d_x.size() && j > 0 && j < d_y.size() )
        return ( i - 1 ) + ( j - 1 ) * ( d_x.size() - 1 );
    return -1;
}
} // namespace AMP::Operator
