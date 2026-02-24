#include "AMP/operators/subchannel/SubchannelFourEqNonlinearOperator.h"
#include "AMP/mesh/MeshListIterator.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"

#include <string>


namespace AMP::Operator {


template<class TYPE>
static TYPE get( std::shared_ptr<const SubchannelOperatorParameters> params,
                 const std::string &key,
                 const TYPE &val )
{
    auto &db = params->d_db;
    if ( db->keyExists( key ) ) {
        return db->getScalar<TYPE>( key );
    } else {
        AMP_WARNING( "Key '" + key + "' was not provided. Using default value: " << val );
        return val;
    }
}


// Constructor
SubchannelFourEqNonlinearOperator::SubchannelFourEqNonlinearOperator(
    std::shared_ptr<const OperatorParameters> inparams )
    : Operator( inparams ),
      d_forceNoConduction( false ),
      d_forceNoTurbulence( false ),
      d_forceNoHeatSource( false ),
      d_forceNoFriction( false ),
      d_Pout( 0 ),
      d_Tin( 0 ),
      d_mass( 0 ),
      d_win( 0 ),
      d_gamma( 0 ),
      d_theta( 0 ),
      d_turbulenceCoef( 0 ),
      d_reynolds( 0 ),
      d_prandtl( 0 ),
      d_KG( 0 ),
      d_friction( 0 ),
      d_roughness( 0 ),
      d_NGrid( 0 ),
      d_Q( 0 ),
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
void SubchannelFourEqNonlinearOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myparams = std::dynamic_pointer_cast<const SubchannelOperatorParameters>( params );
    AMP_INSIST( ( ( myparams.get() ) != nullptr ), "NULL parameters" );
    AMP_INSIST( ( ( ( myparams->d_db ).get() ) != nullptr ), "NULL database" );
    d_params = myparams;

    // Get the subchannel mesh coordinates
    AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( d_Mesh, d_x, d_y, d_z );
    d_numSubchannels = ( d_x.size() - 1 ) * ( d_y.size() - 1 );

    // Get the properties from the database
    d_Pout           = get<double>( myparams, "Exit_Pressure", 15.5132e6 );
    d_Tin            = get<double>( myparams, "Inlet_Temperature", 569.26 );
    d_mass           = get<double>( myparams, "Inlet_Mass_Flow_Rate", 0.3522 * d_numSubchannels );
    d_win            = get<double>( myparams, "Inlet_Lateral_Flow_Rate", 0.0 );
    d_gamma          = get<double>( myparams, "Fission_Heating_Coefficient", 0.0 );
    d_theta          = get<double>( myparams, "Channel_Angle", 0.0 );
    d_reynolds       = get<double>( myparams, "Reynolds", 0.0 );
    d_prandtl        = get<double>( myparams, "Prandtl", 0.0 );
    d_friction       = get<double>( myparams, "Friction_Factor", 0.001 );
    d_turbulenceCoef = get<double>( myparams, "Turbulence_Coefficient", 1.0 );
    d_KG             = get<double>( myparams, "Lateral_Form_Loss_Coefficient", 0.2 );
    d_forceNoConduction = get<bool>( myparams, "Force_No_Conduction", false );
    d_forceNoTurbulence = get<bool>( myparams, "Force_No_Turbulence", false );
    d_forceNoHeatSource = get<bool>( myparams, "Force_No_Heat_Source", false );
    d_forceNoFriction   = get<bool>( myparams, "Force_No_Friction", false );

    // get additional parameters based on heat source type
    d_source = get<std::string>( myparams, "Heat_Source_Type", "totalHeatGeneration" );
    if ( ( d_source == "totalHeatGeneration" ) ||
         ( d_source == "totalHeatGenerationWithDiscretizationError" ) ) {
        d_Q         = get<double>( myparams, "Max_Rod_Power", 66.0e3 );
        d_QFraction = myparams->d_db->getVector<double>( "Rod_Power_Fraction" );
        d_heatShape = get<std::string>( myparams, "Heat_Shape", "Sinusoidal" );
    }

    // get additional parameters based on friction model
    d_frictionModel = get<std::string>( myparams, "Friction_Model", "Constant" );
    if ( d_frictionModel == "Constant" ) {
        d_friction = get<double>( myparams, "Friction_Factor", 0.001 );
    } else if ( d_frictionModel == "Selander" ) {
        d_roughness = get<double>( myparams, "Surface_Roughness", 0.0015e-3 );
    }

    // get form loss parameters if there are grid spacers
    d_NGrid = get<int>( myparams, "Number_GridSpacers", 0 );
    if ( d_NGrid > 0 ) {
        d_zMinGrid = myparams->d_db->getVector<double>( "zMin_GridSpacers" );
        d_zMaxGrid = myparams->d_db->getVector<double>( "zMax_GridSpacers" );
        d_lossGrid = myparams->d_db->getVector<double>( "LossCoefficient_GridSpacers" );
        // check that sizes of grid spacer loss vectors are consistent with the provided number of
        // grid spacers
        if ( !( d_NGrid == d_zMinGrid.size() && d_NGrid == d_zMaxGrid.size() &&
                d_NGrid == d_lossGrid.size() ) )
            AMP_ERROR( "The size of a grid spacer loss vector is inconsistent with the provided "
                       "number of grid spacers" );
    }

    // get subchannel physics model
    d_subchannelPhysicsModel = myparams->d_subchannelPhysicsModel;

    // Check for obsolete properites
    if ( myparams->d_db->keyExists( "Rod_Diameter" ) )
        AMP_WARNING( "Field 'Rod_Diameter' is obsolete and should be removed from database" );
    if ( myparams->d_db->keyExists( "Channel_Diameter" ) )
        AMP_WARNING( "Field 'Channel_Diameter' is obsolete and should be removed from database" );
    if ( myparams->d_db->keyExists( "Lattice_Pitch" ) )
        AMP_WARNING( "Field 'Lattice_Pitch' is obsolete and should be removed from database" );
    if ( myparams->d_db->keyExists( "ChannelFractions" ) )
        AMP_WARNING( "Field 'ChannelFractions' is obsolete and should be removed from database" );
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
    // compute total area
    double total_area = 0.0;
    for ( size_t i = 0; i < d_numSubchannels; i++ )
        total_area += d_channelArea[i];
    // inlet mass flow rate for each subchannel
    d_channelMass.resize( d_numSubchannels, 0.0 );
    for ( size_t i = 0; i < d_numSubchannels; i++ )
        d_channelMass[i] = d_mass * d_channelArea[i] / total_area;

    // Get the subchannel elements
    d_ownSubChannel  = std::vector<bool>( d_numSubchannels, false );
    d_subchannelElem = std::vector<std::vector<ElementPtr>>( d_numSubchannels );
    auto el          = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    for ( size_t i = 0; i < el.size(); i++ ) {
        auto center = el->centroid();
        int index   = getSubchannelIndex( center[0], center[1] );
        if ( index >= 0 ) {
            d_ownSubChannel[index] = true;
            d_subchannelElem[index].push_back( el->clone() );
        }
        ++el;
    }
    d_subchannelFace = std::vector<std::vector<ElementPtr>>( d_numSubchannels );
    for ( size_t i = 0; i < d_numSubchannels; i++ ) {
        if ( !d_ownSubChannel[i] )
            continue;
        std::shared_ptr<std::vector<ElementPtr>> elemPtr( &d_subchannelElem[i], []( auto ) {} );
        auto localSubchannelIt = AMP::Mesh::MeshListIterator( elemPtr );
        auto localSubchannel   = d_Mesh->Subset( localSubchannelIt, false );
        auto face = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( localSubchannel, 0 );
        for ( size_t j = 0; j < face.size(); j++ ) {
            d_subchannelFace[i].push_back( face->clone() );
            ++face;
        }
    }

    d_initialized = true;
}

// function used to get all lateral gaps
void SubchannelFourEqNonlinearOperator::getLateralFaces(
    std::shared_ptr<AMP::Mesh::Mesh> mesh,
    std::map<AMP::Mesh::Point, ElementPtr> &interiorLateralFaceMap,
    std::map<AMP::Mesh::Point, ElementPtr> &exteriorLateralFaceMap )
{
    // get iterator over all faces of mesh
    AMP::Mesh::MeshIterator face = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    // loop over faces
    for ( ; face != face.end(); ++face ) {
        // check that face is vertical
        // ---------------------------
        // get centroid of current face
        auto faceCentroid = face->centroid();
        // get vertices of current face
        auto vertices = face->getElements( AMP::Mesh::GeomType::Vertex );

        bool perpindicular_to_x = true; // is the current face perpindicular to x-axis?
        bool perpindicular_to_y = true; // is the current face perpindicular to y-axis?
        // loop over vertices of current face
        for ( auto &vertice : vertices ) {
            // get coordinates of current vertex
            auto vertexCoord = vertice.coord();
            // if any vertex does not have the same x-coordinate as the face centroid,
            if ( !AMP::Utilities::approx_equal( vertexCoord[0], faceCentroid[0], 1.0e-6 ) )
                // then the face is not perpindicular to x-axis
                perpindicular_to_x = false;
            // if any vertex does not have the same y-coordinate as the face centroid,
            if ( !AMP::Utilities::approx_equal( vertexCoord[1], faceCentroid[1], 1.0e-6 ) )
                // then the face is not perpindicular to y-axis
                perpindicular_to_y = false;
        }
        // check that face is in the interior of the mesh; it must have two adjacent cells
        // -------------------------------------------------------------------------------
        // if the face is vertical
        if ( perpindicular_to_x || perpindicular_to_y ) {
            // if the face has more than 1 adjacent cell
            if ( ( mesh->getElementParents( *face, AMP::Mesh::GeomType::Cell ) ).size() > 1 ) {
                // insert face into map with centroid
                interiorLateralFaceMap.insert(
                    std::pair<AMP::Mesh::Point, ElementPtr>( faceCentroid, face->clone() ) );
            } else {
                // insert face into exterior lateral face map with centroid
                exteriorLateralFaceMap.insert(
                    std::pair<AMP::Mesh::Point, ElementPtr>( faceCentroid, face->clone() ) );
            }
        }
    } // end loop over faces
    return;
}

// function to map x,y position to gap widths
std::map<AMP::Mesh::Point, double>
SubchannelFourEqNonlinearOperator::getGapWidths( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                                 const std::vector<double> &clad_x,
                                                 const std::vector<double> &clad_y,
                                                 const std::vector<double> &clad_d )
{
    std::map<AMP::Mesh::Point, double> gapWidthMap;
    size_t Nz   = d_z.size() - 1;
    double topZ = 0.5 * ( d_z[Nz] + d_z[Nz - 1] );
    // get iterator over all faces of mesh
    AMP::Mesh::MeshIterator face = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; face != face.end(); ++face ) {
        auto faceCentroid = face->centroid();
        if ( AMP::Utilities::approx_equal( faceCentroid[2], topZ, 1.0e-12 ) ) {
            // if the face has more than 1 adjacent cell
            if ( ( mesh->getElementParents( *face, AMP::Mesh::GeomType::Cell ) ).size() > 1 ) {
                // create vector of xy position of gap face
                AMP::Mesh::Point xyPos( faceCentroid[0], faceCentroid[1] );
                // get vertices of current face
                auto vertices = face->getElements( AMP::Mesh::GeomType::Vertex );
                // loop over vertices of current face
                bool topVertex1Found = false;
                double x1            = 0.0;
                double y1            = 0.0;
                double correction    = 0.0; // gap width correction for exclusion of clad
                double gapWidth      = 0.0;
                for ( auto &vertice : vertices ) {
                    // get coordinates of current vertex
                    auto vertexCoord = vertice.coord();
                    if ( AMP::Utilities::approx_equal( vertexCoord[2], d_z[Nz], 1.0e-12 ) ) {
                        if ( topVertex1Found ) { // second top vertex has been found
                            // check for clad centered at this vertex
                            for ( size_t k = 0; k < clad_x.size(); k++ ) {
                                if ( fabs( clad_x[k] - x1 ) < 1.0e-12 &&
                                     fabs( clad_y[k] - y1 ) < 1.0e-12 ) {
                                    correction = correction - 0.5 * clad_d[k];
                                    break;
                                }
                            }
                            // calculate gap width
                            gapWidth = std::pow( std::pow( x1 - vertexCoord[0], 2 ) +
                                                     std::pow( y1 - vertexCoord[1], 2 ),
                                                 0.5 ) +
                                       correction;
                            break;
                        } else { // first top vertex has been found
                            x1 = vertexCoord[0];
                            y1 = vertexCoord[1];
                            // check for clad centered at this vertex
                            for ( size_t k = 0; k < clad_x.size(); k++ ) {
                                if ( fabs( clad_x[k] - x1 ) < 1.0e-12 &&
                                     fabs( clad_y[k] - y1 ) < 1.0e-12 ) {
                                    correction = -0.5 * clad_d[k];
                                    break;
                                }
                            }
                            topVertex1Found = true; // first top vertex has been found
                        }
                    }
                }
                // insert face into map with xy position
                gapWidthMap.insert( std::pair<AMP::Mesh::Point, double>( xyPos, gapWidth ) );
            }
        }
    }
    return gapWidthMap;
}

// function used to get all of the unique x,y,z points in subchannel mesh
void SubchannelFourEqNonlinearOperator::fillSubchannelGrid( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    // Create the grid for all processors
    std::set<double> x, y, z;
    if ( mesh ) {
        AMP::Mesh::MeshIterator vertex = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
        // for all vertices in mesh
        for ( size_t i = 0; i < vertex.size(); i++ ) {
            auto coord = vertex->coord();
            AMP_ASSERT( coord.size() == 3 );
            // insert x,y,z points into sets, even if duplicate
            x.insert( coord[0] );
            y.insert( coord[1] );
            z.insert( coord[2] );
            ++vertex;
        }
    }
    d_Mesh->getComm().setGather( x );
    d_Mesh->getComm().setGather( y );
    d_Mesh->getComm().setGather( z );
    double last = 1.0e300; // arbitary large number
    // erase duplicate x points
    auto it = x.begin();
    while ( it != x.end() ) {
        if ( Utilities::approx_equal( last, *it, 1e-12 ) ) {
            x.erase( it++ ); // increments before erasing
        } else {
            last = *it;
            ++it;
        }
    }
    // erase duplicate y points
    last = 1.0e300; // arbitary large number
    it   = y.begin();
    while ( it != y.end() ) {
        if ( Utilities::approx_equal( last, *it, 1e-12 ) ) {
            y.erase( it++ ); // increments before erasing
        } else {
            last = *it;
            ++it;
        }
    }
    // erase duplicate z points
    last = 1.0e300; // arbitary large number
    it   = z.begin();
    while ( it != z.end() ) {
        if ( Utilities::approx_equal( last, *it, 1e-12 ) ) {
            z.erase( it++ ); // increments before erasing
        } else {
            last = *it;
            ++it;
        }
    }
    d_x       = std::vector<double>( x.begin(), x.end() );
    d_y       = std::vector<double>( y.begin(), y.end() );
    d_z       = std::vector<double>( z.begin(), z.end() );
    size_t Nx = d_x.size() - 1; // number of mesh divisions along x-axis
    size_t Ny = d_y.size() - 1; // number of mesh divisions along y-axis
    size_t Nz = d_z.size() - 1; // number of mesh divisions along z-axis
    if ( mesh )
        // check that computed number of elements matches that found by numGlobalElements()
        AMP_ASSERT( Nx * Ny * Nz == mesh->numGlobalElements( AMP::Mesh::GeomType::Cell ) );
    // compute number of subchannels
    d_numSubchannels = Nx * Ny;
}

// function to get a unique index for a subchannel based on its x,y coordinates
int SubchannelFourEqNonlinearOperator::getSubchannelIndex( double x, double y )
{
    // get index of first entry in subchannel x mesh >= x
    size_t ix = Utilities::findfirst( d_x, x );
    // get index of first entry in subchannel y mesh >= y
    size_t iy = Utilities::findfirst( d_y, y );
    // check that indices are valid
    if ( ix > 0 && ix < d_x.size() && iy > 0 && iy < d_y.size() ) {
        size_t Nx = d_x.size() - 1;
        size_t Ny = d_y.size() - 1;
        return ( ix - 1 ) + ( ( Ny - 1 ) - ( iy - 1 ) ) * Nx;
    } else {
        AMP_ERROR( "Invalid indices found for getSubchannelIndex()" );
    }
    return 0;
}

// apply
void SubchannelFourEqNonlinearOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
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
    const double pi      = 4.0 * atan( 1.0 ); // pi
    const double gravity = 9.805;             // acceleration due to gravity [m/s2]
    const double m_scale = 1.0 / Subchannel::scaleAxialMassFlowRate;
    const double h_scale = 1.0 / Subchannel::scaleEnthalpy;
    const double p_scale = 1.0 / Subchannel::scalePressure;
    const double w_scale = 1.0 / Subchannel::scaleLateralMassFlowRate;

    // Subset the vectors
    auto inputVec  = subsetInputVector( u );
    auto outputVec = subsetOutputVector( r );

    auto dof_manager = inputVec->getDOFManager();
    std::shared_ptr<AMP::Discretization::DOFManager> cladDofManager;
    if ( d_source == "averageCladdingTemperature" ) {
        cladDofManager = d_cladTemperature->getDOFManager();
    }

    // get all of the unique x,y,z points in subchannel mesh
    fillSubchannelGrid( d_Mesh );

    // get map of all of the lateral faces to their centroids
    std::map<AMP::Mesh::Point, ElementPtr> interiorLateralFaceMap;
    std::map<AMP::Mesh::Point, ElementPtr> exteriorLateralFaceMap;
    getLateralFaces( d_Mesh, interiorLateralFaceMap, exteriorLateralFaceMap );

    // get map of gap widths to their xy positions
    auto gapWidthMap = getGapWidths( d_Mesh, d_params->clad_x, d_params->clad_y, d_params->clad_d );

    // compute height of subchannels
    auto box            = d_Mesh->getBoundingBox();
    const double height = box[5] - box[4];

    // create vector of the mid points of each axial interval
    std::vector<double> zMid( d_z.size() - 1 );
    for ( size_t j = 0; j < d_z.size() - 1; ++j )
        zMid[j] = d_z[j] + 0.5 * ( d_z[j + 1] - d_z[j] );

    auto cell = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 ); // iterator for cells of mesh

    std::vector<std::vector<ElementPtr>> d_elem( d_numSubchannels ); // elements for each subchannel
    std::vector<bool> d_ownSubChannel( d_numSubchannels );

    // for each cell,
    for ( ; cell != cell.end(); ++cell ) {
        auto center = cell->centroid();
        // get the index of the subchannel
        int index = getSubchannelIndex( center[0], center[1] );
        if ( index >= 0 ) {
            d_ownSubChannel[index] = true;
            // put cell into array of cells for that subchannel
            d_elem[index].push_back( cell->clone() );
        }
    } // end for cell

    // for each subchannel,
    for ( size_t isub = 0; isub < d_numSubchannels; ++isub ) {
        if ( !d_ownSubChannel[isub] )
            continue;

        // extract subchannel cells from d_elem[isub]
        auto subchannelElements = std::make_shared<std::vector<ElementPtr>>();
        subchannelElements->reserve( d_numSubchannels );
        for ( const auto &ielem : d_elem[isub] ) {
            subchannelElements->push_back( ielem->clone() );
        }
        auto localSubchannelCell = AMP::Mesh::MeshListIterator(
            subchannelElements ); // iterator over elements of current subchannel
        // get subchannel index
        auto subchannelCentroid = localSubchannelCell->centroid();
        size_t currentSubchannelIndex =
            getSubchannelIndex( subchannelCentroid[0], subchannelCentroid[1] );

        // compute flux
        std::vector<double> flux( d_z.size() - 1 );
        if ( d_source == "averageCladdingTemperature" ) {
            std::shared_ptr<std::vector<ElementPtr>> elemPtr( &d_subchannelFace[isub],
                                                              []( auto ) {} );
            auto localSubchannelFace = AMP::Mesh::MeshListIterator( elemPtr );
            AMP_ASSERT( localSubchannelFace.size() == d_z.size() );
            auto face = localSubchannelFace.begin();
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
            AMP_ASSERT( d_QFraction.size() == d_numSubchannels );
            flux = Subchannel::getHeatFluxGeneration( d_heatShape, d_z, d_rodDiameter[isub], d_Q );
            // multiply by power fraction
            for ( double &i : flux )
                i = i * d_QFraction[isub];
        } else if ( d_source == "totalHeatGenerationWithDiscretizationError" ) {
            AMP_ASSERT( d_QFraction.size() == d_numSubchannels );
            flux = Subchannel::getHeatFluxGenerationWithDiscretizationError(
                d_heatShape, d_z, d_rodDiameter[isub], d_Q );
            // multiply by power fraction
            for ( double &i : flux )
                i = i * d_QFraction[isub];
        } else {
            AMP_ERROR( "Heat source type '" + d_source + "' is invalid" );
        }

        // loop over cells of current subchannel
        double area = d_channelArea[isub]; // subchannel area
        double D    = d_channelDiam[isub]; // subchannel hydraulic diameter
        for ( ; localSubchannelCell != localSubchannelCell.end(); ++localSubchannelCell ) {
            // get upper and lower axial faces of current cell
            ElementPtr plusFace;  // upper axial face for current cell
            ElementPtr minusFace; // lower axial face for current cell
            // get the axial faces of cell
            getAxialFaces( *localSubchannelCell, plusFace, minusFace );

            auto plusFaceCentroid  = plusFace->centroid();
            auto minusFaceCentroid = minusFace->centroid();

            // extract unknowns from solution vector
            // -------------------------------------
            std::vector<size_t> plusDofs;
            std::vector<size_t> minusDofs;
            dof_manager->getDOFs( plusFace->globalID(), plusDofs );
            dof_manager->getDOFs( minusFace->globalID(), minusDofs );
            double m_plus  = m_scale * inputVec->getValueByGlobalID( plusDofs[0] );
            double h_plus  = h_scale * inputVec->getValueByGlobalID( plusDofs[1] );
            double p_plus  = p_scale * inputVec->getValueByGlobalID( plusDofs[2] );
            double m_minus = m_scale * inputVec->getValueByGlobalID( minusDofs[0] );
            double h_minus = h_scale * inputVec->getValueByGlobalID( minusDofs[1] );
            double p_minus = p_scale * inputVec->getValueByGlobalID( minusDofs[2] );

            // compute additional quantities
            // -----------------------------
            double m_mid = 0.5 * ( m_plus + m_minus );
            double p_mid = 0.5 * ( p_plus + p_minus );

            // evaluate specific volume
            double vol_plus  = Volume( h_plus, p_plus );   // upper face
            double vol_minus = Volume( h_minus, p_minus ); // lower face

            // determine axial donor quantities
            double h_axialDonor;
            double vol_axialDonor;
            if ( m_mid >= 0.0 ) {
                vol_axialDonor = vol_minus;
                h_axialDonor   = h_minus;
            } else {
                vol_axialDonor = vol_plus;
                h_axialDonor   = h_minus;
            }

            // evaluate density
            double rho_mid = 1.0 / vol_axialDonor;

            // evaluate axial velocity
            double u_plus  = m_plus * vol_plus / area;
            double u_minus = m_minus * vol_minus / area;
            double u_mid   = m_mid * vol_axialDonor / area;

            // evaluate temperature for cell
            double T_mid = Temperature( h_axialDonor, p_mid );

            // evaluate conductivity for cell
            double k_mid = ThermalConductivity( T_mid, rho_mid );

            // evaluate dynamic viscosity
            double visc_mid = DynamicViscosity( T_mid, rho_mid );

            // compute element height
            double z_minus = minusFaceCentroid[2];
            double z_plus  = plusFaceCentroid[2];
            double dz      = z_plus - z_minus;

            // evaluate friction factor
            double Re_mid = rho_mid * u_mid * D / visc_mid;
            double fl     = 64.0 / Re_mid; // laminar friction factor
            double fric;                   // friction factor
            if ( d_frictionModel == "Constant" ) {
                fric = d_friction;
            } else {
                double ft     = 0.; // turbulent friction factor evaluated from computed Re
                double ft4000 = 0.; // turbulent friction factor evaluated from Re = 4000
                if ( d_frictionModel == "Blasius" ) {
                    ft     = 0.316 * std::pow( Re_mid, -0.25 );
                    ft4000 = 0.316 * std::pow( 4000.0, -0.25 );
                } else if ( d_frictionModel == "Drew" ) {
                    ft     = 0.0056 + 0.5 * std::pow( Re_mid, -0.32 );
                    ft4000 = 0.0056 + 0.5 * std::pow( 4000.0, -0.32 );
                } else if ( d_frictionModel == "Filonenko" ) {
                    ft     = std::pow( 1.82 * std::log( Re_mid ) - 1.64, -2 );
                    ft4000 = std::pow( 1.82 * std::log( 4000.0 ) - 1.64, -2 );
                } else if ( d_frictionModel == "Selander" ) {
                    ft = 4.0 *
                         std::pow( 3.8 * std::log( 10.0 / Re_mid + 0.2 * d_roughness / D ), -2 );
                    ft4000 =
                        4.0 *
                        std::pow( 3.8 * std::log( 10.0 / 4000.0 + 0.2 * d_roughness / D ), -2 );
                } else {
                    AMP_ERROR( "Invalid choice for Friction_Model." );
                }
                if ( Re_mid < 4000.0 )
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

            // initialize sum terms
            double mass_crossflow_sum    = 0.0;
            double energy_crossflow_sum  = 0.0;
            double energy_heatflux_sum   = 0.0;
            double energy_turbulence_sum = 0.0;
            double energy_conduction_sum = 0.0;
            double axial_crossflow_sum   = 0.0;
            double axial_turbulence_sum  = 0.0;

            double zMidCell     = 0.5 * ( plusFaceCentroid[2] + minusFaceCentroid[2] );
            size_t j            = Utilities::findfirst( zMid, zMidCell );
            energy_heatflux_sum = ( 1 + d_gamma ) * pi * d_rodDiameter[isub] * flux[j];

            // loop over gap faces
            auto cellFaces = localSubchannelCell->getElements( AMP::Mesh::GeomType::Face );
            for ( auto &cellFace : cellFaces ) {
                auto faceCentroid        = cellFace.centroid();
                auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
                if ( lateralFaceIterator != interiorLateralFaceMap.end() ) {
                    // get face
                    auto lateralFace = lateralFaceIterator->second->clone();
                    // get crossflow
                    std::vector<size_t> gapDofs;
                    dof_manager->getDOFs( lateralFace->globalID(), gapDofs );
                    double w = w_scale * inputVec->getValueByGlobalID( gapDofs[0] );
                    // get index of neighboring subchannel
                    auto adjacentCells =
                        d_Mesh->getElementParents( *lateralFace, AMP::Mesh::GeomType::Cell );
                    AMP_INSIST( adjacentCells.size() == 2,
                                "There were not 2 adjacent cells to a lateral gap face" );
                    auto subchannelCentroid1 = adjacentCells[0].centroid();
                    auto subchannelCentroid2 = adjacentCells[1].centroid();
                    size_t subchannelIndex1 =
                        getSubchannelIndex( subchannelCentroid1[0], subchannelCentroid1[1] );
                    size_t subchannelIndex2 =
                        getSubchannelIndex( subchannelCentroid2[0], subchannelCentroid2[1] );
                    size_t neighborSubchannelIndex = 0;
                    ElementPtr neighborCell;
                    if ( subchannelIndex1 == currentSubchannelIndex ) {
                        AMP_INSIST( subchannelIndex2 != currentSubchannelIndex,
                                    "Adjacent cells have the same subchannel index." );
                        neighborSubchannelIndex = subchannelIndex2;
                        neighborCell            = adjacentCells[1].clone();
                    } else if ( subchannelIndex2 == currentSubchannelIndex ) {
                        neighborSubchannelIndex = subchannelIndex1;
                        neighborCell            = adjacentCells[0].clone();
                    } else {
                        AMP_ERROR( "Neither of adjacent cells had the same index as the current "
                                   "subchannel." );
                    }
                    // determine sign of crossflow term
                    double crossflowSign = 0.;
                    if ( currentSubchannelIndex < neighborSubchannelIndex ) {
                        crossflowSign = 1.0;
                    } else if ( currentSubchannelIndex > neighborSubchannelIndex ) {
                        crossflowSign = -1.0;
                    } else {
                        AMP_ERROR( "Adjacent cells have the same subchannel index." );
                    }
                    // get upper and lower axial faces of neighbor cell
                    // ------------------------------------------------
                    auto neighborCentroid = neighborCell->centroid();
                    // get axial faces of current cell
                    ElementPtr neighborPlusFace;  // upper axial face for current cell
                    ElementPtr neighborMinusFace; // lower axial face for current cell
                    getAxialFaces( *neighborCell, neighborPlusFace, neighborMinusFace );

                    double neighborArea =
                        d_channelArea[neighborSubchannelIndex]; // neighbor subchannel area
                    double neighborDiam =
                        d_channelDiam[neighborSubchannelIndex]; // neighbor hydraulic diameter

                    // extract unknowns from solution vector
                    // -------------------------------------
                    std::vector<size_t> neighborPlusDofs;
                    std::vector<size_t> neighborMinusDofs;
                    dof_manager->getDOFs( neighborPlusFace->globalID(), neighborPlusDofs );
                    dof_manager->getDOFs( neighborMinusFace->globalID(), neighborMinusDofs );
                    double m_plus_neighbor =
                        m_scale * inputVec->getValueByGlobalID( neighborPlusDofs[0] );
                    double h_plus_neighbor =
                        h_scale * inputVec->getValueByGlobalID( neighborPlusDofs[1] );
                    double p_plus_neighbor =
                        p_scale * inputVec->getValueByGlobalID( neighborPlusDofs[2] );
                    double m_minus_neighbor =
                        m_scale * inputVec->getValueByGlobalID( neighborMinusDofs[0] );
                    double h_minus_neighbor =
                        h_scale * inputVec->getValueByGlobalID( neighborMinusDofs[1] );
                    double p_minus_neighbor =
                        p_scale * inputVec->getValueByGlobalID( neighborMinusDofs[2] );

                    // compute additional quantities from neighboring cell
                    double m_mid_neighbor = 0.5 * ( m_plus_neighbor + m_minus_neighbor );
                    double p_mid_neighbor = 0.5 * ( p_plus_neighbor + p_minus_neighbor );

                    // evaluate specific volume at upper face
                    double vol_plus_neighbor = Volume( h_plus_neighbor, p_plus_neighbor );

                    // evaluate specific volume at lower face
                    double vol_minus_neighbor = Volume( h_minus_neighbor, p_minus_neighbor );

                    double h_axialDonor_neighbor;
                    double vol_axialDonor_neighbor;
                    if ( m_mid_neighbor >= 0.0 ) {
                        h_axialDonor_neighbor   = h_minus_neighbor;
                        vol_axialDonor_neighbor = vol_minus_neighbor;
                    } else {
                        h_axialDonor_neighbor   = h_plus_neighbor;
                        vol_axialDonor_neighbor = vol_plus_neighbor;
                    }

                    double rho_mid_neighbor = 1.0 / vol_axialDonor_neighbor;
                    double u_mid_neighbor = m_mid_neighbor * vol_axialDonor_neighbor / neighborArea;

                    double h_lateralDonor;
                    double u_lateralDonor;
                    if ( crossflowSign * w >= 0.0 ) {
                        h_lateralDonor = h_axialDonor;
                        u_lateralDonor = u_mid;
                    } else {
                        h_lateralDonor = h_axialDonor_neighbor;
                        u_lateralDonor = u_mid_neighbor;
                    }

                    // evaluate temperature for neighbor cell
                    double T_mid_neighbor = Temperature( h_axialDonor_neighbor, p_mid_neighbor );

                    // evaluate conductivity for cell
                    double k_mid_neighbor = ThermalConductivity( T_mid_neighbor, rho_mid_neighbor );

                    // compute thermal conductivity across gap
                    double k_gap_harmonic_avg =
                        2.0 * k_mid * k_mid_neighbor / ( k_mid + k_mid_neighbor );

                    // evaluate dynamic viscosity for cell
                    double visc_mid_neighbor = DynamicViscosity( T_mid_neighbor, rho_mid_neighbor );

                    // compute distance between centroids of cells adjacent to gap
                    auto cellCentroid = localSubchannelCell->centroid();
                    double x_distance = std::abs( neighborCentroid[0] - cellCentroid[0] );
                    double y_distance = std::abs( neighborCentroid[1] - cellCentroid[1] );
                    double pitch =
                        std::pow( std::pow( x_distance, 2 ) + std::pow( y_distance, 2 ), 0.5 );

                    // compute gap width
                    auto lateralFaceCentroid = lateralFace->centroid();
                    AMP::Mesh::Point xyPos( lateralFaceCentroid[0], lateralFaceCentroid[1] );
                    auto gapWidthIt = gapWidthMap.find( xyPos );
                    AMP_INSIST( gapWidthIt != gapWidthMap.end(), "Gap was not found." );
                    double gapWidth = gapWidthIt->second;

                    double conductance = 1.0 * k_gap_harmonic_avg / pitch;

                    // compute turbulent crossflow
                    double D_gap_avg = 0.5 * ( D + neighborDiam );
                    double Re_mid_neighbor =
                        rho_mid_neighbor * u_mid_neighbor * neighborDiam / visc_mid_neighbor;
                    double Re_gap_avg = 0.5 * ( Re_mid + Re_mid_neighbor );
                    double rodDiameter =
                        0.5 * ( d_rodDiameter[isub] + d_rodDiameter[neighborSubchannelIndex] );
                    double beta = 0.005 * D_gap_avg / gapWidth *
                                  std::pow( gapWidth / rodDiameter, 0.106 ) *
                                  std::pow( Re_gap_avg, -0.1 );
                    double massFlux_gap_avg =
                        0.5 * ( m_mid / area + m_mid_neighbor / neighborArea );
                    double wt = beta * gapWidth * massFlux_gap_avg;

                    // add to sums
                    mass_crossflow_sum += crossflowSign * w;
                    energy_crossflow_sum += crossflowSign * w * h_lateralDonor;
                    energy_turbulence_sum += wt * ( h_axialDonor - h_axialDonor_neighbor );
                    energy_conduction_sum += conductance * gapWidth * ( T_mid - T_mid_neighbor );
                    axial_crossflow_sum += crossflowSign * w * u_lateralDonor;
                    axial_turbulence_sum += wt * ( u_mid - u_mid_neighbor );

                } // end if (lateralFaceIterator != interiorLateralFaceMap.end()) {
            }     // end loop over gap faces

            // force terms to zero if requested
            double force_factor_conduction  = 1.0;
            double force_factor_turbulence  = 1.0;
            double force_factor_heat_source = 1.0;
            double force_factor_friction    = 1.0;
            if ( d_forceNoConduction )
                force_factor_conduction = 0.0;
            if ( d_forceNoTurbulence )
                force_factor_turbulence = 0.0;
            if ( d_forceNoHeatSource )
                force_factor_heat_source = 0.0;
            if ( d_forceNoFriction )
                force_factor_friction = 0.0;

            // calculate residuals for current cell
            // ------------------------------------
            // mass
            double R_m = m_plus - m_minus + dz * mass_crossflow_sum;
            // energy
            double R_h = m_plus * h_plus - m_minus * h_minus + dz * energy_crossflow_sum -
                         dz * energy_heatflux_sum * force_factor_heat_source +
                         dz * energy_turbulence_sum * force_factor_turbulence +
                         dz * energy_conduction_sum * force_factor_conduction;
            // axial momentum
            double R_p = m_plus * u_plus - m_minus * u_minus + dz * axial_crossflow_sum +
                         area * ( p_plus - p_minus ) +
                         gravity * area * dz * std::cos( d_theta ) / vol_axialDonor +
                         1.0 / ( 2.0 * area ) * ( dz * fric / D + K ) * std::abs( m_mid ) * m_mid *
                             vol_axialDonor * force_factor_friction +
                         d_turbulenceCoef * dz * axial_turbulence_sum * force_factor_turbulence;

            // put residuals into global residual vector
            double val = Subchannel::scaleAxialMassFlowRate * R_m;
            outputVec->setValuesByGlobalID( 1, &plusDofs[0], &val );
            val = Subchannel::scaleEnthalpy * R_h;
            outputVec->setValuesByGlobalID( 1, &plusDofs[1], &val );
            val = Subchannel::scalePressure * R_p;
            outputVec->setValuesByGlobalID( 1, &minusDofs[2], &val );

            // impose boundary conditions
            // --------------------------
            // if face is exit face,
            if ( AMP::Utilities::approx_equal( plusFaceCentroid[2], height ) ) {
                // impose fixed exit pressure boundary condition
                val = Subchannel::scalePressure * ( p_plus - d_Pout );
                outputVec->setValuesByGlobalID( 1, &plusDofs[2], &val );
            }
            if ( AMP::Utilities::approx_equal( minusFaceCentroid[2], 0.0 ) ) {
                // impose fixed inlet axial mass flow rate boundary condition
                val = Subchannel::scaleAxialMassFlowRate * ( m_minus - d_channelMass[isub] );
                outputVec->setValuesByGlobalID( 1, &minusDofs[0], &val );

                // evaluate enthalpy at inlet
                double h_eval = Enthalpy( d_Tin, p_minus );
                // impose fixed inlet temperature boundary condition
                val = Subchannel::scaleEnthalpy * ( h_minus - h_eval );
                outputVec->setValuesByGlobalID( 1, &minusDofs[1], &val );
            }
        } // end loop over cells of current subchannel
    }     // end loop over subchannels

    // loop over lateral faces
    auto face = d_Mesh->getIterator( AMP::Mesh::GeomType::Face, 0 ); // iterator for cells of mesh
    for ( ; face != face.end(); ++face ) {
        auto faceCentroid        = face->centroid();
        auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
        if ( lateralFaceIterator != interiorLateralFaceMap.end() ) {
            // get lateral face
            ElementPtr lateralFace = lateralFaceIterator->second->clone();
            // get crossflow from solution vector
            std::vector<size_t> gapDofs;
            dof_manager->getDOFs( lateralFace->globalID(), gapDofs );
            double w_mid = w_scale * inputVec->getValueByGlobalID( gapDofs[0] );

            // get adjacent cells
            auto adjacentCells =
                d_Mesh->getElementParents( *lateralFace, AMP::Mesh::GeomType::Cell );
            AMP_INSIST( adjacentCells.size() == 2,
                        "There were not 2 adjacent cells to a lateral gap face" );
            auto &cell1        = adjacentCells[0];
            auto &cell2        = adjacentCells[1];
            auto cell1Centroid = cell1.centroid();
            auto cell2Centroid = cell2.centroid();
            // get upper and lower axial faces
            ElementPtr cell1PlusFace;  // upper axial face for cell 1
            ElementPtr cell1MinusFace; // lower axial face for cell 1
            ElementPtr cell2PlusFace;  // upper axial face for cell 2
            ElementPtr cell2MinusFace; // lower axial face for cell 2
            getAxialFaces( cell1, cell1PlusFace, cell1MinusFace );
            getAxialFaces( cell2, cell2PlusFace, cell2MinusFace );

            // determine if cells are in the first axial interval
            auto cell1MinusFaceCentroid = cell1MinusFace->centroid();
            auto cell1PlusFaceCentroid  = cell1PlusFace->centroid();
            auto cell2MinusFaceCentroid = cell2MinusFace->centroid();
            // if bottom face is at z = 0,
            if ( AMP::Utilities::approx_equal( cell1MinusFaceCentroid[2], 0.0 ) ) {
                // implement fixed lateral mass flow rates inlet boundary condition
                const double val = Subchannel::scaleLateralMassFlowRate * w_mid;
                outputVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
            } else {
                // get cells below bottom faces
                // get adjacent cells
                auto cell1MinusFaceAdjacentCells =
                    d_Mesh->getElementParents( *cell1MinusFace, AMP::Mesh::GeomType::Cell );
                auto cell2MinusFaceAdjacentCells =
                    d_Mesh->getElementParents( *cell2MinusFace, AMP::Mesh::GeomType::Cell );
                AMP_INSIST( cell1MinusFaceAdjacentCells.size() == 2,
                            "There were not 2 adjacent cells to an axial face" );
                AMP_INSIST( cell2MinusFaceAdjacentCells.size() == 2,
                            "There were not 2 adjacent cells to an axial face" );
                auto &axialCell11        = cell1MinusFaceAdjacentCells[0];
                auto &axialCell12        = cell1MinusFaceAdjacentCells[1];
                auto &axialCell21        = cell2MinusFaceAdjacentCells[0];
                auto &axialCell22        = cell2MinusFaceAdjacentCells[1];
                auto axialCell11Centroid = axialCell11.centroid();
                auto axialCell12Centroid = axialCell12.centroid();
                auto axialCell21Centroid = axialCell21.centroid();
                auto axialCell22Centroid = axialCell22.centroid();
                // determine which cell is bottom cell
                const AMP::Mesh::MeshElement *bottomCell1 = nullptr;
                const AMP::Mesh::MeshElement *bottomCell2 = nullptr;
                if ( axialCell11Centroid[2] < cell1MinusFaceCentroid[2] ) {
                    // axialCell11 is bottom cell
                    // ensure that axialCell12 is above
                    AMP_INSIST( axialCell12Centroid[2] > cell1MinusFaceCentroid[2],
                                "Both adjacent cells are below axial face parent" );
                    bottomCell1 = &axialCell11;
                } else {
                    // axialCell12 is bottom cell
                    // ensure that axialCell11 is above
                    AMP_INSIST( axialCell11Centroid[2] > cell1MinusFaceCentroid[2],
                                "Both adjacent cells are below axial face parent" );
                    bottomCell1 = &axialCell12;
                }
                if ( axialCell21Centroid[2] < cell2MinusFaceCentroid[2] ) {
                    // axialCell21 is bottom cell
                    // ensure that axialCell22 is above
                    AMP_INSIST( axialCell22Centroid[2] > cell2MinusFaceCentroid[2],
                                "Both adjacent cells are below axial face parent" );
                    bottomCell2 = &axialCell21;
                } else {
                    // axialCell22 is bottom cell
                    // ensure that axialCell21 is above
                    AMP_INSIST( axialCell21Centroid[2] > cell2MinusFaceCentroid[2],
                                "Both adjacent cells are below axial face parent" );
                    bottomCell2 = &axialCell22;
                }
                auto belowLateralFace = getAxiallyAdjacentLateralFace(
                    bottomCell1, *lateralFace, interiorLateralFaceMap );
                std::vector<size_t> belowDofs;
                dof_manager->getDOFs( belowLateralFace->globalID(), belowDofs );
                double w_minus = w_scale * inputVec->getValueByGlobalID( belowDofs[0] );

                // get axial faces of bottom cells
                ElementPtr bottomCell1PlusFace;
                ElementPtr bottomCell1MinusFace;
                ElementPtr bottomCell2PlusFace;
                ElementPtr bottomCell2MinusFace;
                getAxialFaces( *bottomCell1, bottomCell1PlusFace, bottomCell1MinusFace );
                getAxialFaces( *bottomCell2, bottomCell2PlusFace, bottomCell2MinusFace );
                // get unknowns from axial faces of bottom cells
                std::vector<size_t> dofs;
                dof_manager->getDOFs( bottomCell1PlusFace->globalID(), dofs );
                double m1_bottomPlus = m_scale * inputVec->getValueByGlobalID( dofs[0] );
                dof_manager->getDOFs( bottomCell1MinusFace->globalID(), dofs );
                double m1_bottomMinus = m_scale * inputVec->getValueByGlobalID( dofs[0] );
                dof_manager->getDOFs( bottomCell2PlusFace->globalID(), dofs );
                double m2_bottomPlus = m_scale * inputVec->getValueByGlobalID( dofs[0] );
                dof_manager->getDOFs( bottomCell2MinusFace->globalID(), dofs );
                double m2_bottomMinus = m_scale * inputVec->getValueByGlobalID( dofs[0] );

                double m1_bottomMid = 0.5 * ( m1_bottomPlus + m1_bottomMinus );
                double m2_bottomMid = 0.5 * ( m2_bottomPlus + m2_bottomMinus );
                double m_bottomMid  = 0.5 * ( m1_bottomMid + m2_bottomMid );

                std::vector<size_t> cell1PlusDofs;
                std::vector<size_t> cell2PlusDofs;
                std::vector<size_t> cell1MinusDofs;
                std::vector<size_t> cell2MinusDofs;
                dof_manager->getDOFs( cell1PlusFace->globalID(), cell1PlusDofs );
                dof_manager->getDOFs( cell2PlusFace->globalID(), cell2PlusDofs );
                dof_manager->getDOFs( cell1MinusFace->globalID(), cell1MinusDofs );
                dof_manager->getDOFs( cell2MinusFace->globalID(), cell2MinusDofs );

                double m1_plus   = m_scale * inputVec->getValueByGlobalID( cell1PlusDofs[0] );
                double m2_plus   = m_scale * inputVec->getValueByGlobalID( cell2PlusDofs[0] );
                double m1_minus  = m_scale * inputVec->getValueByGlobalID( cell1MinusDofs[0] );
                double m2_minus  = m_scale * inputVec->getValueByGlobalID( cell2MinusDofs[0] );
                double m1_mid    = 0.5 * ( m1_plus + m1_minus );
                double m2_mid    = 0.5 * ( m2_plus + m2_minus );
                double m_gap_avg = 0.5 * ( m1_mid + m2_mid );

                double h1_plus  = h_scale * inputVec->getValueByGlobalID( cell1PlusDofs[1] );
                double h2_plus  = h_scale * inputVec->getValueByGlobalID( cell2PlusDofs[1] );
                double h1_minus = h_scale * inputVec->getValueByGlobalID( cell1MinusDofs[1] );
                double h2_minus = h_scale * inputVec->getValueByGlobalID( cell2MinusDofs[1] );

                double p1_plus  = p_scale * inputVec->getValueByGlobalID( cell1PlusDofs[2] );
                double p2_plus  = p_scale * inputVec->getValueByGlobalID( cell2PlusDofs[2] );
                double p1_minus = p_scale * inputVec->getValueByGlobalID( cell1MinusDofs[2] );
                double p2_minus = p_scale * inputVec->getValueByGlobalID( cell2MinusDofs[2] );

                double vol1_plus  = Volume( h1_plus, p1_plus );
                double vol1_minus = Volume( h1_minus, p1_minus );
                double vol1_axialDonor;
                if ( m1_mid >= 0.0 )
                    vol1_axialDonor = vol1_minus;
                else
                    vol1_axialDonor = vol1_plus;

                double vol2_plus  = Volume( h2_plus, p2_plus );
                double vol2_minus = Volume( h2_minus, p2_minus );
                double vol2_axialDonor;
                if ( m2_mid >= 0.0 )
                    vol2_axialDonor = vol2_minus;
                else
                    vol2_axialDonor = vol2_plus;

                double vol_gap_avg = 0.5 * ( vol1_axialDonor + vol2_axialDonor );

                size_t isubCell1 = getSubchannelIndex( cell1Centroid[0], cell1Centroid[1] );
                size_t isubCell2 = getSubchannelIndex( cell2Centroid[0], cell2Centroid[1] );
                double crossflowSign;
                if ( isubCell1 < isubCell2 )
                    crossflowSign = 1.0;
                else
                    crossflowSign = -1.0;
                double area1    = d_channelArea[isubCell1];
                double area2    = d_channelArea[isubCell2];
                double u1_plus  = m1_plus * vol1_plus / area1;
                double u1_minus = m1_minus * vol1_minus / area1;
                double u2_plus  = m2_plus * vol2_plus / area2;
                double u2_minus = m2_minus * vol2_minus / area2;
                double u_plus   = 0.5 * ( u1_plus + u2_plus );
                double u_minus  = 0.5 * ( u1_minus + u2_minus );

                double w_axialDonor_plus;
                double w_axialDonor_minus;
                if ( m_gap_avg >= 0.0 )
                    w_axialDonor_plus = w_mid;
                else {
                    if ( AMP::Utilities::approx_equal(
                             cell1PlusFaceCentroid[2], height, 1.0e-6 ) ) {
                        // cell is in the uppermost axial interval
                        w_axialDonor_plus = w_mid;
                    } else {
                        auto cell1PlusFaceAdjacentCells =
                            d_Mesh->getElementParents( *cell1PlusFace, AMP::Mesh::GeomType::Cell );
                        AMP_INSIST( cell1PlusFaceAdjacentCells.size() == 2,
                                    "There were not 2 adjacent cells to an axial gap face" );
                        auto &axialCell1        = cell1PlusFaceAdjacentCells[0];
                        auto &axialCell2        = cell1PlusFaceAdjacentCells[1];
                        auto axialCell1Centroid = axialCell1.centroid();
                        auto axialCell2Centroid = axialCell2.centroid();
                        // determine which cell is top cell
                        const AMP::Mesh::MeshElement *topCell = nullptr;
                        if ( axialCell1Centroid[2] > cell1PlusFaceCentroid[2] ) {
                            // axialCell1 is top cell
                            // ensure that axialCell2 is above
                            AMP_INSIST( axialCell2Centroid[2] < cell1PlusFaceCentroid[2],
                                        "Both adjacent cells are above axial face parent" );
                            topCell = &axialCell1;
                        } else {
                            // axialCell2 is top cell
                            // ensure that axialCell1 is above
                            AMP_INSIST( axialCell1Centroid[2] < cell1PlusFaceCentroid[2],
                                        "Both adjacent cells are above axial face parent" );
                            topCell = &axialCell2;
                        }
                        auto aboveLateralFace = getAxiallyAdjacentLateralFace(
                            topCell, *lateralFace, interiorLateralFaceMap );
                        std::vector<size_t> aboveDofs;
                        dof_manager->getDOFs( aboveLateralFace->globalID(), aboveDofs );
                        double w_plus     = w_scale * inputVec->getValueByGlobalID( aboveDofs[0] );
                        w_axialDonor_plus = w_plus;
                    }
                }
                if ( m_bottomMid >= 0.0 )
                    w_axialDonor_minus = w_minus;
                else
                    w_axialDonor_minus = w_mid;

                // compute distance between centroids of cells adjacent to gap
                double x_distance = std::abs( cell1Centroid[0] - cell2Centroid[0] );
                double y_distance = std::abs( cell1Centroid[1] - cell2Centroid[1] );
                double pitch =
                    std::pow( std::pow( x_distance, 2 ) + std::pow( y_distance, 2 ), 0.5 );

                // compute gap width
                auto lateralFaceCentroid = lateralFace->centroid();
                AMP::Mesh::Point xyPos( lateralFaceCentroid[0], lateralFaceCentroid[1] );
                auto gapWidthIt = gapWidthMap.find( xyPos );
                AMP_INSIST( gapWidthIt != gapWidthMap.end(), "Gap was not found." );
                double gapWidth = gapWidthIt->second;

                // compute element height
                double dz = cell1PlusFaceCentroid[2] - cell1MinusFaceCentroid[2];

                double R_w = u_plus * w_axialDonor_plus - u_minus * w_axialDonor_minus -
                             crossflowSign * gapWidth / pitch * dz * ( p1_minus - p2_minus ) +
                             dz * d_KG / ( 2.0 * gapWidth * pitch ) * std::abs( w_mid ) * w_mid *
                                 vol_gap_avg +
                             gapWidth * pitch * dz * gravity * std::sin( d_theta ) / vol_gap_avg;
                const double val = Subchannel::scaleLateralMassFlowRate * R_w;
                outputVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
            }
        } else {
            // determine if face is an external gap face; in this case, a zero must by set
            // to the nonlinear residual entry for this DOF, i.e., a zero Dirichlet BC
            // is applied to exterior gap face lateral mass flow rates
            lateralFaceIterator = exteriorLateralFaceMap.find( faceCentroid );
            if ( lateralFaceIterator != exteriorLateralFaceMap.end() ) {
                // get lateral face
                auto &lateralFace = lateralFaceIterator->second;
                std::vector<size_t> gapDofs;
                dof_manager->getDOFs( lateralFace->globalID(), gapDofs );
                const double val = 0.0;
                outputVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
            }
        }
    } // end loop over lateral faces
    outputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
} // end of apply function

std::shared_ptr<OperatorParameters> SubchannelFourEqNonlinearOperator::getJacobianParameters(
    AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{
    std::shared_ptr<AMP::Database> tmp_db = d_params->d_db->cloneDatabase();

    tmp_db->putScalar( "name", "SubchannelFourEqLinearOperator" );
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


// Create the VectorSelector, the vectors are simple vectors and
//    we need to subset for the current comm instead of the mesh
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
SubchannelFourEqNonlinearOperator::selectOutputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Comm>( d_Mesh->getComm() ) );
    auto var = getOutputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
SubchannelFourEqNonlinearOperator::selectInputVector() const
{
    std::vector<std::shared_ptr<AMP::LinearAlgebra::VectorSelector>> selectors;
    if ( d_Mesh )
        selectors.push_back( std::make_shared<AMP::LinearAlgebra::VS_Comm>( d_Mesh->getComm() ) );
    auto var = getInputVariable();
    if ( var )
        selectors.push_back( var->createVectorSelector() );
    return AMP::LinearAlgebra::VectorSelector::create( selectors );
}


double SubchannelFourEqNonlinearOperator::Volume( double h, double p )
{
    // evaluates specific volume
    // h: enthalpy
    // p: pressure
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    argMap.insert( std::make_pair( std::string( "enthalpy" ),
                                   std::make_shared<std::vector<double>>( 1, h ) ) );
    argMap.insert( std::make_pair( std::string( "pressure" ),
                                   std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> result( 1 );
    d_subchannelPhysicsModel->getProperty( "SpecificVolume", result, argMap );
    return result[0];
}

double SubchannelFourEqNonlinearOperator::Temperature( double h, double p )
{
    // evaluates temperature
    // h: enthalpy
    // p: pressure
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    argMap.insert( std::make_pair( std::string( "enthalpy" ),
                                   std::make_shared<std::vector<double>>( 1, h ) ) );
    argMap.insert( std::make_pair( std::string( "pressure" ),
                                   std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> result( 1 );
    d_subchannelPhysicsModel->getProperty( "Temperature", result, argMap );
    return result[0];
}

double SubchannelFourEqNonlinearOperator::ThermalConductivity( double T, double rho )
{
    // evaluates thermal conductivity
    // T: temperature
    // rho: density
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    argMap.insert( std::make_pair( std::string( "temperature" ),
                                   std::make_shared<std::vector<double>>( 1, T ) ) );
    argMap.insert( std::make_pair( std::string( "density" ),
                                   std::make_shared<std::vector<double>>( 1, rho ) ) );
    std::vector<double> result( 1 );
    d_subchannelPhysicsModel->getProperty( "ThermalConductivity", result, argMap );
    return result[0];
}

double SubchannelFourEqNonlinearOperator::DynamicViscosity( double T, double rho )
{
    // evaluates dynamic viscosity
    // T: temperature
    // rho: density
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    argMap.insert( std::make_pair( std::string( "temperature" ),
                                   std::make_shared<std::vector<double>>( 1, T ) ) );
    argMap.insert( std::make_pair( std::string( "density" ),
                                   std::make_shared<std::vector<double>>( 1, rho ) ) );
    std::vector<double> result( 1 );
    d_subchannelPhysicsModel->getProperty( "DynamicViscosity", result, argMap );
    return result[0];
}

double SubchannelFourEqNonlinearOperator::Enthalpy( double T, double p )
{
    // evaluates specific enthalpy
    // T: temperature
    // p: pressure
    std::map<std::string, std::shared_ptr<std::vector<double>>> argMap;
    argMap.insert( std::make_pair( std::string( "temperature" ),
                                   std::make_shared<std::vector<double>>( 1, T ) ) );
    argMap.insert( std::make_pair( std::string( "pressure" ),
                                   std::make_shared<std::vector<double>>( 1, p ) ) );
    std::vector<double> result( 1 );
    d_subchannelPhysicsModel->getProperty( "Enthalpy", result, argMap );
    return result[0];
}

void SubchannelFourEqNonlinearOperator::getAxialFaces( const AMP::Mesh::MeshElement &cell,
                                                       ElementPtr &upperFace,
                                                       ElementPtr &lowerFace )
{
    // gets upper and lower faces of a cell
    bool upperFaceFound = false;
    bool lowerFaceFound = false;
    auto cellCentroid   = cell.centroid();
    // get all faces of cell
    auto cellFaces = cell.getElements( AMP::Mesh::GeomType::Face );
    // loop over faces of cell
    for ( auto &cellFace : cellFaces ) {
        auto faceCentroid = cellFace.centroid();
        // if z-coordinates of centroids of the cell and face are not equal,
        if ( !AMP::Utilities::approx_equal( faceCentroid[2], cellCentroid[2], 1.0e-6 ) ) {
            // if face is above cell centroid,
            if ( faceCentroid[2] > cellCentroid[2] )
                // face is the upper axial face
                if ( upperFaceFound )
                    AMP_ERROR( "Two upper axial faces were found for the same cell." );
                else {
                    upperFace      = cellFace.clone();
                    upperFaceFound = true;
                }
            else
                // face is the lower axial face
                if ( lowerFaceFound )
                    AMP_ERROR( "Two lower axial faces were found for the same cell." );
                else {
                    lowerFace      = cellFace.clone();
                    lowerFaceFound = true;
                }
        }
    } // end loop over faces of cell
    if ( !( upperFaceFound && lowerFaceFound ) )
        AMP_ERROR( "One or both axial faces of a cell were not found." );
}

std::unique_ptr<AMP::Mesh::MeshElement>
SubchannelFourEqNonlinearOperator::getAxiallyAdjacentLateralFace(
    const AMP::Mesh::MeshElement *daughterCell,
    const AMP::Mesh::MeshElement &parentLateralFace,
    const std::map<AMP::Mesh::Point, ElementPtr> &interiorLateralFaceMap )
{
    // gets the lateral face that is either below or above another lateral face
    // daughterCell: cell that is either above or below the parent cell
    // parentLateralFace: lateral face belonging to the parent cell for which the function seeks to
    // find an axially
    // adjacent lateral face

    // has the axially adjacent lateral face been found?
    bool axiallyAdjacentLateralFaceFound = false;
    // get centroid of parent lateral face
    auto parentLateralFaceCentroid = parentLateralFace.centroid();
    auto daughterCellCentroid      = daughterCell->centroid();
    // loop over faces of daughter cell to determine which face is the lateral face axially adjacent
    // to the current
    // lateral face
    ElementPtr axiallyAdjacentLateralFace;
    auto daughterCellFaces = daughterCell->getElements( AMP::Mesh::GeomType::Face );
    for ( auto &daughterCellFace : daughterCellFaces ) {
        auto faceCentroid        = daughterCellFace.centroid();
        auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
        if ( lateralFaceIterator != interiorLateralFaceMap.end() ) {
            // get lateral face
            auto daughterLateralFace         = lateralFaceIterator->second->clone();
            auto daughterLateralFaceCentroid = daughterLateralFace->centroid();
            // loop through coordinates to determine if lateral face is the lateral face axially
            // adjacent to the current
            // lateral face
            double knownCentroid[3]           = { parentLateralFaceCentroid[0],
                                        parentLateralFaceCentroid[1],
                                        daughterCellCentroid[2] };
            bool isAxiallyAdjacentLateralFace = true;
            for ( size_t i = 0; i < 3; i++ ) {
                if ( !AMP::Utilities::approx_equal(
                         daughterLateralFaceCentroid[i], knownCentroid[i], 1.0e-6 ) )
                    isAxiallyAdjacentLateralFace = false;
            }
            if ( isAxiallyAdjacentLateralFace ) {
                axiallyAdjacentLateralFace = daughterLateralFace->clone();
                if ( axiallyAdjacentLateralFaceFound )
                    AMP_ERROR( "Found multiple axially adjacent lateral faces." );
                else
                    axiallyAdjacentLateralFaceFound = true;
            }
        }
    } // end loop over faces of daughter cell
    AMP_INSIST( axiallyAdjacentLateralFaceFound, "Axially adjacent lateral face was not found." );
    return axiallyAdjacentLateralFace;
}


} // namespace AMP::Operator
