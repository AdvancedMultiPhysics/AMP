#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include <array>
#include <cmath>

namespace AMP::Operator::Subchannel {


// Convert the operator parameters
std::shared_ptr<const SubchannelOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const SubchannelOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const SubchannelOperatorParameters>( params );
    auto subchannelParams    = std::make_shared<SubchannelOperatorParameters>( params->d_db );
    subchannelParams->d_Mesh = params->d_Mesh;
    if ( params->d_db->keyExists( "LocalModel" ) ) {
        auto model_db = params->d_db->getDatabase( "LocalModel" );
        auto model    = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
        subchannelParams->d_subchannelPhysicsModel =
            std::dynamic_pointer_cast<SubchannelPhysicsModel>( model );
    }
    return subchannelParams;
}


// Get the number of subchannels from the mesh
size_t getNumberOfSubchannels( std::shared_ptr<AMP::Mesh::Mesh> subchannel )
{
    std::vector<double> x, y, z;
    AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( subchannel, x, y, z );
    AMP_ASSERT( x.size() >= 2 && y.size() >= 2 );
    size_t Nx = x.size() - 1;
    size_t Ny = y.size() - 1;
    return Nx * Ny;
}


// Subset the subchannel mesh for a particular subchannel
std::shared_ptr<AMP::Mesh::Mesh>
subsetForSubchannel( std::shared_ptr<AMP::Mesh::Mesh> subchannel, size_t i, size_t j )
{
    // Get the coordinates of the subchannel mesh
    std::vector<double> x, y, z;
    AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( subchannel, x, y, z );
    AMP_ASSERT( x.size() >= 2 && y.size() >= 2 );
    size_t Nx = x.size() - 1;
    size_t Ny = y.size() - 1;
    // Get the elements in the subchannel of interest
    auto el       = subchannel->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto elements = std::make_shared<std::vector<AMP::Mesh::MeshElement>>();
    elements->reserve( el.size() / ( Nx * Ny ) );
    for ( size_t k = 0; k < el.size(); ++k, ++el ) {
        auto coord = el->centroid();
        if ( coord[0] >= x[i] && coord[0] <= x[i + 1] && coord[1] >= y[j] && coord[1] <= y[j + 1] )
            elements->push_back( *el );
    }
    return subchannel->Subset( AMP::Mesh::MeshElementVectorIterator( elements ) );
}


// Compute basic properties from the subchannel mesh
void getSubchannelProperties( std::shared_ptr<AMP::Mesh::Mesh> subchannel,
                              const std::vector<double> &clad_x,
                              const std::vector<double> &clad_y,
                              const std::vector<double> &clad_d,
                              std::vector<double> &x,
                              std::vector<double> &y,
                              std::vector<double> &area,
                              std::vector<double> &hydraulic_diam,
                              std::vector<double> &rod_diameter,
                              std::vector<double> &channel_fraction )
{
    const double pi = 3.1415926535897932;
    AMP_MPI comm    = subchannel->getComm();
    // First get the x-y-z coordinates of the subchannel mesh
    std::vector<double> z;
    AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( subchannel, x, y, z );
    AMP_ASSERT( x.size() >= 2 && y.size() >= 2 );
    size_t Nx = x.size() - 1;
    size_t Ny = y.size() - 1;
    // Check the clad properties
    AMP_ASSERT( clad_x.size() == clad_y.size() && clad_x.size() == clad_d.size() );
    AMP_ASSERT( clad_x.size() == comm.maxReduce( clad_x.size() ) );
    AMP_ASSERT( clad_x.size() > 0 );
    // For each subchannel, get the area, wetted perimeter, and the hydraulic diameter
    size_t N_subchannels = ( x.size() - 1 ) * ( y.size() - 1 );
    area                 = std::vector<double>( N_subchannels, 0.0 );
    hydraulic_diam       = std::vector<double>( N_subchannels, 0.0 );
    rod_diameter         = std::vector<double>( N_subchannels, 0.0 );
    channel_fraction     = std::vector<double>( N_subchannels, 0.0 );
    std::vector<double> perimeter1( N_subchannels, 0.0 ); // Clad only perimeter
    std::vector<double> perimeter2( N_subchannels, 0.0 ); // Clad + water perimeter
    // Get the area of the subchannel without the clad
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        size_t ix     = i % ( x.size() - 1 );
        size_t iy     = i / ( x.size() - 1 );
        area[i]       = ( x[ix + 1] - x[ix] ) * ( y[iy + 1] - y[iy] );
        perimeter2[i] = 2 * ( x[ix + 1] - x[ix] ) + 2 * ( y[iy + 1] - y[iy] );
    }
    // Add the area and perimeter corrections of the clad (assuming no clads overlap)
    const double TOL = 1e-12;
    for ( size_t k = 0; k < clad_x.size(); k++ ) {
        double xc      = clad_x[k];
        double yc      = clad_y[k];
        double dc      = clad_d[k];
        size_t index_x = AMP::Utilities::findfirst( x, xc - TOL );
        size_t index_y = AMP::Utilities::findfirst( y, yc - TOL );
        if ( index_x == x.size() ) {
            index_x--;
        }
        if ( index_y == y.size() ) {
            index_y--;
        }
        if ( fabs( x[index_x] - xc ) <= TOL && fabs( y[index_y] - yc ) <= TOL ) {
            // The clad is located at the subchannel boundaries
            double dA  = 0.25 * pi * 0.25 * dc * dc;
            double dP1 = 0.25 * pi * dc;
            double dP2 = ( 1.0 - 0.25 * pi ) * dc;
            size_t i[4];
            for ( auto &elem : i )
                elem = static_cast<size_t>( -1 );
            if ( index_x > 0 && index_y > 0 )
                i[0] = index_x - 1 + ( index_y - 1 ) * Nx;
            if ( index_x < Nx && index_y > 0 )
                i[1] = index_x + ( index_y - 1 ) * Nx;
            if ( index_x > 0 && index_y < Ny )
                i[2] = index_x - 1 + index_y * Nx;
            if ( index_x < Nx && index_y < Ny )
                i[3] = index_x + index_y * Nx;
            for ( auto &elem : i ) {
                if ( elem == static_cast<size_t>( -1 ) )
                    continue;
                area[elem] -= dA;
                perimeter1[elem] += dP1;
                perimeter2[elem] -= dP2;
                double ratio = 1.0 / ( channel_fraction[elem] + 1.0 );
                channel_fraction[elem] += 0.25;
                rod_diameter[elem] = ( 1.0 - ratio ) * rod_diameter[elem] + ratio * dc;
            }
        } else {
            if ( index_x == Nx ) {
                index_x--;
            }
            if ( index_y == Nx ) {
                index_y--;
            }
            if ( ( xc - 0.5 * dc ) > x[index_x] && ( xc + 0.5 * dc ) < x[index_x + 1] &&
                 ( yc - 0.5 * dc ) > y[index_y] && ( yc + 0.5 * dc ) < y[index_y + 1] ) {
                // The clad inside the subchannel
                size_t i = index_x + index_y * Nx;
                area[i] -= 0.25 * pi * dc * dc;
                perimeter1[i] += pi * dc;
                perimeter2[i] += pi * dc;
                double R = 1.0 / ( channel_fraction[i] + 1.0 );
                channel_fraction[i] += 1.0;
                rod_diameter[i] = ( 1.0 - R ) * rod_diameter[i] + R * dc;
            } else {
                AMP_ERROR( "General case not handled yet\n" );
            }
        }
    }
    // Compute the hydraulic diameter
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        hydraulic_diam[i] = 4.0 * area[i] / perimeter1[i];
    }
}


// Function to get the clad properties
void getCladProperties( AMP::AMP_MPI comm,
                        std::shared_ptr<AMP::Mesh::Mesh> clad,
                        std::vector<double> &x,
                        std::vector<double> &y,
                        std::vector<double> &diam )
{
    // Get the center of each local clad
    std::set<std::array<double, 3>> center;
    if ( clad ) {
        AMP_ASSERT( clad->getComm() <= comm );
        auto ids = clad->getLocalBaseMeshIDs();
        for ( auto &id : ids ) {
            auto mesh = clad->Subset( id );
            auto box  = mesh->getBoundingBox();
            std::array<double, 3> tmp;
            tmp[0] = 0.5 * ( box[0] + box[1] );
            tmp[1] = 0.5 * ( box[2] + box[3] );
            tmp[2] = std::max( std::max( tmp[0] - box[0], box[1] - tmp[0] ),
                               std::max( tmp[1] - box[2], box[3] - tmp[1] ) );
            center.insert( tmp );
        }
    }
    // Get the global set and check that there are no duplicates
    comm.setGather( center );
    std::vector<std::array<double, 3>> center2( center.begin(), center.end() );
    for ( size_t i = 0; i < center2.size(); i++ ) {
        for ( size_t j = i + 1; j < center2.size(); j++ ) {
            if ( AMP::Utilities::approx_equal( center2[i][0], center2[j][0] ) &&
                 AMP::Utilities::approx_equal( center2[i][1], center2[j][1] ) ) {
                AMP_ERROR( "Duplicate clads detected" );
            }
        }
    }
    x.resize( center2.size() );
    y.resize( center2.size() );
    diam.resize( center2.size() );
    for ( size_t i = 0; i < center2.size(); i++ ) {
        x[i]    = std::get<0>( center2[i] );
        y[i]    = std::get<1>( center2[i] );
        diam[i] = 2.0 * std::get<2>( center2[i] );
    }
}


// Compute the heat flux for the subchannel assuming a heat generation rate
std::vector<double>
getHeatFluxGeneration( std::string heatShape, std::vector<double> z, double diameter, double Q_tot )
{
    for ( size_t i = 1; i < z.size(); i++ )
        AMP_ASSERT( z[i] > z[i - 1] );
    double height = z.back() - z.front();
    std::vector<double> dz( z.size() - 1, 0.0 );
    for ( size_t i = 0; i < dz.size(); i++ )
        dz[i] = z[i + 1] - z[i];
    const double pi = 3.1415926535897932;
    std::vector<double> flux( dz.size(), 0.0 );
    if ( heatShape == "Flat" ) {
        // sinusoidal
        for ( size_t i = 0; i < dz.size(); i++ )
            flux[i] = Q_tot / ( pi * diameter * height );
    } else if ( heatShape == "Sinusoidal" ) {
        // sinusoidal
        for ( size_t i = 0; i < dz.size(); i++ )
            flux[i] =
                Q_tot / ( 2.0 * pi * diameter * dz[i] ) *
                ( cos( pi * ( z[i] - z[0] ) / height ) - cos( pi * ( z[i + 1] - z[0] ) / height ) );
    } else {
        AMP_ERROR( "Heat shape '" + heatShape + " is invalid" );
    }
    return flux;
}

// Compute the heat flux for the subchannel assuming a heat generation rate
std::vector<double> getHeatFluxGenerationWithDiscretizationError( std::string heatShape,
                                                                  std::vector<double> z,
                                                                  double diameter,
                                                                  double Q_tot )
{
    for ( size_t i = 1; i < z.size(); i++ )
        AMP_ASSERT( z[i] > z[i - 1] );
    double height = z.back() - z.front();
    std::vector<double> dz( z.size() - 1, 0.0 );
    for ( size_t i = 0; i < dz.size(); i++ )
        dz[i] = z[i + 1] - z[i];
    const double pi = 3.1415926535897932;
    std::vector<double> flux( dz.size(), 0.0 );
    if ( heatShape == "Flat" ) {
        // sinusoidal
        for ( size_t i = 0; i < dz.size(); i++ )
            flux[i] = Q_tot / ( pi * diameter * height );
    } else if ( heatShape == "Sinusoidal" ) {
        // sinusoidal
        for ( size_t i = 0; i < dz.size(); i++ )
            flux[i] =
                Q_tot / ( 4.0 * diameter * height ) *
                ( sin( pi * ( z[i] - z[0] ) / height ) + sin( pi * ( z[i + 1] - z[0] ) / height ) );
    } else {
        AMP_ERROR( "Heat shape '" + heatShape + " is invalid" );
    }
    return flux;
}

// Compute the heat flux for the subchannel using the clad temperature
std::vector<double> getHeatFluxClad( std::vector<double> z,
                                     std::vector<AMP::Mesh::MeshElementID> face_ids,
                                     double channelDiam,
                                     double reynolds,
                                     double prandtl,
                                     double fraction,
                                     std::shared_ptr<SubchannelPhysicsModel> subchannelPhysicsModel,
                                     AMP::LinearAlgebra::Vector::const_shared_ptr flow,
                                     AMP::LinearAlgebra::Vector::const_shared_ptr clad_temp )
{
    for ( size_t i = 1; i < z.size(); i++ )
        AMP_ASSERT( z[i] > z[i - 1] );
    std::vector<double> dz( z.size() - 1, 0.0 );
    for ( size_t i = 0; i < dz.size(); i++ )
        dz[i] = z[i + 1] - z[i];
    // const double pi = 3.1415926535897932;
    AMP_ASSERT( face_ids.size() == z.size() );
    AMP_ASSERT( flow );
    AMP_ASSERT( clad_temp );
    auto flow_manager    = flow->getDOFManager();
    auto clad_manager    = clad_temp->getDOFManager();
    const double h_scale = 1.0 / Subchannel::scaleEnthalpy; // Scale to change to correct units
    const double P_scale = 1.0 / Subchannel::scalePressure; // Scale to change to correct units

    // Get the enthalapy, pressure, flow temperature, and clad temperature at the faces
    auto h  = std::make_shared<std::vector<double>>( z.size(), 0.0 );
    auto P  = std::make_shared<std::vector<double>>( z.size(), 0.0 );
    auto Tf = std::make_shared<std::vector<double>>( z.size(), 0.0 );
    auto Tc = std::make_shared<std::vector<double>>( z.size(), 0.0 );
    std::vector<size_t> flow_dofs( 2 ), clad_dofs( 1 );
    for ( size_t i = 0; i < z.size(); i++ ) {
        flow_manager->getDOFs( face_ids[i], flow_dofs );
        clad_manager->getDOFs( face_ids[i], clad_dofs );
        AMP_ASSERT( flow_dofs.size() == 2 );
        AMP_ASSERT( clad_dofs.size() == 1 );
        ( *h )[i]  = h_scale * flow->getValueByGlobalID( flow_dofs[0] );
        ( *P )[i]  = P_scale * flow->getValueByGlobalID( flow_dofs[1] );
        ( *Tc )[i] = clad_temp->getValueByGlobalID( clad_dofs[0] );
    }
    std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
    temperatureArgMap.insert( std::make_pair( std::string( "enthalpy" ), h ) );
    temperatureArgMap.insert( std::make_pair( std::string( "pressure" ), P ) );
    subchannelPhysicsModel->getProperty( "Temperature", *Tf, temperatureArgMap );
    // Get the properties at cell centers
    size_t N      = dz.size();
    auto flowTemp = std::make_shared<std::vector<double>>( N );
    auto cladTemp = std::make_shared<std::vector<double>>( N );
    auto flowDens = std::make_shared<std::vector<double>>( N );
    std::vector<double> specificVolume( z.size(), 0.0 );
    subchannelPhysicsModel->getProperty( "SpecificVolume", specificVolume, temperatureArgMap );
    for ( size_t i = 0; i < N; i++ ) {
        ( *flowTemp )[i] = 0.5 * ( ( *Tf )[i] + ( *Tf )[i + 1] );
        ( *cladTemp )[i] = 0.5 * ( ( *Tc )[i] + ( *Tc )[i + 1] );
        ( *flowDens )[i] = 0.5 * ( 1. / specificVolume[i] + 1. / specificVolume[+1] );
    }
    std::map<std::string, std::shared_ptr<std::vector<double>>> convectiveHeatArgMap;
    convectiveHeatArgMap.insert( std::make_pair( std::string( "temperature" ), cladTemp ) );
    convectiveHeatArgMap.insert( std::make_pair( std::string( "density" ), flowDens ) );
    convectiveHeatArgMap.insert( std::make_pair(
        std::string( "diameter" ), std::make_shared<std::vector<double>>( N, channelDiam ) ) );
    convectiveHeatArgMap.insert( std::make_pair(
        std::string( "reynolds" ), std::make_shared<std::vector<double>>( N, reynolds ) ) );
    convectiveHeatArgMap.insert( std::make_pair(
        std::string( "prandtl" ), std::make_shared<std::vector<double>>( N, prandtl ) ) );
    std::vector<double> heff( N );
    subchannelPhysicsModel->getProperty( "ConvectiveHeat", heff, convectiveHeatArgMap );
    std::vector<double> flux( dz.size(), 0.0 );
    for ( size_t i = 0; i < N; i++ ) {
        double dT = ( *cladTemp )[i] - ( *flowTemp )[i];
        flux[i]   = heff[i] * dT * fraction;
    }
    return flux;
}


// Function to compute the hydraulic diameter of the subchannels on the clad surface
AMP::LinearAlgebra::Vector::shared_ptr
getCladHydraulicDiameter( std::shared_ptr<AMP::Mesh::Mesh> clad,
                          std::shared_ptr<AMP::Mesh::Mesh> subchannel,
                          AMP::AMP_MPI comm )
{
    if ( clad )
        AMP_ASSERT( clad->getComm() <= comm );
    if ( subchannel )
        AMP_ASSERT( subchannel->getComm() <= comm );
    // Get the clad properties
    std::vector<double> clad_x, clad_y, clad_d;
    getCladProperties( comm, clad, clad_x, clad_y, clad_d );
    std::shared_ptr<AMP::Mesh::Mesh> clad_surface;
    if ( clad )
        clad_surface =
            clad->Subset( clad->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, 4, 1 ) );
    // Get the subchannel properties
    size_t N[2] = { 0 };
    std::vector<double> x, y, hydraulic_diam;
    int root = -1;
    if ( subchannel ) {
        std::vector<double> area, rod_diameter, channel_fraction;
        getSubchannelProperties( subchannel,
                                 clad_x,
                                 clad_y,
                                 clad_d,
                                 x,
                                 y,
                                 area,
                                 hydraulic_diam,
                                 rod_diameter,
                                 channel_fraction );
        N[0] = x.size();
        N[1] = y.size();
        AMP_ASSERT( ( N[0] - 1 ) * ( N[1] - 1 ) == hydraulic_diam.size() );
        root = comm.getRank();
    }
    root = comm.maxReduce( root );
    comm.bcast( N, 2, root );
    size_t N_subchannels = ( N[0] - 1 ) * ( N[1] - 1 );
    if ( subchannel.get() == nullptr ) {
        x.resize( N[0] );
        y.resize( N[1] );
        hydraulic_diam.resize( N_subchannels );
    }
    comm.bcast( &x[0], N[0], root );
    comm.bcast( &y[0], N[1], root );
    comm.bcast( &hydraulic_diam[0], N_subchannels, root );
    // Return if we are not on the clad surface
    if ( clad_surface.get() == nullptr )
        return AMP::LinearAlgebra::Vector::shared_ptr();
    // Create and initialize the vector
    auto DOF = AMP::Discretization::simpleDOFManager::create(
        clad_surface, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto variable = std::make_shared<AMP::LinearAlgebra::Variable>( "ChannelDiameter" );
    auto diameter = AMP::LinearAlgebra::createVector( DOF, variable );
    diameter->zero();
    auto it = clad_surface->getIterator( AMP::Mesh::GeomType::Vertex );
    std::vector<size_t> dofs( 1 );
    size_t Nx = x.size() - 1;
    size_t Ny = y.size() - 1;
    for ( size_t i = 0; i < it.size(); i++ ) {
        auto pos = it->coord();
        DOF->getDOFs( it->globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        size_t ix = AMP::Utilities::findfirst( x, pos[0] );
        size_t iy = AMP::Utilities::findfirst( y, pos[1] );
        if ( ix == 0 ) {
            ix = 1;
        }
        if ( iy == 0 ) {
            iy = 1;
        }
        ix--;
        iy--;
        if ( ix == Nx ) {
            ix = Nx - 1;
        }
        if ( iy == Nx ) {
            iy = Ny - 1;
        }
        diameter->setValuesByGlobalID( 1, &dofs[0], &hydraulic_diam[ix + iy * Nx] );
        ++it;
    }
    diameter->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    return diameter;
}
} // namespace AMP::Operator::Subchannel
