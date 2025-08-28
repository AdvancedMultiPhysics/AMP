#include "AMP/operators/subchannel/SubchannelToPointMap.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"
#include "ProfilerApp.h"


namespace AMP::Operator {


// Constructor
SubchannelToPointMap::SubchannelToPointMap(
    std::shared_ptr<const SubchannelToPointMapParameters> params )
{
    // Copy the inputs
    AMP_ASSERT( params );
    d_Mesh                   = params->d_Mesh;
    d_comm                   = params->d_comm;
    d_point_x                = params->x;
    d_point_y                = params->y;
    d_point_z                = params->z;
    d_outputVar              = params->d_outputVar;
    d_subchannelPhysicsModel = params->d_subchannelPhysicsModel;
    // Check the inputs
    AMP_INSIST( !d_comm.isNull(), "d_comm must not by NULL" );
    AMP_INSIST( d_comm.anyReduce( d_Mesh.get() != nullptr ),
                "d_Mesh must be set on at least one processor" );
    AMP_MPI mesh_comm( AMP_COMM_SELF );
    if ( d_Mesh )
        mesh_comm = d_Mesh->getComm();
    AMP_INSIST( d_comm >= mesh_comm, "d_comm must be >= comm of the subchannel mesh" );
    AMP_ASSERT( d_point_x.size() == d_point_y.size() && d_point_x.size() == d_point_z.size() );
    AMP_ASSERT( d_subchannelPhysicsModel );
    if ( !d_point_x.empty() ) {
        AMP_INSIST( d_outputVar, "Output variable is NULL for subchannel to point map" );
        AMP_INSIST( d_outputVar->getName() == "Density" || d_outputVar->getName() == "Temperature",
                    "Invalid output variable for subchannel to point map" );
    }
    // Get the coordinates of the subchannel mesh
    this->createGrid();
}


// Perform the map
void SubchannelToPointMap::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                  AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );
    // Get the list of all subchannel face densities
    std::vector<double> output( N_subchannels * d_subchannel_z.size(), 0.0 );
    if ( d_Mesh ) {
        auto uInternal = subsetInputVector( u );
        AMP_ASSERT( uInternal );
        auto faceDOFManager = uInternal->getDOFManager();
        AMP::Mesh::MeshIterator it =
            AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( d_Mesh, 0 );
        std::vector<size_t> dofs;
        const double h_scale =
            1.0 /
            Subchannel::scaleEnthalpy; // Scale to change the input vector back to correct units
        const double P_scale =
            1.0 /
            Subchannel::scalePressure; // Scale to change the input vector back to correct units
        for ( size_t i = 0; i < it.size(); i++ ) {
            // Get the output variable for the current face
            double output_face = 0.0;
            faceDOFManager->getDOFs( it->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 2 );
            std::map<std::string, std::shared_ptr<std::vector<double>>> subchannelArgMap;
            subchannelArgMap.insert(
                std::make_pair( std::string( "enthalpy" ),
                                std::make_shared<std::vector<double>>(
                                    1, h_scale * uInternal->getValueByGlobalID( dofs[0] ) ) ) );
            subchannelArgMap.insert(
                std::make_pair( std::string( "pressure" ),
                                std::make_shared<std::vector<double>>(
                                    1, P_scale * uInternal->getValueByGlobalID( dofs[1] ) ) ) );
            if ( d_outputVar->getName() == "Density" ) {
                std::vector<double> specificVolume( 1 );
                d_subchannelPhysicsModel->getProperty(
                    "SpecificVolume", specificVolume, subchannelArgMap );
                output_face = 1.0 / specificVolume[0];
            } else if ( d_outputVar->getName() == "Temperature" ) {
                std::vector<double> temperature( 1 );
                d_subchannelPhysicsModel->getProperty(
                    "Temperature", temperature, subchannelArgMap );
                output_face = temperature[0];
            } else {
                AMP_ERROR( "Unknown output property" );
            }
            // Add it to the subchannel density vector
            auto center = it->centroid();
            size_t ix   = AMP::Utilities::findfirst( d_subchannel_x, center[0] - 1e-9 );
            size_t iy   = AMP::Utilities::findfirst( d_subchannel_y, center[1] - 1e-9 );
            size_t iz   = AMP::Utilities::findfirst( d_subchannel_z, center[2] - 1e-9 );
            AMP_ASSERT( AMP::Utilities::approx_equal( d_subchannel_x[ix], center[0] ) );
            AMP_ASSERT( AMP::Utilities::approx_equal( d_subchannel_y[iy], center[1] ) );
            AMP_ASSERT( AMP::Utilities::approx_equal( d_subchannel_z[iz], center[2] ) );
            size_t index = ix + iy * d_subchannel_x.size() + iz * N_subchannels;
            output[index] += output_face;
            ++it;
        }
    }
    d_comm.sumReduce<double>( &output[0], output.size() );

    // Perform tri-linear interpolation to fill the output
    AMP::LinearAlgebra::VS_Comm commSelector( d_comm );
    auto outputVec = r->select( commSelector );
    if ( outputVec )
        outputVec = outputVec->subsetVectorForVariable( getOutputVariable() );
    std::vector<double> localOutput( d_point_x.size(), 0.0 );
    if ( d_subchannel_x.size() > 1 && d_subchannel_y.size() > 1 ) {
        for ( size_t i = 0; i < d_point_x.size(); i++ )
            localOutput[i] = AMP::Utilities::trilinear( d_subchannel_x,
                                                        d_subchannel_y,
                                                        d_subchannel_z,
                                                        output,
                                                        d_point_x[i],
                                                        d_point_y[i],
                                                        d_point_z[i] );
    } else if ( d_subchannel_x.size() > 1 ) {
        for ( size_t i = 0; i < d_point_x.size(); i++ )
            localOutput[i] = AMP::Utilities::bilinear(
                d_subchannel_x, d_subchannel_z, output, d_point_x[i], d_point_z[i] );
    } else if ( d_subchannel_y.size() > 1 ) {
        for ( size_t i = 0; i < d_point_x.size(); i++ )
            localOutput[i] = AMP::Utilities::bilinear(
                d_subchannel_y, d_subchannel_z, output, d_point_y[i], d_point_z[i] );
    } else {
        for ( size_t i = 0; i < d_point_x.size(); i++ )
            localOutput[i] = AMP::Utilities::linear( d_subchannel_z, output, d_point_z[i] );
    }
    if ( d_point_x.size() > 0 ) {
        AMP_ASSERT( outputVec );
        AMP_ASSERT( outputVec->getLocalSize() == d_point_x.size() );
        std::vector<size_t> dofs( d_point_x.size() );
        for ( size_t i = 0; i < d_point_x.size(); i++ )
            dofs[i] = i;
        outputVec->setValuesByLocalID( dofs.size(), &dofs[0], &localOutput[0] );
    }
    if ( outputVec )
        outputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


// Create the subchannel grid for all processors
void SubchannelToPointMap::createGrid()
{
    PROFILE( "createGrid" );
    // Create the grid for all processors
    std::vector<double> x, y, z;
    int root = -1;
    if ( d_Mesh ) {
        root = d_comm.getRank();
        AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( d_Mesh, x, y, z );
    }
    root    = d_comm.maxReduce( root );
    auto Nx = d_comm.bcast<size_t>( x.size() - 1, root );
    auto Ny = d_comm.bcast<size_t>( y.size() - 1, root );
    auto Nz = d_comm.bcast<size_t>( z.size(), root );
    x.resize( Nx + 1, 0.0 );
    y.resize( Ny + 1, 0.0 );
    z.resize( Nz, 0.0 );
    d_comm.bcast<double>( &x[0], Nx + 1, root );
    d_comm.bcast<double>( &y[0], Ny + 1, root );
    d_comm.bcast<double>( &z[0], Nz, root );
    d_subchannel_x.resize( Nx );
    d_subchannel_y.resize( Ny );
    d_subchannel_z = z;
    for ( size_t i = 0; i < Nx; i++ )
        d_subchannel_x[i] = 0.5 * ( x[i + 1] + x[i] );
    for ( size_t i = 0; i < Ny; i++ )
        d_subchannel_y[i] = 0.5 * ( y[i + 1] + y[i] );
    N_subchannels = Nx * Ny;
}
} // namespace AMP::Operator
