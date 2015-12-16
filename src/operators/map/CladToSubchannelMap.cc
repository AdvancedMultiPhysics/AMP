#include "operators/map/CladToSubchannelMap.h"
#include "ProfilerApp.h"
#include "ampmesh/StructuredMeshHelper.h"
#include "discretization/DOF_Manager.h"
#include "utils/PIO.h"


namespace AMP {
namespace Operator {


static void create_map( const std::vector<std::pair<double, double>> &,
                        std::vector<double> &,
                        std::vector<double> & );
static double interp_linear( const std::vector<double> &, const std::vector<double> &, double );


/************************************************************************
*  Default constructor                                                  *
************************************************************************/
CladToSubchannelMap::CladToSubchannelMap(
    const AMP::shared_ptr<AMP::Operator::OperatorParameters> &p )
    : AsyncMapOperator( p )
{
    AMP::shared_ptr<CladToSubchannelMapParameters> params =
        AMP::dynamic_pointer_cast<CladToSubchannelMapParameters>( p );
    AMP_ASSERT( params );

    int DofsPerObj = params->d_db->getInteger( "DOFsPerObject" );
    AMP_INSIST( DofsPerObj == 1,
                "CladToSubchannelMap is currently only designed for 1 DOF per node" );

    // Clone the communicator to protect the communication (we need a large number of unique tags)
    d_MapComm      = d_MapComm.dup();
    d_currRequests = std::vector<MPI_Request>();

    // Get the iterators
    if ( d_mesh1.get() != nullptr )
        d_iterator1 = d_mesh1->getBoundaryIDIterator( AMP::Mesh::Vertex, params->d_BoundaryID1, 0 );
    if ( d_mesh2.get() != nullptr )
        d_iterator2 = getSubchannelIterator( d_mesh2 );

    // Get the x-y-z grid for the subchannel mesh
    fillSubchannelGrid( d_mesh2 );

    // For each subchannel, get the list of local MeshElement in that channel
    d_elem = std::vector<std::vector<AMP::Mesh::MeshElementID>>( N_subchannels );
    if ( d_mesh1.get() != nullptr ) {
        AMP::Mesh::MeshIterator it = d_iterator1.begin();
        for ( size_t k = 0; k < it.size(); k++ ) {
            std::vector<double> center = it->centroid();
            int index                  = getSubchannelIndex( center[0], center[1] );
            if ( index >= 0 )
                d_elem[index].push_back( it->globalID() );
            ++it;
        }
    }

    // Get the list of processors that we will recieve from for each subchannel
    d_subchannelSend = std::vector<std::vector<int>>( N_subchannels );
    std::vector<char> tmp( d_MapComm.getSize() );
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        d_MapComm.allGather( (char) ( d_elem[i].size() > 0 ? 1 : 0 ), &tmp[0] );
        for ( size_t j = 0; j < tmp.size(); j++ ) {
            if ( tmp[j] == 1 )
                d_subchannelSend[i].push_back( j );
        }
    }

    // Create the send/recv buffers
    d_sendMaxBufferSize = 0;
    d_sendBuffer        = std::vector<double *>( N_subchannels, nullptr );
    if ( d_mesh1.get() != nullptr ) {
        for ( size_t i = 0; i < N_subchannels; i++ ) {
            if ( d_elem[i].size() > 0 ) {
                d_sendBuffer[i] = new double[2 * d_elem[i].size()];
                if ( d_elem[i].size() > d_sendMaxBufferSize )
                    d_sendMaxBufferSize = d_elem[i].size();
            }
        }
    }
    d_sendMaxBufferSize = d_MapComm.maxReduce( d_sendMaxBufferSize );
}


/************************************************************************
*  De-constructor                                                       *
************************************************************************/
CladToSubchannelMap::~CladToSubchannelMap()
{
    for ( size_t i = 0; i < d_sendBuffer.size(); i++ ) {
        delete[] d_sendBuffer[i];
        d_sendBuffer[i] = nullptr;
    }
    d_sendBuffer.resize( 0 );
    d_x.resize( 0 );
    d_y.resize( 0 );
    d_z.resize( 0 );
    d_ownSubChannel.resize( 0 );
    d_subchannelRanks.resize( 0 );
    d_subchannelSend.resize( 0 );
    d_elem.resize( 0 );
    d_currRequests.resize( 0 );
}


/************************************************************************
*  Check if the map type is "CladToSubchannelMap"                       *
************************************************************************/
bool CladToSubchannelMap::validMapType( const std::string &t )
{
    if ( t == "CladToSubchannelMap" )
        return true;
    return false;
}


/************************************************************************
*  Function to fill the grid of the subchannel for all processors       *
************************************************************************/
void CladToSubchannelMap::fillSubchannelGrid( AMP::Mesh::Mesh::shared_ptr mesh )
{
    // Create the grid for all processors
    int root = -1;
    if ( mesh != nullptr ) {
        root = d_MapComm.getRank();
        AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( mesh, d_x, d_y, d_z );
    }
    root      = d_MapComm.maxReduce( root );
    size_t Nx = d_MapComm.bcast<size_t>( d_x.size() - 1, root );
    size_t Ny = d_MapComm.bcast<size_t>( d_y.size() - 1, root );
    size_t Nz = d_MapComm.bcast<size_t>( d_z.size(), root );
    d_x.resize( Nx + 1, 0.0 );
    d_y.resize( Ny + 1, 0.0 );
    d_z.resize( Nz, 0.0 );
    d_MapComm.bcast<double>( &d_x[0], Nx + 1, root );
    d_MapComm.bcast<double>( &d_y[0], Ny + 1, root );
    d_MapComm.bcast<double>( &d_z[0], Nz, root );
    N_subchannels = Nx * Ny;
    // Get a list of processors that need each x-y point
    d_ownSubChannel = std::vector<bool>( Nx * Ny, false );
    if ( mesh.get() != nullptr ) {
        AMP::Mesh::MeshIterator it = getSubchannelIterator( mesh );
        AMP_ASSERT( it.size() > 0 );
        for ( size_t k = 0; k < it.size(); k++ ) {
            std::vector<double> center = it->centroid();
            int index                  = getSubchannelIndex( center[0], center[1] );
            AMP_ASSERT( index >= 0 );
            d_ownSubChannel[index] = true;
            ++it;
        }
    }
    d_subchannelRanks = std::vector<std::vector<int>>( Nx * Ny );
    std::vector<char> tmp( d_MapComm.getSize() );
    for ( size_t i = 0; i < Nx * Ny; i++ ) {
        d_MapComm.allGather( (char) ( d_ownSubChannel[i] ? 1 : 0 ), &tmp[0] );
        for ( size_t j = 0; j < tmp.size(); j++ ) {
            if ( tmp[j] == 1 )
                d_subchannelRanks[i].push_back( j );
        }
    }
}


/************************************************************************
*  Get an iterator over the desired faces in the subchannel mesh        *
************************************************************************/
AMP::Mesh::MeshIterator
CladToSubchannelMap::getSubchannelIterator( AMP::Mesh::Mesh::shared_ptr mesh )
{
    std::multimap<double, AMP::Mesh::MeshElement> xyFace;
    AMP::Mesh::MeshIterator iterator = mesh->getIterator( AMP::Mesh::Face, 0 );
    for ( size_t i = 0; i < iterator.size(); ++i ) {
        std::vector<AMP::Mesh::MeshElement> nodes = iterator->getElements( AMP::Mesh::Vertex );
        std::vector<double> center                = iterator->centroid();
        bool is_valid                             = true;
        for ( size_t j = 0; j < nodes.size(); ++j ) {
            std::vector<double> coord = nodes[j].coord();
            if ( !AMP::Utilities::approx_equal( coord[2], center[2], 1e-6 ) )
                is_valid = false;
        }
        if ( is_valid ) {
            xyFace.insert( std::pair<double, AMP::Mesh::MeshElement>( center[2], *iterator ) );
        }
        ++iterator;
    }
    AMP::shared_ptr<std::vector<AMP::Mesh::MeshElement>> elements(
        new std::vector<AMP::Mesh::MeshElement>() );
    elements->reserve( xyFace.size() );
    for ( std::multimap<double, AMP::Mesh::MeshElement>::iterator it = xyFace.begin();
          it != xyFace.end();
          ++it )
        elements->push_back( it->second );
    return AMP::Mesh::MultiVectorIterator( elements );
}


/************************************************************************
*  Start the communication                                              *
************************************************************************/
void CladToSubchannelMap::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                      AMP::LinearAlgebra::Vector::shared_ptr )
{
    // Check if we have any data to send
    if ( d_mesh1.get() == nullptr )
        return;

    // Subset the vector for the variable (we only need the local portion of the vector)
    AMP::LinearAlgebra::Variable::shared_ptr var = getInputVariable();
    AMP::LinearAlgebra::VS_Comm commSelector( AMP_MPI( AMP_COMM_SELF ) );
    AMP::LinearAlgebra::Vector::const_shared_ptr commSubsetVec =
        u->constSelect( commSelector, u->getVariable()->getName() );
    AMP::LinearAlgebra::Vector::const_shared_ptr curPhysics =
        commSubsetVec->constSubsetVectorForVariable( var );
    AMP_ASSERT( curPhysics );

    // Fill the send buffer
    AMP::Discretization::DOFManager::shared_ptr DOF = curPhysics->getDOFManager();
    std::vector<size_t> dofs;
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( d_elem[i].empty() )
            continue;
        for ( size_t j         = 0; j < 2 * d_elem[i].size(); j++ )
            d_sendBuffer[i][j] = 0.0;
        for ( size_t j = 0; j < d_elem[i].size(); j++ ) {
            DOF->getDOFs( d_elem[i][j], dofs );
            AMP_ASSERT( dofs.size() == 1 );
            std::vector<double> pos    = d_mesh1->getElement( d_elem[i][j] ).centroid();
            double val                 = curPhysics->getLocalValueByGlobalID( dofs[0] );
            d_sendBuffer[i][2 * j + 0] = pos[2];
            d_sendBuffer[i][2 * j + 1] = val;
        }
    }
    // Send the data
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( d_elem[i].empty() )
            continue;
        int tag = (int) i; // We have an independent comm
        for ( size_t j = 0; j < d_subchannelRanks[i].size(); j++ ) {
            int rank = d_subchannelRanks[i][j];
            d_currRequests.push_back(
                d_MapComm.Isend( &d_sendBuffer[i][0], 2 * d_elem[i].size(), rank, tag ) );
        }
    }
}


/************************************************************************
*  Finish the communication                                             *
************************************************************************/
void CladToSubchannelMap::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                       AMP::LinearAlgebra::Vector::shared_ptr )
{
    if ( d_mesh2.get() == nullptr ) {
        // We don't have an output vector to fill, wait for communication to finish and return
        if ( d_currRequests.size() > 0 )
            AMP::AMP_MPI::waitAll( (int) d_currRequests.size(), &d_currRequests[0] );
        d_currRequests.resize( 0 );
        return;
    }
    // Recieve the data
    std::vector<std::vector<double>> x( N_subchannels );
    std::vector<std::vector<double>> f( N_subchannels );
    std::vector<std::pair<double, double>> mapData;
    double *tmp_data = new double[2 * d_sendMaxBufferSize];
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( d_ownSubChannel[i] ) {
            int tag = (int) i; // We have an independent comm
            mapData.resize( 0 );
            for ( size_t j = 0; j < d_subchannelSend[i].size(); j++ ) {
                int length = 2 * d_sendMaxBufferSize;
                d_MapComm.recv( tmp_data, length, d_subchannelSend[i][j], true, tag );
                AMP_ASSERT( length % 2 == 0 );
                mapData.reserve( mapData.size() + length / 2 );
                for ( int k = 0; k < length / 2; k++ )
                    mapData.push_back(
                        std::pair<double, double>( tmp_data[2 * k + 0], tmp_data[2 * k + 1] ) );
            }
            create_map( mapData, x[i], f[i] );
        }
    }
    delete[] tmp_data;
    // Fill the output vector
    AMP::Discretization::DOFManager::shared_ptr DOF = d_OutputVector->getDOFManager();
    AMP::Mesh::MeshIterator it                      = d_iterator2.begin();
    std::vector<size_t> dofs;
    for ( size_t k = 0; k < it.size(); k++ ) {
        DOF->getDOFs( it->globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        std::vector<double> pos = it->centroid();
        int index               = getSubchannelIndex( pos[0], pos[1] );
        AMP_ASSERT( index >= 0 );
        if ( x[index].size() > 0 ) {
            double val = interp_linear( x[index], f[index], pos[2] );
            d_OutputVector->setLocalValueByGlobalID( dofs[0], val );
        }
        ++it;
    }
    // Wait for all communication to finish
    if ( d_currRequests.size() > 0 )
        AMP::AMP_MPI::waitAll( (int) d_currRequests.size(), &d_currRequests[0] );
    d_currRequests.resize( 0 );
    // Call makeConsistent
    d_OutputVector->makeConsistent( AMP::LinearAlgebra::Vector::CONSISTENT_SET );
}


/************************************************************************
*  Set the vector                                                       *
************************************************************************/
void CladToSubchannelMap::setVector( AMP::LinearAlgebra::Vector::shared_ptr result )
{
    if ( result.get() != nullptr )
        d_OutputVector = subsetInputVector( result );
    else
        d_OutputVector = AMP::LinearAlgebra::Vector::shared_ptr();
    if ( d_mesh2.get() != nullptr )
        AMP_ASSERT( d_OutputVector.get() != nullptr );
}


/************************************************************************
*  Misc functions                                                       *
************************************************************************/
int CladToSubchannelMap::getSubchannelIndex( double x, double y )
{
    size_t i = Utilities::findfirst( d_x, x );
    size_t j = Utilities::findfirst( d_y, y );
    if ( i > 0 && i < d_x.size() && j > 0 && j < d_y.size() )
        return ( i - 1 ) + ( j - 1 ) * ( d_x.size() - 1 );
    return -1;
}
void create_map( const std::vector<std::pair<double, double>> &points,
                 std::vector<double> &x,
                 std::vector<double> &f )
{
    if ( points.empty() ) {
        x.resize( 0 );
        f.resize( 0 );
        return;
    }
    std::vector<double> x1( points.size() );
    std::vector<double> f1( points.size() );
    for ( size_t i = 0; i < points.size(); i++ ) {
        x1[i] = points[i].first;
        f1[i] = points[i].second;
    }
    AMP::Utilities::quicksort( x1, f1 );
    std::vector<int> count( points.size(), 1 );
    x.resize( points.size() );
    f.resize( points.size() );
    size_t index = 0;
    x[0]         = x1[0];
    f[0]         = f1[0];
    for ( size_t i = 1; i < x1.size(); i++ ) {
        if ( AMP::Utilities::approx_equal( x1[i], x1[i - 1], 1e-10 ) ) {
            f[index] += f1[i];
            count[index]++;
        } else {
            index++;
            x[index] = x1[i];
            f[index] = f1[i];
        }
    }
    x.resize( index + 1 );
    f.resize( index + 1 );
    count.resize( index + 1 );
    for ( size_t i = 0; i < x.size(); i++ )
        f[i] /= (double) count[i];
}
double interp_linear( const std::vector<double> &x, const std::vector<double> &f, double pos )
{
    AMP_ASSERT( x.size() >= 2 && x.size() == f.size() );
    size_t i = AMP::Utilities::findfirst( x, pos );
    if ( i == 0 ) {
        i = 1;
    }
    return f[i - 1] + ( pos - x[i - 1] ) / ( x[i] - x[i - 1] ) * ( f[i] - f[i - 1] );
}


} // Operator namespace
} // AMP namespace
