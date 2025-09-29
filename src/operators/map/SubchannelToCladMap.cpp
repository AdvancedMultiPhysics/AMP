#include "AMP/operators/map/SubchannelToCladMap.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


static double interp_linear( const std::vector<double> &, const std::vector<double> &, double );


/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
SubchannelToCladMap::SubchannelToCladMap(
    std::shared_ptr<const AMP::Operator::OperatorParameters> p )
    : AsyncMapOperator( p )
{
    auto params = std::dynamic_pointer_cast<const SubchannelToCladMapParameters>( p );
    AMP_ASSERT( params );

    // Clone the communicator to protect the communication (we need a large number of unique tags)
    d_MapComm = d_MapComm.dup();
    d_currRequests.clear();

    // Get the iterators
    if ( d_mesh1 )
        d_iterator1 = getSubchannelIterator( d_mesh1 );
    if ( d_mesh2 ) {
        int type    = params->d_db->getWithDefault<int>( "GeomType", 0 );
        d_iterator2 = d_mesh2->getBoundaryIDIterator(
            static_cast<AMP::Mesh::GeomType>( type ), params->d_BoundaryID2, 0 );
    }

    // Get the x-y-z grid for the subchannel mesh
    fillSubchannelGrid( d_mesh1 );

    // For each subchannel, get the list of local MeshElement in that channel
    if ( d_mesh2 )
        d_elem = this->getElementsInSubchannel( d_x, d_y, d_iterator2 );
    else
        d_elem = std::vector<std::vector<AMP::Mesh::MeshElementID>>( N_subchannels );

    // Get the list of processors that we will receive from for each subchannel
    d_subchannelRecv = std::vector<std::vector<int>>( N_subchannels );
    std::vector<char> tmp( d_MapComm.getSize() );
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        d_MapComm.allGather( (char) ( d_elem[i].size() > 0 ? 1 : 0 ), &tmp[0] );
        for ( size_t j = 0; j < tmp.size(); j++ ) {
            if ( tmp[j] == 1 )
                d_subchannelRecv[i].push_back( j );
        }
    }

    // Create the send/recv buffers
    d_sendBuffer = std::vector<std::vector<double>>( N_subchannels );
    if ( d_mesh1 ) {
        for ( size_t i = 0; i < N_subchannels; i++ ) {
            if ( d_ownSubChannel[i] )
                d_sendBuffer[i] = std::vector<double>( d_z.size() );
        }
    }
}


/************************************************************************
 *  De-constructor                                                       *
 ************************************************************************/
SubchannelToCladMap::~SubchannelToCladMap() = default;


/************************************************************************
 *  Check if the map type is "SubchannelToCladMap"                       *
 ************************************************************************/
bool SubchannelToCladMap::validMapType( const std::string &t )
{
    if ( t == "SubchannelToCladMap" )
        return true;
    return false;
}


/************************************************************************
 *  Function to fill the grid of the subchannel for all processors       *
 ************************************************************************/
void SubchannelToCladMap::fillSubchannelGrid( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    // Create the grid for all processors
    int root = -1;
    if ( mesh != nullptr ) {
        root = d_MapComm.getRank();
        AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( mesh, d_x, d_y, d_z );
    }
    root    = d_MapComm.maxReduce( root );
    auto Nx = d_MapComm.bcast<size_t>( d_x.size() - 1, root );
    auto Ny = d_MapComm.bcast<size_t>( d_y.size() - 1, root );
    auto Nz = d_MapComm.bcast<size_t>( d_z.size(), root );
    d_x.resize( Nx + 1, 0.0 );
    d_y.resize( Ny + 1, 0.0 );
    d_z.resize( Nz, 0.0 );
    d_MapComm.bcast<double>( &d_x[0], Nx + 1, root );
    d_MapComm.bcast<double>( &d_y[0], Ny + 1, root );
    d_MapComm.bcast<double>( &d_z[0], Nz, root );
    N_subchannels = Nx * Ny;
    // Get a list of processors that need each x-y point
    d_ownSubChannel = std::vector<bool>( Nx * Ny, false );
    if ( mesh ) {
        auto it = getSubchannelIterator( mesh );
        AMP_ASSERT( it.size() > 0 );
        for ( size_t k = 0; k < it.size(); k++, ++it ) {
            auto center = it->centroid();
            int index   = getSubchannelIndex( center[0], center[1] );
            AMP_ASSERT( index >= 0 );
            d_ownSubChannel[index] = true;
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
 *  Return the list of local MeshElements in each subchannel             *
 ************************************************************************/
std::vector<std::vector<AMP::Mesh::MeshElementID>> SubchannelToCladMap::getElementsInSubchannel(
    const std::vector<double> &, const std::vector<double> &, AMP::Mesh::MeshIterator iterator )
{
    std::vector<std::vector<AMP::Mesh::MeshElementID>> list( N_subchannels );
    auto it = iterator.begin();
    for ( size_t k = 0; k < it.size(); k++, ++it ) {
        auto center = it->centroid();
        int index   = getSubchannelIndex( center[0], center[1] );
        AMP_ASSERT( index >= 0 );
        list[index].push_back( it->globalID() );
    }
    return list;
}


/************************************************************************
 *  Get an iterator over the desired faces in the subchannel mesh        *
 ************************************************************************/
AMP::Mesh::MeshIterator
SubchannelToCladMap::getSubchannelIterator( std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    std::multimap<double, AMP::Mesh::MeshElement> xyFace;
    auto it = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( size_t i = 0; i < it.size(); ++i, ++it ) {
        auto nodes    = it->getElements( AMP::Mesh::GeomType::Vertex );
        auto center   = it->centroid();
        bool is_valid = true;
        for ( auto &node : nodes ) {
            auto coord = node.coord();
            if ( !AMP::Utilities::approx_equal( coord[2], center[2], 1e-6 ) )
                is_valid = false;
        }
        if ( is_valid ) {
            xyFace.insert( std::pair<double, AMP::Mesh::MeshElement>( center[2], *it ) );
        }
    }
    auto elements = std::make_shared<std::vector<AMP::Mesh::MeshElement>>();
    elements->reserve( xyFace.size() );
    for ( auto &elem : xyFace )
        elements->push_back( elem.second );
    return AMP::Mesh::MeshElementVectorIterator( elements );
}


/************************************************************************
 *  Start the communication                                              *
 ************************************************************************/
void SubchannelToCladMap::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                      AMP::LinearAlgebra::Vector::shared_ptr )
{
    PROFILE( "applyStart" );
    // Check if we have any data to send
    if ( d_mesh1.get() == nullptr )
        return;

    // Subset the vector for the variable (we only need the local portion of the vector)
    auto var = getInputVariable();
    AMP::LinearAlgebra::VS_Comm commSelector( AMP_COMM_SELF );
    auto commSubsetVec = u->select( commSelector );
    auto curPhysics    = commSubsetVec->subsetVectorForVariable( var );
    AMP_ASSERT( curPhysics );

    // Zero out the send buffers
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( !d_ownSubChannel[i] )
            continue;
        for ( size_t j = 0; j < d_z.size(); j++ )
            d_sendBuffer[i][j] = 0.0;
    }
    // Fill the faces I own
    auto DOF = curPhysics->getDOFManager();
    std::vector<size_t> dofs;
    auto it = d_iterator1.begin();
    for ( size_t i = 0; i < it.size(); i++, ++it ) {
        auto id = it->globalID();
        if ( !id.is_local() )
            continue;
        DOF->getDOFs( id, dofs );
        AMP_ASSERT( dofs.size() == 1 );
        double val = curPhysics->getLocalValueByGlobalID( dofs[0] );
        auto pos   = it->centroid();
        int index  = getSubchannelIndex( pos[0], pos[1] );
        AMP_ASSERT( index >= 0 );
        AMP_ASSERT( d_ownSubChannel[index] );
        int index2 = Utilities::findfirst( d_z, pos[2] - 1e-12 );
        AMP_ASSERT( Utilities::approx_equal( pos[2], d_z[index2] ) );
        d_sendBuffer[index][index2] = val;
    }

    // Send the data
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( !d_ownSubChannel[i] )
            continue;
        auto tag = (int) i; // We have an independent comm
        for ( size_t j = 0; j < d_subchannelRecv[i].size(); j++ ) {
            int rank = d_subchannelRecv[i][j];
            d_currRequests.push_back(
                d_MapComm.Isend<double>( &d_sendBuffer[i][0], d_sendBuffer[i].size(), rank, tag ) );
        }
    }
}


/************************************************************************
 *  Finish the communication                                             *
 ************************************************************************/
void SubchannelToCladMap::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                       AMP::LinearAlgebra::Vector::shared_ptr )
{
    PROFILE( "applyFinish" );
    if ( d_mesh2.get() == nullptr ) {
        // We don't have an output vector to fill, wait for communication to finish and return
        if ( d_currRequests.size() > 0 )
            AMP::AMP_MPI::waitAll( (int) d_currRequests.size(), &d_currRequests[0] );
        d_currRequests.resize( 0 );
        return;
    }
    // Recieve the data
    std::vector<std::vector<double>> f( N_subchannels );
    auto tmp_data = new double[d_z.size()];
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( d_elem[i].size() > 0 ) {
            f[i]     = std::vector<double>( d_z.size(), 0.0 );
            auto tag = (int) i; // We have an independent comm
            for ( size_t j = 0; j < d_subchannelRanks[i].size(); j++ ) {
                int length = d_z.size();
                d_MapComm.recv<double>( tmp_data, length, d_subchannelRanks[i][j], false, tag );
                for ( size_t k = 0; k < d_z.size(); k++ )
                    f[i][k] +=
                        tmp_data[k]; // This works since each z-point in only owned by one processor
            }
        }
    }
    delete[] tmp_data;
    // Fill the output vector
    // std::shared_ptr<AMP::Discretization::DOFManager>  DOF = d_OutputVector->getDOFManager( );
    double range[4];
    int Nx = d_x.size() - 1;
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( d_elem[i].size() > 0 ) {
            range[0] = d_x[( i % Nx )];
            range[1] = d_x[( i % Nx ) + 1];
            range[2] = d_y[( i / Nx )];
            range[3] = d_y[( i / Nx ) + 1];
            this->fillReturnVector( d_OutputVector, range, d_mesh2, d_elem[i], d_z, f[i] );
        }
    }
    // Wait for all communication to finish
    if ( d_currRequests.size() > 0 )
        AMP::AMP_MPI::waitAll( (int) d_currRequests.size(), &d_currRequests[0] );
    d_currRequests.resize( 0 );
    // Call makeConsistent
    d_OutputVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


/************************************************************************
 *  Fill the return vector for the given subchannel                      *
 ************************************************************************/
void SubchannelToCladMap::fillReturnVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                            double[4],
                                            std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                            const std::vector<AMP::Mesh::MeshElementID> &ids,
                                            const std::vector<double> &z,
                                            const std::vector<double> &f )
{
    PROFILE( "fillReturnVector" );
    std::shared_ptr<AMP::Discretization::DOFManager> DOF = vec->getDOFManager();
    std::vector<size_t> dofs( 1 );
    for ( auto &id : ids ) {
        DOF->getDOFs( id, dofs );
        AMP_ASSERT( dofs.size() == 1 );
        auto pos   = mesh->getElement( id ).coord();
        double val = interp_linear( z, f, pos[2] );
        vec->setValuesByGlobalID( 1, &dofs[0], &val );
    }
}


/************************************************************************
 *  Set the vector                                                       *
 ************************************************************************/
void SubchannelToCladMap::setVector( AMP::LinearAlgebra::Vector::shared_ptr result )
{
    if ( result )
        d_OutputVector = subsetOutputVector( result );
    else
        d_OutputVector = nullptr;
    if ( d_mesh2 )
        AMP_ASSERT( d_OutputVector );
}


/************************************************************************
 *  Misc functions                                                       *
 ************************************************************************/
int SubchannelToCladMap::getSubchannelIndex( double x, double y )
{
    size_t i = Utilities::findfirst( d_x, x );
    size_t j = Utilities::findfirst( d_y, y );
    if ( i > 0 && i < d_x.size() && j > 0 && j < d_y.size() )
        return ( i - 1 ) + ( j - 1 ) * ( d_x.size() - 1 );
    return -1;
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


} // namespace AMP::Operator
