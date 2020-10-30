#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/ampmesh/MeshElement.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/map/NodeToNodeMapParameters.h"
#include "AMP/vectors/Variable.h"
#include "ProfilerApp.h"

#include <set>


namespace AMP {
namespace Operator {


template<class T>
static T *getPtr( std::vector<T> &x )
{
    if ( x.empty() )
        return nullptr;
    return &x[0];
}


/********************************************************
 * Constructor                                           *
 ********************************************************/
NodeToNodeMap::NodeToNodeMap( const std::shared_ptr<AMP::Operator::OperatorParameters> &params )
    : AMP::Operator::AsyncMapOperator( params )
{
    // Cast the params appropriately
    d_OutputVector = AMP::LinearAlgebra::Vector::shared_ptr();
    AMP_ASSERT( params );
    NodeToNodeMapParameters &Params =
        *( std::dynamic_pointer_cast<NodeToNodeMapParameters>( params ) );

    // Set class members
    dim = -1;
    if ( d_mesh1.get() != nullptr )
        dim = d_mesh1->getDim();
    dim = d_MapComm.maxReduce( dim );
    AMP_INSIST( dim <= 3, "Node to Node map only works up to 3d (see Point)" );
    DofsPerObj = Params.d_db->getWithDefault( "DOFsPerObject", 1 );
    AMP::Mesh::GeomType geomType =
        static_cast<AMP::Mesh::GeomType>( Params.d_db->getWithDefault( "GeomType", 0 ) );
    d_commTag               = Params.d_commTag;
    d_callMakeConsistentSet = Params.callMakeConsistentSet;

    // Create the element iterators
    if ( d_mesh1.get() != nullptr )
        d_iterator1 = d_mesh1->getBoundaryIDIterator( geomType, Params.d_BoundaryID1, 0 );
    if ( d_mesh2.get() != nullptr )
        d_iterator2 = d_mesh2->getBoundaryIDIterator( geomType, Params.d_BoundaryID2, 0 );

    // Create the pairs of points that are aligned
    createPairs();

    // Now we need to create the DOF lists and setup the communication
    buildSendRecvList();

    // Determine the number of pair-wise communications
    size_t numPartners = 0;
    for ( auto &elem : d_count ) {
        if ( elem > 0 )
            numPartners++;
    }
    reserveRequests( 2 * numPartners );

    // Finished setting up the communication
}


/********************************************************
 * De-constructor                                        *
 ********************************************************/
NodeToNodeMap::~NodeToNodeMap() = default;


/********************************************************
 * Check if the string matches a NodeToNode map          *
 ********************************************************/
bool NodeToNodeMap::validMapType( const std::string &t )
{
    if ( t == "NodeToNode" )
        return true;
    return false;
}


/********************************************************
 * Set the vector                                        *
 ********************************************************/
void NodeToNodeMap::setVector( AMP::LinearAlgebra::Vector::shared_ptr p )
{
    d_OutputVector = subsetInputVector( p );
    AMP_INSIST( d_OutputVector, "setVector received bogus stuff" );
}
bool NodeToNodeMap::requiresMakeConsistentSet() { return !d_callMakeConsistentSet; }
AMP::LinearAlgebra::Vector::shared_ptr NodeToNodeMap::getVector() { return d_OutputVector; }


/********************************************************
 * Start the communication                               *
 ********************************************************/
void NodeToNodeMap::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                AMP::LinearAlgebra::Vector::shared_ptr )
{
    PROFILE_START( "applyStart" );

    // Subset the vector for the variable (we only need the local portion of the vector)
    PROFILE_START( "subset", 1 );
    AMP::LinearAlgebra::Variable::shared_ptr var = getInputVariable();
    AMP::LinearAlgebra::VS_Comm commSelector( AMP_MPI( AMP_COMM_SELF ) );
    AMP::LinearAlgebra::Vector::const_shared_ptr commSubsetVec =
        u->constSelect( commSelector, u->getVariable()->getName() );
    AMP::LinearAlgebra::Vector::const_shared_ptr curPhysics =
        commSubsetVec->constSubsetVectorForVariable( var );
    PROFILE_STOP( "subset", 1 );
    AMP_INSIST( curPhysics, "apply received bogus stuff" );

    // Get the DOFs to send
    PROFILE_START( "getDOFs", 1 );
    AMP::Discretization::DOFManager::shared_ptr DOF = curPhysics->getDOFManager();
    std::vector<size_t> dofs( DofsPerObj * d_sendList.size() );
    std::vector<size_t> local_dofs( DofsPerObj );
    for ( size_t i = 0; i < d_sendList.size(); i++ ) {
        DOF->getDOFs( d_sendList[i], local_dofs );
        AMP_ASSERT( (int) local_dofs.size() == DofsPerObj );
        for ( int j = 0; j < DofsPerObj; j++ )
            dofs[j + i * DofsPerObj] = local_dofs[j];
    }
    PROFILE_STOP( "getDOFs", 1 );

    // Get the values to send
    PROFILE_START( "getValues", 1 );
    curPhysics->getValuesByGlobalID( dofs.size(), getPtr( dofs ), getPtr( d_sendBuffer ) );
    PROFILE_STOP( "getValues", 1 );

    // Start the communication
    PROFILE_START( "startCommunication", 1 );
    auto curReq = beginRequests();
    for ( int i = 0; i < d_MapComm.getSize(); i++ ) {
        int count  = DofsPerObj * d_count[i];
        int offset = DofsPerObj * d_displ[i];
        if ( i == d_MapComm.getRank() ) {
            // Perform a local copy
            for ( int j = offset; j < offset + count; j++ )
                d_recvBuffer[j] = d_sendBuffer[j];
        } else if ( count > 0 ) {
            // Start asyncronous communication
            *curReq = d_MapComm.Isend( &d_sendBuffer[offset], count, i, d_commTag );
            ++curReq;
            *curReq = d_MapComm.Irecv( &d_recvBuffer[offset], count, i, d_commTag );
            ++curReq;
        }
    }
    PROFILE_STOP( "startCommunication", 1 );
    PROFILE_STOP( "applyStart" );
}


/********************************************************
 * Finish the communication and fill the vector          *
 ********************************************************/
void NodeToNodeMap::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                 AMP::LinearAlgebra::Vector::shared_ptr )
{
    PROFILE_START( "applyFinish" );

    // Get the DOFs to recv
    AMP::Discretization::DOFManager::shared_ptr DOF = d_OutputVector->getDOFManager();
    std::vector<size_t> dofs( DofsPerObj * d_recvList.size() );
    std::vector<size_t> local_dofs( DofsPerObj );
    for ( size_t i = 0; i < d_recvList.size(); i++ ) {
        DOF->getDOFs( d_recvList[i], local_dofs );
        AMP_ASSERT( (int) local_dofs.size() == DofsPerObj );
        for ( int j = 0; j < DofsPerObj; j++ )
            dofs[j + i * DofsPerObj] = local_dofs[j];
    }

    // Wait to receive all data
    waitForAllRequests();

    // Store the DOFs
    d_OutputVector->setLocalValuesByGlobalID( dofs.size(), getPtr( dofs ), getPtr( d_recvBuffer ) );

    // Update ghost cells (this should be done on the full output vector)
    if ( d_callMakeConsistentSet )
        d_OutputVector->makeConsistent(
            AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    PROFILE_STOP( "applyFinish" );
}


/****************************************************************
 * Function to compute the send/recv lists                       *
 * We need to group the DOFs to send/receive by processor and    *
 * ensure that both processors will agree on the order.          *
 ****************************************************************/
void NodeToNodeMap::buildSendRecvList()
{
    // First, count the total number of elemnt IDs to send/recv for each processor
    int commSize = d_MapComm.getSize();
    d_count      = std::vector<int>( commSize, 0 );
    d_displ      = std::vector<int>( commSize, 0 );
    for ( auto &elem : d_localPairsMesh1 ) {
        int rank = elem.second.proc;
        d_count[rank]++;
    }
    for ( auto &elem : d_localPairsMesh2 ) {
        int rank = elem.second.proc;
        d_count[rank]++;
    }
    d_displ = std::vector<int>( commSize, 0 );
    for ( int i = 1; i < commSize; i++ )
        d_displ[i] = d_displ[i - 1] + d_count[i - 1];
    int N_tot = d_displ[commSize - 1] + d_count[commSize - 1];
    // Create the send/recv lists and remote DOF lists for each processor
    std::vector<std::vector<AMP::Mesh::MeshElementID>> send_elements( commSize );
    std::vector<std::vector<AMP::Mesh::MeshElementID>> recv_elements( commSize );
    std::vector<std::vector<AMP::Mesh::MeshElementID>> remote_elements( commSize );
    for ( int i = 0; i < commSize; i++ ) {
        send_elements[i].reserve( d_count[i] );
        recv_elements[i].reserve( d_count[i] );
        remote_elements[i].reserve( d_count[i] );
    }
    for ( auto &elem : d_localPairsMesh1 ) {
        int rank = elem.second.proc;
        send_elements[rank].push_back( elem.first.id );
        recv_elements[rank].push_back( elem.first.id );
        remote_elements[rank].push_back( elem.second.id );
    }
    for ( auto &elem : d_localPairsMesh2 ) {
        int rank = elem.second.proc;
        send_elements[rank].push_back( elem.first.id );
        recv_elements[rank].push_back( elem.first.id );
        remote_elements[rank].push_back( elem.second.id );
    }
    // Sort the send/recv lists by the sending processor's MeshElementID
    for ( int i = 0; i < commSize; i++ ) {
        AMP::Utilities::quicksort( send_elements[i] );
        AMP::Utilities::quicksort( remote_elements[i], recv_elements[i] );
    }
    // Create the final lists and allocate space for the send/recv
    d_sendList   = std::vector<AMP::Mesh::MeshElementID>( N_tot );
    d_recvList   = std::vector<AMP::Mesh::MeshElementID>( N_tot );
    d_sendBuffer = std::vector<double>( N_tot * DofsPerObj, 0 );
    d_recvBuffer = std::vector<double>( N_tot * DofsPerObj, 0 );
    for ( int i = 0; i < commSize; i++ ) {
        int k = d_displ[i];
        for ( size_t j = 0; j < send_elements[i].size(); j++ ) {
            d_sendList[k + j] = send_elements[i][j];
            d_recvList[k + j] = recv_elements[i][j];
        }
    }
}


/****************************************************************
 * Function to compute the pairs of points for each mesh         *
 * We have the option of requiring if all points are matched.    *
 ****************************************************************/
void NodeToNodeMap::createPairs( bool requireAllPaired )
{
    d_localPairsMesh1.resize( 0 );
    d_localPairsMesh2.resize( 0 );

    // For each mesh, get the list of points owned by the current processor
    std::vector<Point> ownedPointsMesh1;
    std::vector<Point> ownedPointsMesh2;
    if ( d_mesh1.get() != nullptr )
        ownedPointsMesh1 = createOwnedPoints( d_iterator1 );
    if ( d_mesh2.get() != nullptr )
        ownedPointsMesh2 = createOwnedPoints( d_iterator2 );


    // Send the list of points on mesh1 to all processors
    int commSize  = d_MapComm.getSize();
    auto send_cnt = (int) ownedPointsMesh1.size();
    std::vector<int> recv_cnt( commSize, 0 );
    std::vector<int> recv_disp( commSize, 0 );
    d_MapComm.allGather( send_cnt, &recv_cnt[0] );
    for ( int i = 1; i < commSize; i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    int N_recv_tot                = recv_disp[commSize - 1] + recv_cnt[commSize - 1];
    std::vector<Point> surfacePts = std::vector<Point>( N_recv_tot );
    d_MapComm.allGather(
        getPtr( ownedPointsMesh1 ), send_cnt, &surfacePts[0], &recv_cnt[0], &recv_disp[0], true );

    // Sort the points for fast searching
    AMP::Utilities::quicksort( surfacePts );

    // Find the points in mesh1 that align with the points owned by the current processor on mesh2
    d_localPairsMesh2.reserve( ownedPointsMesh2.size() );
    for ( auto &elem : ownedPointsMesh2 ) {
        // Search for the point
        int index = AMP::Utilities::findfirst( surfacePts, elem );
        // Check if the point was found
        bool found = index >= 0 && index < (int) surfacePts.size();
        if ( found ) {
            if ( surfacePts[index] != elem ) {
                found = false;
            }
        }
        // Add the pair to the list
        if ( found ) {
            std::pair<Point, Point> pair( elem, surfacePts[index] );
            d_localPairsMesh2.push_back( pair );
        } else if ( requireAllPaired ) {
            AMP_ERROR( "All points are required to be paired, and some points were not found" );
        }
    }
    surfacePts.clear();


    // Send the list of points on mesh2 to all processors
    send_cnt = (int) ownedPointsMesh2.size();
    d_MapComm.allGather( send_cnt, &recv_cnt[0] );
    for ( int i = 1; i < commSize; i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    N_recv_tot = recv_disp[commSize - 1] + recv_cnt[commSize - 1];
    surfacePts = std::vector<Point>( N_recv_tot );
    d_MapComm.allGather(
        getPtr( ownedPointsMesh2 ), send_cnt, &surfacePts[0], &recv_cnt[0], &recv_disp[0], true );

    // Sort the points for fast searching
    AMP::Utilities::quicksort( surfacePts );

    // Find the points in mesh1 that align with the points owned by the current processor on mesh2
    d_localPairsMesh1.reserve( ownedPointsMesh1.size() );
    for ( auto &elem : ownedPointsMesh1 ) {
        // Search for the point
        int index = AMP::Utilities::findfirst( surfacePts, elem );
        // Check if the point was found
        bool found = index >= 0 && index < (int) surfacePts.size();
        if ( found ) {
            if ( surfacePts[index] != elem ) {
                found = false;
            }
        }
        // Add the pair to the list
        if ( found ) {
            std::pair<Point, Point> pair( elem, surfacePts[index] );
            d_localPairsMesh1.push_back( pair );
        } else if ( requireAllPaired ) {
            AMP_ERROR( "All points are required to be paired, and some points were not found" );
        }
    }
    surfacePts.clear();

    // Syncronize and return
    d_MapComm.barrier();
}


/********************************************************
 * Function to create the list of owned points from the  *
 * iterator over the surface nodes                       *
 ********************************************************/
std::vector<NodeToNodeMap::Point>
NodeToNodeMap::createOwnedPoints( const AMP::Mesh::MeshIterator &iterator )
{
    // Create the list of points for each node
    std::vector<Point> surfacePts( iterator.size() );
    int rank = d_MapComm.getRank();
    auto cur = iterator.begin();
    for ( size_t i = 0; i < surfacePts.size(); ++i, ++cur ) {
        // Get the properties of the current element
        auto id  = cur->globalID();
        auto pos = cur->centroid();
        if ( !id.is_local() )
            continue;
        // Create the point
        Point temp;
        temp.id = cur->globalID();
        for ( size_t j = 0; j < pos.size(); ++j )
            temp.pos[j] = pos[j];
        temp.proc     = rank;
        surfacePts[i] = temp;
    }
    // Sort the points
    AMP::Utilities::quicksort( surfacePts );
    return surfacePts;
}


/********************************************************
 * Constructors for Point                                *
 ********************************************************/
NodeToNodeMap::Point::Point() : proc( -1 )
{
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 0.0;
}


/********************************************************
 * Operators for Point                                   *
 * Note: we use a tolerance of 1e-8 for checking points  *
 ********************************************************/
bool NodeToNodeMap::Point::operator==( const Point &rhs ) const
{
    // Two points are == if they share the same position (within tolerance)
    double dist = 0.0;
    for ( size_t i = 0; i != 3; i++ )
        dist += ( rhs.pos[i] - pos[i] ) * ( rhs.pos[i] - pos[i] );
    if ( dist < 1e-16 ) // check the square of the distance (faster without sqrt)
        return true;
    return false;
}
bool NodeToNodeMap::Point::operator!=( const Point &rhs ) const { return !operator==( rhs ); }
bool NodeToNodeMap::Point::operator<( const Point &rhs ) const
{
    // Sort the points based on the x value, y value, then z-value
    for ( size_t i = 0; i != 3; i++ ) {
        if ( ( pos[i] - rhs.pos[i] ) < -1e-8 ) {
            return true;
        }
        if ( ( pos[i] - rhs.pos[i] ) > 1e-8 ) {
            return false;
        }
    }
    return false;
}
bool NodeToNodeMap::Point::operator<=( const Point &rhs ) const
{
    return operator==( rhs ) || operator<( rhs );
}
bool NodeToNodeMap::Point::operator>=( const Point &rhs ) const { return !operator<( rhs ); }
bool NodeToNodeMap::Point::operator>( const Point &rhs ) const { return !operator<=( rhs ); }
} // namespace Operator
} // namespace AMP
