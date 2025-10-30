#include "AMP/vectors/CommunicationList.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorIndexer.h"
#include "AMP/vectors/data/VectorData.h"

#include <iostream>
#include <memory>
#include <vector>


namespace AMP::LinearAlgebra {


/************************************************************************
 * CommunicationListParameters                                           *
 ************************************************************************/
CommunicationListParameters::CommunicationListParameters() { d_localsize = (size_t) -1; }
CommunicationListParameters::CommunicationListParameters( const CommunicationListParameters &rhs )
    : ParameterBase(),
      d_comm( rhs.d_comm ),
      d_localsize( rhs.d_localsize ),
      d_remote_DOFs( rhs.d_remote_DOFs )
{
}


/************************************************************************
 * Constructors                                                          *
 ************************************************************************/
CommunicationList::CommunicationList() : d_partition( { 0 } ) {}
CommunicationList::CommunicationList( std::shared_ptr<const CommunicationListParameters> params )
    : d_comm( params->d_comm )
{
    AMP_ASSERT( !d_comm.isNull() );
    d_partition = d_comm.allGather( params->d_localsize );
    for ( size_t i = 1; i < d_partition.size(); i++ )
        d_partition[i] += d_partition[i - 1];
    d_ReceiveDOFList = params->d_remote_DOFs;
    AMP::Utilities::quicksort( d_ReceiveDOFList );
}
CommunicationList::CommunicationList( size_t local, const AMP_MPI &comm ) : d_comm( comm )
{
    d_partition = d_comm.allGather( local );
    for ( size_t i = 1; i < d_partition.size(); i++ )
        d_partition[i] += d_partition[i - 1];
    const int size  = std::max( d_comm.getSize(), 1 );
    d_ReceiveSizes  = std::vector<int>( size, 0 );
    d_ReceiveDisp   = std::vector<int>( size, 0 );
    d_SendSizes     = std::vector<int>( size, 0 );
    d_SendDisp      = std::vector<int>( size, 0 );
    d_SendDOFList   = {};
    d_initialized   = true;
    d_anyRankRemote = false;
}
CommunicationList::CommunicationList( const std::vector<size_t> &partition, const AMP_MPI &comm )
    : d_comm( comm ), d_partition{ partition }
{
    const int size  = std::max( d_comm.getSize(), 1 );
    d_ReceiveSizes  = std::vector<int>( size, 0 );
    d_ReceiveDisp   = std::vector<int>( size, 0 );
    d_SendSizes     = std::vector<int>( size, 0 );
    d_SendDisp      = std::vector<int>( size, 0 );
    d_SendDOFList   = {};
    d_initialized   = true;
    d_anyRankRemote = false;
}

CommunicationList::CommunicationList( const AMP_MPI &comm,
                                      std::vector<size_t> local,
                                      std::vector<size_t> remote )
    : d_comm( comm ), d_ReceiveDOFList( std::move( remote ) ), d_partition( std::move( local ) )
{
    AMP_ASSERT( (int) d_partition.size() == d_comm.getSize() );
    for ( size_t i = 1; i < d_partition.size(); i++ )
        d_partition[i] += d_partition[i - 1];
    AMP::Utilities::quicksort( d_ReceiveDOFList );
    d_anyRankRemote = true;
}

std::shared_ptr<CommunicationList> CommunicationList::getNoCommunicationList()
{
    return std::make_shared<CommunicationList>( d_partition, d_comm );
}

/************************************************************************
 * Subset                                                                *
 ************************************************************************/
std::shared_ptr<CommunicationList> CommunicationList::subset( std::shared_ptr<VectorIndexer> ndx )
{
    if ( !d_initialized )
        initialize();
    // Create the parameters for the subset
    auto params           = std::make_shared<CommunicationListParameters>();
    params->d_comm        = d_comm;
    params->d_localsize   = 0;
    params->d_remote_DOFs = {};
    // Count the number of local dofs in the subset
    for ( size_t i = getStartGID(); i != getStartGID() + numLocalRows(); i++ ) {
        if ( ndx->isInSub( i ) )
            params->d_localsize++;
    }
    // Get the remote dofs to keep in the subset
    params->d_remote_DOFs.reserve( d_SendDOFList.size() );
    for ( size_t i = 0; i < d_SendDOFList.size(); i++ ) {
        size_t dof = d_SendDOFList[i];
        if ( ndx->isInSub( dof ) ) {
            params->d_remote_DOFs.push_back( ndx->getSubID( dof ) );
        }
    }
    return std::make_shared<CommunicationList>( params );
}


/************************************************************************
 * Get ids                                                               *
 ************************************************************************/
size_t CommunicationList::getLocalGhostID( size_t GID ) const
{
    // Search d_ReceiveDOFList for GID
    // Note: d_ReceiveDOFList must be sorted for this to work
    AMP_INSIST( !d_ReceiveDOFList.empty(),
                "Tried to access ghost entry, but vector does not contain ghosts" );
    size_t pos = AMP::Utilities::findfirst( d_ReceiveDOFList, (size_t) GID );
    bool found = pos < d_ReceiveDOFList.size();
    if ( found ) {
        if ( d_ReceiveDOFList[pos] != GID )
            found = false;
    }
    if ( !found ) {
        std::cout << "GID = " << GID << std::endl;
        AMP_ERROR( "GID was not found in the ghost list" );
    }
    return pos;
}


/************************************************************************
 * Build the arrays                                                      *
 ************************************************************************/
void CommunicationList::initialize() const
{
    if ( d_initialized )
        return;
    d_initialized   = true;
    d_anyRankRemote = false;

    // Allocate initial data
    const int size = std::max( d_comm.getSize(), 1 );
    AMP_ASSERT( (int) d_partition.size() == size );
    d_ReceiveSizes = std::vector<int>( size, 0 );
    d_ReceiveDisp  = std::vector<int>( size, 0 );
    d_SendSizes    = std::vector<int>( size, 0 );
    d_SendDisp     = std::vector<int>( size, 0 );
    d_SendDOFList  = {};

    // Check if we are working in serial
    if ( size <= 1 ) {
        AMP_INSIST( d_ReceiveDOFList.empty(),
                    "Error in communication list, remote DOFs are present for a serial vector" );
        return;
    }

    // Check d_ReceiveDOFList
    if ( !d_ReceiveDOFList.empty() )
        AMP_ASSERT( d_ReceiveDOFList.back() < d_partition.back() );
    for ( size_t i = 1; i < d_ReceiveDOFList.size(); i++ )
        AMP_ASSERT( d_ReceiveDOFList[i] >= d_ReceiveDOFList[i - 1] );

    // Determine the number of DOFs received from each processor (requires DOFs to be sorted)
    for ( size_t i = 0, j = 0; i < d_ReceiveDOFList.size(); i++ ) {
        while ( d_ReceiveDOFList[i] >= d_partition[j] )
            j++;
        d_ReceiveSizes[j]++;
    }
    AMP_ASSERT( d_ReceiveSizes[d_comm.getRank()] == 0 );
    for ( int i = 1; i < size; i++ )
        d_ReceiveDisp[i] = d_ReceiveDisp[i - 1] + d_ReceiveSizes[i - 1];

    // Get the send sizes and displacements
    d_SendSizes.resize( size, 0 );
    d_SendDisp.resize( size, 0 );
    d_comm.allToAll( 1, &( d_ReceiveSizes[0] ), &( d_SendSizes[0] ) );
    d_SendDisp[0] = 0;
    for ( int i = 1; i < size; i++ )
        d_SendDisp[i] = d_SendDisp[i - 1] + d_SendSizes[i - 1];
    size_t send_buf_size = d_SendDisp[size - 1] + d_SendSizes[size - 1];

    // Get the send DOFs (requires the recv DOFs to be sorted)
    d_SendDOFList.resize( send_buf_size );
    d_SendDOFList =
        d_comm.allToAll( d_ReceiveDOFList, d_ReceiveSizes, d_ReceiveDisp, d_SendSizes, d_SendDisp );

    // Check if there is any communication
    for ( int i = 0; i < size; i++ )
        d_anyRankRemote = d_anyRankRemote || d_ReceiveSizes[i] > 0 || d_SendSizes[i] > 0;
}
bool CommunicationList::anyCommunication()
{
    if ( !d_initialized )
        initialize();
    return d_anyRankRemote;
}


/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t CommunicationList::getID() const
{
    return getComm().bcast( reinterpret_cast<uint64_t>( this ), 0 );
}


/************************************************************************
 * Get the local start/size, and total size                              *
 ************************************************************************/
size_t CommunicationList::getStartGID() const
{
    int rank = d_comm.getRank();
    if ( rank == 0 )
        return 0;
    return d_partition[rank - 1];
}
size_t CommunicationList::numLocalRows() const
{
    int rank = d_comm.getRank();
    if ( rank == 0 )
        return d_partition[0];
    return d_partition[rank] - d_partition[rank - 1];
}
size_t CommunicationList::getTotalSize() const { return d_partition.back(); }


/************************************************************************
 * Misc. functions                                                       *
 ************************************************************************/
void CommunicationList::clearBuffers()
{
    d_ReceiveSizes.clear();
    d_ReceiveDisp.clear();
    d_SendSizes.clear();
    d_SendDisp.clear();
    d_SendDOFList.clear();
    d_ReceiveDOFList.clear();
}

size_t CommunicationList::getVectorSendBufferSize() const
{
    if ( !d_initialized )
        initialize();
    return d_SendDOFList.size();
}
size_t CommunicationList::getVectorReceiveBufferSize() const { return d_ReceiveDOFList.size(); }
const std::vector<size_t> &CommunicationList::getPartition() const { return d_partition; }
const std::vector<size_t> &CommunicationList::getGhostIDList() const { return d_ReceiveDOFList; }
const std::vector<size_t> &CommunicationList::getReplicatedIDList() const
{
    if ( !d_initialized )
        initialize();
    return d_SendDOFList;
}
const std::vector<int> &CommunicationList::getReceiveSizes() const
{
    if ( !d_initialized )
        initialize();
    return d_ReceiveSizes;
}
const std::vector<int> &CommunicationList::getSendSizes() const
{
    if ( !d_initialized )
        initialize();
    return d_SendSizes;
}
const std::vector<int> &CommunicationList::getReceiveDisp() const
{
    if ( !d_initialized )
        initialize();
    return d_ReceiveDisp;
}
const std::vector<int> &CommunicationList::getSendDisp() const
{
    if ( !d_initialized )
        initialize();
    return d_SendDisp;
}
const AMP_MPI &CommunicationList::getComm() const { return d_comm; }


} // namespace AMP::LinearAlgebra


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CommunicationList>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::CommunicationList> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    manager->registerComm( data->getComm() );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CommunicationList>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "commHash", d_data->getComm().hash() );
    writeHDF5( gid, "localsize", d_data->numLocalRows() );
    writeHDF5( gid, "remote_DOFs", d_data->getGhostIDList() );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::CommunicationList>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::CommunicationList>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    std::string type;
    uint64_t commHash;
    auto params = std::make_shared<AMP::LinearAlgebra::CommunicationListParameters>();
    readHDF5( gid, "commHash", commHash );
    readHDF5( gid, "localsize", params->d_localsize );
    readHDF5( gid, "remote_DOFs", params->d_remote_DOFs );
    params->d_comm = manager->getComm( commHash );
    closeGroup( gid );
    return std::make_shared<AMP::LinearAlgebra::CommunicationList>( params );
}
