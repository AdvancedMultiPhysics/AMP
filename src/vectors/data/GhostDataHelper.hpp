#ifndef included_AMP_GhostDataHelper_hpp
#define included_AMP_GhostDataHelper_hpp

#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/data/GhostDataHelper.h"

namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<class TYPE, class Allocator>
GhostDataHelper<TYPE, Allocator>::GhostDataHelper()
    : d_UpdateState{ std::make_shared<UpdateState>() }
{
    *d_UpdateState = UpdateState::UNCHANGED;
}
template<class TYPE, class Allocator>
GhostDataHelper<TYPE, Allocator>::GhostDataHelper( std::shared_ptr<CommunicationList> list )
    : d_UpdateState{ std::make_shared<UpdateState>() }
{
    *d_UpdateState = UpdateState::UNCHANGED;
    setCommunicationList( list );
}


/****************************************************************
 * Get/Set the communication list                                *
 ****************************************************************/
template<class TYPE, class Allocator>
std::shared_ptr<CommunicationList> GhostDataHelper<TYPE, Allocator>::getCommunicationList() const
{
    return d_CommList;
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::setCommunicationList(
    std::shared_ptr<CommunicationList> comm )
{
    AMP_ASSERT( comm );
    d_CommList = comm;
    if ( d_CommList ) {
        this->d_Ghosts.resize( d_CommList->getVectorReceiveBufferSize() );
        this->d_AddBuffer.resize( d_CommList->getVectorReceiveBufferSize() );
    }
}


/****************************************************************
 * makeConsistent / UpdateState                                  *
 ****************************************************************/
template<class TYPE, class Allocator>
UpdateState GhostDataHelper<TYPE, Allocator>::getLocalUpdateStatus() const
{
    return *d_UpdateState;
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::setUpdateStatus( UpdateState state )
{
    *d_UpdateState = state;
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::setUpdateStatusPtr( std::shared_ptr<UpdateState> rhs )
{
    d_UpdateState = rhs;
}
template<class TYPE, class Allocator>
std::shared_ptr<UpdateState> GhostDataHelper<TYPE, Allocator>::getUpdateStatusPtr() const
{
    return d_UpdateState;
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::dataChanged()
{
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    fireDataChange();
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::makeConsistent( ScatterType t )
{
    PROFILE( "makeConsistent" );
    if ( d_CommList ) {
        if ( t == ScatterType::CONSISTENT_ADD ) {
            AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
            scatter_add();
            for ( auto &elem : this->d_AddBuffer )
                elem = 0.0;
        }
        *d_UpdateState = UpdateState::SETTING;
        scatter_set();
        *d_UpdateState = UpdateState::UNCHANGED;
    }
    this->setUpdateStatus( UpdateState::UNCHANGED );
}


/************************************************************************
 * set/recv data                                                         *
 ************************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::scatter_set()
{
    AMP_ASSERT( d_CommList );
    const auto &sendSizes = d_CommList->getSendSizes();
    const auto &recvSizes = d_CommList->getReceiveSizes();
    if ( sendSizes.empty() && recvSizes.empty() )
        return;
    const auto &comm     = d_CommList->getComm();
    const auto &sendDisp = d_CommList->getSendDisp();
    const auto &recvDisp = d_CommList->getReceiveDisp();
    const auto &sendDOFs = d_CommList->getReplicatedIDList();
    const auto &recvDOFs = d_CommList->getGhostIDList();
    // Pack the set buffers
    std::vector<TYPE> send( d_CommList->getVectorSendBufferSize() );
    if ( !send.empty() )
        getLocalValuesByGlobalID( send.size(), sendDOFs.data(), send.data() );
    // Communicate
    auto recv = comm.allToAll( send, sendSizes, sendDisp, recvSizes, recvDisp );
    // Unpack the set buffers
    if ( !recv.empty() )
        setGhostValuesByGlobalID( recv.size(), recvDOFs.data(), recv.data() );
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::scatter_add()
{
    AMP_ASSERT( d_CommList );
    const auto &sendSizes = d_CommList->getSendSizes();
    const auto &recvSizes = d_CommList->getReceiveSizes();
    if ( sendSizes.empty() && recvSizes.empty() )
        return;
    const auto &comm     = d_CommList->getComm();
    const auto &sendDisp = d_CommList->getSendDisp();
    const auto &recvDisp = d_CommList->getReceiveDisp();
    const auto &sendDOFs = d_CommList->getReplicatedIDList();
    const auto &recvDOFs = d_CommList->getGhostIDList();
    // Pack the add buffers
    std::vector<TYPE> send( d_CommList->getVectorReceiveBufferSize() );
    if ( !send.empty() )
        getGhostAddValuesByGlobalID( send.size(), recvDOFs.data(), send.data() );
    // Communicate
    auto recv = comm.allToAll( send, recvSizes, recvDisp, sendSizes, sendDisp );
    // Unpack the add buffers
    if ( !recv.empty() )
        addLocalValuesByGlobalID( recv.size(), sendDOFs.data(), recv.data() );
}


/****************************************************************
 * Get ghost size                                                *
 ****************************************************************/
template<class TYPE, class Allocator>
size_t GhostDataHelper<TYPE, Allocator>::getGhostSize() const
{
    return this->d_Ghosts.size();
}


/****************************************************************
 * Alias ghost buffer                                            *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::aliasGhostBuffer(
    [[maybe_unused]] std::shared_ptr<VectorData> in )
{
    AMP_ERROR( "Not finished" );
#if 0
    auto ghostData = std::dynamic_pointer_cast<GhostDataHelper<TYPE, Allocator>>( in );
    AMP_ASSERT( ghostData );
    this->d_Ghosts = ghostData->d_Ghosts;
#endif
}


/****************************************************************
 * Zero ghost buffers                                            *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::fillGhosts( const Scalar &scalar )
{
    const auto y = static_cast<TYPE>( scalar );
    std::fill( this->d_Ghosts.begin(), this->d_Ghosts.end(), y );
    std::fill( this->d_AddBuffer.begin(), this->d_AddBuffer.end(), static_cast<TYPE>( 0 ) );
}


/****************************************************************
 * Clear ghost buffers                                            *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::setNoGhosts()
{
    this->d_Ghosts.clear();
    this->d_AddBuffer.clear();
    if ( this->d_CommList ) {
        this->d_CommList->clearBuffers();
    }
}


/****************************************************************
 * Check if vector contains a particular element                 *
 ****************************************************************/
template<class TYPE, class Allocator>
bool GhostDataHelper<TYPE, Allocator>::containsGlobalElement( size_t i ) const
{
    if ( ( i >= d_CommList->getStartGID() ) &&
         ( i < d_CommList->getStartGID() + d_CommList->numLocalRows() ) )
        return true;
    return std::find( d_CommList->getGhostIDList().begin(),
                      d_CommList->getGhostIDList().end(),
                      i ) != d_CommList->getGhostIDList().end();
}


/****************************************************************
 * Get/Set ghost values by global id                             *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::setGhostValuesByGlobalID( size_t N,
                                                                 const size_t *ndx,
                                                                 const void *vals,
                                                                 const typeID &id )
{
    if ( id == AMP::getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
        *d_UpdateState = UpdateState::SETTING;
        for ( size_t i = 0; i < N; i++ ) {
            if ( ( ndx[i] < d_localStart ) || ( ndx[i] >= ( d_localStart + d_localSize ) ) ) {
                this->d_Ghosts[d_CommList->getLocalGhostID( ndx[i] )] = data[i];
            } else {
                AMP_ERROR( "Non ghost index" );
            }
        }
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::addGhostValuesByGlobalID( size_t N,
                                                                 const size_t *ndx,
                                                                 const void *vals,
                                                                 const typeID &id )
{
    if ( id == AMP::getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
        *d_UpdateState = UpdateState::ADDING;
        for ( size_t i = 0; i < N; i++ ) {
            if ( ( ndx[i] < d_localStart ) || ( ndx[i] >= ( d_localStart + d_localSize ) ) ) {
                this->d_AddBuffer[d_CommList->getLocalGhostID( ndx[i] )] += data[i];
            } else {
                AMP_ERROR( "Non ghost index" );
            }
        }
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::getGhostValuesByGlobalID( size_t N,
                                                                 const size_t *ndx,
                                                                 void *vals,
                                                                 const typeID &id ) const
{
    if ( id == AMP::getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<TYPE *>( vals );
        for ( size_t i = 0; i < N; i++ ) {
            if ( ( ndx[i] < d_localStart ) || ( ndx[i] >= ( d_localStart + d_localSize ) ) ) {
                data[i] = this->d_Ghosts[d_CommList->getLocalGhostID( ndx[i] )] +
                          this->d_AddBuffer[d_CommList->getLocalGhostID( ndx[i] )];
            } else {
                AMP_ERROR( "Tried to get a non-ghost ghost value" );
            }
        }
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::getGhostAddValuesByGlobalID( size_t N,
                                                                    const size_t *ndx,
                                                                    void *vals,
                                                                    const typeID &id ) const
{
    if ( id == AMP::getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<TYPE *>( vals );
        for ( size_t i = 0; i < N; i++ ) {
            if ( ( ndx[i] < d_localStart ) || ( ndx[i] >= ( d_localStart + d_localSize ) ) ) {
                data[i] = this->d_AddBuffer[d_CommList->getLocalGhostID( ndx[i] )];
            } else {
                AMP_ERROR( "Tried to get a non-ghost ghost value" );
            }
        }
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}


/****************************************************************
 * dump data to ostream                                          *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::dumpGhostedData( std::ostream &out, size_t offset ) const
{
    if ( !getCommunicationList() )
        return;
    const std::vector<size_t> &ghosts = getCommunicationList()->getGhostIDList();
    auto curVal                       = this->d_Ghosts.begin();
    for ( auto &ghost : ghosts ) {
        out << "  GID: " << ( ghost + offset ) << "  Value: " << ( *curVal ) << "\n";
        ++curVal;
    }
}


/****************************************************************
 * Default clone                                                 *
 ****************************************************************/
template<class TYPE, class Allocator>
const AMP_MPI &GhostDataHelper<TYPE, Allocator>::getComm() const
{
    AMP_ASSERT( d_CommList );
    return d_CommList->getComm();
}


/****************************************************************
 * dump data to ostream                                          *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::copyGhostValues( const VectorData &rhs )
{
    if ( getGhostSize() == 0 ) {
        // No ghosts to fill, copy the consistency state from the rhs
        *d_UpdateState = rhs.getLocalUpdateStatus();
    } else if ( getGhostSize() == rhs.getGhostSize() ) {
        // The ghosts in the src vector match the current vector
        // Copy the ghosts from the rhs
        auto ghostIDs = getCommunicationList()->getGhostIDList();
        std::vector<TYPE> values( ghostIDs.size() );
        rhs.getGhostValuesByGlobalID( ghostIDs.size(), &ghostIDs[0], values.data() );
        this->setGhostValuesByGlobalID( ghostIDs.size(), &ghostIDs[0], values.data() );
        // Copy the consistency state from the rhs
        *d_UpdateState = rhs.getLocalUpdateStatus();
    } else {
        // We can't copy the ghosts from the rhs
        // Use makeConsistent to fill the ghosts
        // Note: this will insure global communication
        *d_UpdateState = rhs.getLocalUpdateStatus();
        if ( *d_UpdateState == UpdateState::UNCHANGED )
            *d_UpdateState = UpdateState::LOCAL_CHANGED;
    }
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::registerChildObjects(
    AMP::IO::RestartManager *manager ) const
{
    VectorData::registerChildObjects( manager );
    if ( d_CommList ) {
        auto id = manager->registerObject( d_CommList );
        AMP_ASSERT( id == d_CommList->getID() );
    }
    if ( d_UpdateState ) {
        auto id = manager->registerObject( d_UpdateState );
        AMP_ASSERT( id == reinterpret_cast<uint64_t>( d_UpdateState.get() ) );
    }
}
template<class TYPE, class Allocator>
void GhostDataHelper<TYPE, Allocator>::writeRestart( int64_t fid ) const
{
    VectorData::writeRestart( fid );
    uint64_t commListID = d_CommList ? d_CommList->getID() : 0;
    uint64_t updateID   = reinterpret_cast<uint64_t>( d_UpdateState.get() );
    IO::writeHDF5( fid, "commListID", commListID );
    IO::writeHDF5( fid, "updateID", updateID );
    IO::writeHDF5( fid, "ghosts", this->d_Ghosts );
    IO::writeHDF5( fid, "addBuffer", this->d_AddBuffer );
}
template<class TYPE, class Allocator>
GhostDataHelper<TYPE, Allocator>::GhostDataHelper( int64_t fid, AMP::IO::RestartManager *manager )
    : VectorData( fid, manager )
{
    uint64_t commListID, updateID;
    IO::readHDF5( fid, "ghosts", this->d_Ghosts );
    IO::readHDF5( fid, "addBuffer", this->d_AddBuffer );

    IO::readHDF5( fid, "commListID", commListID );
    IO::readHDF5( fid, "updateID", updateID );
    if ( commListID != 0 )
        d_CommList = manager->getData<CommunicationList>( commListID );
    if ( updateID != 0 )
        d_UpdateState = manager->getData<UpdateState>( updateID );
}


} // namespace AMP::LinearAlgebra


#endif
