#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/DataChangeListener.h"


namespace AMP {
namespace LinearAlgebra {

VectorData::VectorData() : d_UpdateState { std::make_shared<UpdateState>() },
                           d_Ghosts{ std::make_shared<std::vector<double>>() },
			   d_AddBuffer { std::make_shared<std::vector<double>>() }
{
    *d_UpdateState = UpdateState::UNCHANGED;
}

void VectorData::setCommunicationList( CommunicationList::shared_ptr comm )
{
    AMP_ASSERT( comm );
    d_CommList = comm;
    if ( comm ) {
        d_Ghosts =
            std::make_shared<std::vector<double>>( d_CommList->getVectorReceiveBufferSize() );
        d_AddBuffer =
            std::make_shared<std::vector<double>>( d_CommList->getVectorReceiveBufferSize() );
    }
}
  
/****************************************************************
 * Set/Get individual values                                     *
 ****************************************************************/
void VectorData::setValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
    *d_UpdateState = UpdateState::SETTING;
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            ( *d_Ghosts )[d_CommList->getLocalGhostID( ndx[i] )] = vals[i];
        } else {
            setLocalValuesByGlobalID( 1, ndx + i, vals + i );
        }
    }
}
void VectorData::setGhostValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::ADDING );
    *d_UpdateState = UpdateState::SETTING;
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            ( *d_Ghosts )[d_CommList->getLocalGhostID( ndx[i] )] = vals[i];
        } else {
            AMP_ERROR( "Non ghost index" );
        }
    }
}
void VectorData::addValuesByGlobalID( int numVals, size_t *ndx, const double *vals )
{
    AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
    *d_UpdateState = UpdateState::ADDING;
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            ( *d_AddBuffer )[d_CommList->getLocalGhostID( ndx[i] )] += vals[i];
        } else {
            addLocalValuesByGlobalID( 1, ndx + i, vals + i );
        }
    }
}
void VectorData::getValuesByLocalID( int num, size_t *ndx, double *vals ) const
{
    for ( int i = 0; i != num; i++ ) {
        size_t block_number = 0;
        size_t offset       = ndx[i];
        while ( offset >= sizeOfDataBlock( block_number ) ) {
            offset -= sizeOfDataBlock( block_number );
            block_number++;
            if ( block_number >= numberOfDataBlocks() ) {
                AMP_ERROR( "Bad local id!" );
            }
        }
        vals[i] = getRawDataBlock<double>( block_number )[offset];
    }
}
void VectorData::getValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            getGhostValuesByGlobalID( 1, ndx + i, vals + i );
        } else {
            getLocalValuesByGlobalID( 1, ndx + i, vals + i );
        }
    }
}
void VectorData::getGhostValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            vals[i] = ( *d_Ghosts )[d_CommList->getLocalGhostID( ndx[i] )] +
                      ( *d_AddBuffer )[d_CommList->getLocalGhostID( ndx[i] )];
        } else {
            AMP_ERROR( "Tried to get a non-ghost ghost value" );
        }
    }
}
void VectorData::getGhostAddValuesByGlobalID( int numVals, size_t *ndx, double *vals ) const
{
    for ( int i = 0; i < numVals; i++ ) {
        if ( ( ndx[i] < getLocalStartID() ) ||
             ( ndx[i] >= ( getLocalStartID() + getLocalMaxID() ) ) ) {
            vals[i] = ( *d_AddBuffer )[d_CommList->getLocalGhostID( ndx[i] )];
        } else {
            AMP_ERROR( "Tried to get a non-ghost ghost value" );
        }
    }
}


/****************************************************************
 * makeConsistent                                                *
 ****************************************************************/
void VectorData::makeConsistent( ScatterType t )
{
    if ( t == ScatterType::CONSISTENT_ADD ) {
        AMP_ASSERT( *d_UpdateState != UpdateState::SETTING );
        std::vector<double> send_vec_add( d_CommList->getVectorReceiveBufferSize() );
        std::vector<double> recv_vec_add( d_CommList->getVectorSendBufferSize() );
        d_CommList->packReceiveBuffer( send_vec_add, *this );
        d_CommList->scatter_add( send_vec_add, recv_vec_add );
        d_CommList->unpackSendBufferAdd( recv_vec_add, *this );
        for ( auto &elem : *d_AddBuffer ) {
            elem = 0.0;
        }
    }
    *d_UpdateState = UpdateState::SETTING;
    std::vector<double> send_vec( d_CommList->getVectorSendBufferSize() );
    std::vector<double> recv_vec( d_CommList->getVectorReceiveBufferSize() );
    d_CommList->packSendBuffer( send_vec, *this );
    d_CommList->scatter_set( send_vec, recv_vec );
    d_CommList->unpackReceiveBufferSet( recv_vec, *this );
    *d_UpdateState = UpdateState::UNCHANGED;
    this->setUpdateStatus( UpdateState::UNCHANGED );
}


/****************************************************************
 * dataChanged                                                   *
 ****************************************************************/
void VectorData::dataChanged()
{
    if ( *d_UpdateState == UpdateState::UNCHANGED )
        *d_UpdateState = UpdateState::LOCAL_CHANGED;
    fireDataChange();
}


/****************************************************************
 * Default clone                                                 *
 ****************************************************************/
std::shared_ptr<VectorData> VectorData::cloneData() const { return std::shared_ptr<VectorData>(); }
AMP_MPI VectorData::getComm() const
{
    AMP_ASSERT( d_CommList );
    return d_CommList->getComm();
}


/****************************************************************
 * dump data to ostream                                          *
 ****************************************************************/
void VectorData::dumpOwnedData( std::ostream &out, size_t GIDoffset, size_t LIDoffset ) const
{
    auto curElement = begin();
    size_t gid      = GIDoffset;
    if ( getCommunicationList() )
        gid += getCommunicationList()->getStartGID();
    size_t lid = LIDoffset;
    while ( curElement != end() ) {
        out << "  GID: " << gid << "  LID: " << lid << "  Value: " << *curElement << "\n";
        ++curElement;
        ++gid;
        ++lid;
    }
}
void VectorData::dumpGhostedData( std::ostream &out, size_t offset ) const
{
    if ( !getCommunicationList() )
        return;
    const std::vector<size_t> &ghosts = getCommunicationList()->getGhostIDList();
    auto curVal                       = d_Ghosts->begin();
    for ( auto &ghost : ghosts ) {
        out << "  GID: " << ( ghost + offset ) << "  Value: " << ( *curVal ) << "\n";
        ++curVal;
    }
}


/****************************************************************
 * dump data to ostream                                          *
 ****************************************************************/
void VectorData::copyGhostValues( const VectorData &rhs )
{
    if ( getGhostSize() == 0 ) {
        // No ghosts to fill, we don't need to do anything
    } else if ( getGhostSize() == rhs.getGhostSize() ) {
        // The ghosts in the src vector match the current vector
        // Copy the ghosts from the rhs
        std::vector<size_t> ghostIDs = getCommunicationList()->getGhostIDList();
        std::vector<double> values( ghostIDs.size() );
        rhs.getGhostValuesByGlobalID( ghostIDs.size(), &ghostIDs[0], &values[0] );
        this->setGhostValuesByGlobalID( ghostIDs.size(), &ghostIDs[0], &values[0] );
        // Copy the consistency state from the rhs
        *d_UpdateState = *( rhs.getUpdateStatusPtr() );
    } else {
        // We can't copy the ghosts from the rhs
        // Use makeConsistent to fill the ghosts
        // Note: this will incure global communication
        *d_UpdateState = *( rhs.getUpdateStatusPtr() );
        if ( *d_UpdateState == UpdateState::UNCHANGED )
            *d_UpdateState = UpdateState::LOCAL_CHANGED;
    }
}

} // namespace LinearAlgebra
} // namespace AMP