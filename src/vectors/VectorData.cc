#include "VectorData.h"


namespace AMP {
namespace LinearAlgebra {


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


} // LinearAlgebra namespace
} // AMP namespace

