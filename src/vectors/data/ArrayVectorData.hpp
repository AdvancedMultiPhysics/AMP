#ifndef included_AMP_ArrayVectorData_hpp
#define included_AMP_ArrayVectorData_hpp
#include "AMP/discretization/DOF_Manager.h"

#include "math.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
std::shared_ptr<ArrayVectorData<T, FUN, Allocator>>
ArrayVectorData<T, FUN, Allocator>::create( const ArraySize &localSize )
{
    return create( localSize, { 0, 0, 0 }, AMP_COMM_SELF );
}
template<typename T, typename FUN, typename Allocator>
std::shared_ptr<ArrayVectorData<T, FUN, Allocator>> ArrayVectorData<T, FUN, Allocator>::create(
    const ArraySize &localSize, const ArraySize &blockIndex, AMP_MPI comm )
{
    PROFILE( "ArrayVectorData::create" );

    // Get the number of blocks
    size_t tmp[10] = { 0 };
    for ( int d = 0; d < 5; d++ ) {
        tmp[d + 0] = blockIndex[d] + 1;
        tmp[d + 5] = localSize[d];
    }
    comm.maxReduce( tmp, 10 );
    AMP::ArraySize N_blocks( blockIndex.ndim(), tmp );
    AMP_ASSERT( N_blocks.length() == (size_t) comm.getSize() );
    for ( int d = 0; d < 5; d++ )
        AMP_INSIST( tmp[d + 5] == localSize[d], "All local blocks must be the same size" );
    // Get the global size and offset
    for ( int d = 0; d < 5; d++ )
        tmp[d] = localSize[d] * N_blocks[d];
    AMP::ArraySize globalSize( localSize.ndim(), tmp );
    size_t blockOffset =
        blockIndex[0] + N_blocks[0] * ( blockIndex[1] + blockIndex[2] * N_blocks[1] );
    // Create the ArrayVectorData
    auto commList = std::make_shared<CommunicationList>( localSize.length(), comm );
    auto retVal   = std::make_shared<ArrayVectorData<T, FUN, Allocator>>();
    retVal->d_array.resize( localSize );
    retVal->d_comm            = comm;
    retVal->d_blockIndex      = blockIndex;
    retVal->d_globalArraySize = globalSize;
    retVal->d_offset          = blockOffset * localSize.length();
    retVal->d_localSize       = localSize.length();
    retVal->d_globalSize      = retVal->d_globalArraySize.length();
    retVal->d_localStart      = commList->getStartGID();
    retVal->setCommunicationList( commList );
    retVal->setUpdateStatus( UpdateState::UNCHANGED );
    return retVal;
}

template<typename T, typename FUN, typename Allocator>
std::shared_ptr<ArrayVectorData<T, FUN, Allocator>> ArrayVectorData<T, FUN, Allocator>::create(
    const size_t localSize, std::shared_ptr<CommunicationList> commList, T *data )
{
    PROFILE( "ArrayVectorData::create" );

    AMP_ASSERT( localSize == commList->numLocalRows() );
    auto retVal = std::make_shared<ArrayVectorData<T, FUN, Allocator>>();
    retVal->setCommunicationList( commList );
    // set the state to be unchanged since setCommunicationList sets
    // it to LOCAL_CHANGED
    retVal->setUpdateStatus( UpdateState::UNCHANGED );
    retVal->d_comm            = commList->getComm();
    retVal->d_localStart      = commList->getStartGID();
    retVal->d_localSize       = commList->numLocalRows();
    retVal->d_globalSize      = commList->getTotalSize();
    retVal->d_blockIndex      = { (uint32_t) retVal->d_comm.getRank() };
    retVal->d_globalArraySize = { retVal->d_globalSize };
    retVal->d_offset          = retVal->d_localStart;
    retVal->d_array.viewRaw( { retVal->d_localSize }, data );
    return retVal;
}


/****************************************************************
 * Clone/Swap data                                               *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
inline std::shared_ptr<VectorData>
ArrayVectorData<T, FUN, Allocator>::cloneData( const std::string & ) const
{
    PROFILE( "ArrayVectorData::cloneData" );

    auto retVal               = std::make_shared<ArrayVectorData<T, FUN, Allocator>>();
    retVal->d_array           = this->d_array;
    retVal->d_comm            = this->d_comm;
    retVal->d_blockIndex      = this->d_blockIndex;
    retVal->d_globalArraySize = this->d_globalSize;
    retVal->d_localSize       = this->d_localSize;
    retVal->d_globalSize      = this->d_globalSize;
    retVal->d_localStart      = this->d_localStart;
    retVal->setCommunicationList( this->getCommunicationList() );
    // set the state to be unchanged since setCommunicationList sets
    // it to LOCAL_CHANGED
    retVal->setUpdateStatus( UpdateState::UNCHANGED );
    return retVal;
}
template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::swapData( VectorData &rhs )
{
    PROFILE( "ArrayVectorData::swapData" );

    // get internal arrays
    auto &internalArray = this->getArray();
    auto &otherArray    = dynamic_cast<ArrayVectorData<T, FUN, Allocator> &>( rhs ).getArray();
    // reset views
    internalArray.swap( otherArray );
}


/****************************************************************
 * Resize data                                                   *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::resize( const ArraySize &localDims )
{
    PROFILE( "ArrayVectorData::resize" );

    AMP_ASSERT( this->getComm().getSize() == 1 );
    this->d_array.resize( localDims );
    this->d_globalArraySize = localDims;
    this->setCommunicationList(
        std::make_shared<CommunicationList>( localDims.length(), this->getComm() ) );
    this->d_localSize  = localDims.length();
    this->d_globalSize = this->d_globalArraySize.length();
    this->d_localStart = this->d_CommList->getStartGID();
}


/****************************************************************
 * Get/Set raw data                                              *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::putRawData( const void *buf, const typeID &id )
{
    PROFILE( "ArrayVectorData::putRawData" );

    auto &array = this->getArray();
    if ( id == getTypeID<T>() ) {
        auto data = reinterpret_cast<const T *>( buf );
        array.copy( data );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( buf );
        array.copy( data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::getRawData( void *buf, const typeID &id ) const
{
    PROFILE( "ArrayVectorData::getRawData" );

    auto &array = this->getArray();
    if ( id == getTypeID<T>() ) {
        auto data = reinterpret_cast<T *>( buf );
        array.copyTo( data );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<double *>( buf );
        array.copyTo( data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


/****************************************************************
 * Get/Set values                                                *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::setValuesByLocalID( size_t num,
                                                             const size_t *indices,
                                                             const void *vals,
                                                             const typeID &id )
{
    PROFILE( "ArrayVectorData::setValuesByLocalID" );

    if ( id == getTypeID<T>() ) {
        auto data = reinterpret_cast<const T *>( vals );
        for ( size_t i = 0; i != num; i++ )
            d_array( indices[i] ) = data[i];
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            d_array( indices[i] ) = static_cast<T>( data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::addValuesByLocalID( size_t num,
                                                             const size_t *indices,
                                                             const void *vals,
                                                             const typeID &id )
{
    PROFILE( "ArrayVectorData::addValuesByLocalID" );

    if ( id == getTypeID<T>() ) {
        auto data = reinterpret_cast<const T *>( vals );
        for ( size_t i = 0; i != num; i++ )
            d_array( indices[i] ) += data[i];
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            d_array( indices[i] ) = static_cast<T>( data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename T, typename FUN, typename Allocator>
void ArrayVectorData<T, FUN, Allocator>::getValuesByLocalID( size_t num,
                                                             const size_t *indices,
                                                             void *vals,
                                                             const typeID &id ) const
{
    PROFILE( "ArrayVectorData::getValuesByLocalID" );

    if ( id == getTypeID<T>() ) {
        auto data = reinterpret_cast<T *>( vals );
        for ( size_t i = 0; i != num; i++ )
            data[i] = d_array( indices[i] );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            data[i] = static_cast<double>( d_array( indices[i] ) );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


} // namespace AMP::LinearAlgebra
#endif
