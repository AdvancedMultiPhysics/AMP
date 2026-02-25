#ifndef included_AMP_VectorDataDefault_hpp
#define included_AMP_VectorDataDefault_hpp

#include "AMP/IO/RestartManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include <cstring>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


// Suppresses implicit instantiation below
extern template class VectorDataDefault<double>;
extern template class VectorDataDefault<float>;


/****************************************************************
 * Get the class type                                            *
 ****************************************************************/
template<typename TYPE, class Allocator>
std::string VectorDataDefault<TYPE, Allocator>::VectorDataName() const
{
    constexpr typeID id = getTypeID<TYPE>();
    constexpr AMP::Utilities::MemoryType allocMemType =
        AMP::Utilities::getAllocatorMemoryType<Allocator>();

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::host ) {
        return "VectorDataDefault<" + std::string( id.name ) + ">";
    }

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::managed ) {
        return "VectorDataDefault<" + std::string( id.name ) + ",AMP::ManagedAllocator>";
    }

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::device ) {
        return "VectorDataDefault<" + std::string( id.name ) + ",AMP::DeviceAllocator>";
    }

    return "VectorDataDefault<" + std::string( id.name ) + ",UnknownAllocator>";
}


/****************************************************************
 * Allocate the data                                             *
 ****************************************************************/
template<typename TYPE, class Allocator>
VectorDataDefault<TYPE, Allocator>::VectorDataDefault( size_t start,
                                                       size_t localSize,
                                                       size_t globalSize )
{
    PROFILE( "VectorDataDefault::constructor" );

    static_assert( std::is_same_v<typename Allocator::value_type, void> );
    this->d_localSize  = localSize;
    this->d_globalSize = globalSize;
    this->d_localStart = start;
    this->d_data       = this->d_alloc.allocate( localSize );
    AMP::Utilities::memset( this->d_data, 0, localSize * sizeof( TYPE ) );
}

template<typename TYPE, class Allocator>
VectorDataDefault<TYPE, Allocator>::~VectorDataDefault()
{
    for ( size_t i = 0; i < this->d_localSize; ++i )
        this->d_data[i].~TYPE();
    this->d_alloc.deallocate( this->d_data, this->d_localSize );
}


/****************************************************************
 * Clone the data                                                *
 ****************************************************************/
template<typename TYPE, class Allocator>
std::shared_ptr<VectorData>
VectorDataDefault<TYPE, Allocator>::cloneData( const std::string & ) const
{
    PROFILE( "VectorDataDefault::cloneData" );

    auto retVal = std::make_shared<VectorDataDefault<TYPE, Allocator>>(
        this->d_localStart, this->d_localSize, this->d_globalSize );
    auto comm = this->getCommunicationList();
    if ( comm )
        retVal->setCommunicationList( comm );

    if ( this->hasGhosts() ) {
        retVal->allocateBuffers( this->d_ghostSize );
        retVal->copyGhostValues( *this );
        for ( size_t i = 0; i < this->d_ghostSize; ++i ) {
            retVal->d_AddBuffer[i] = this->d_AddBuffer[i];
        }
    }

    return retVal;
}

/****************************************************************
 * Return basic properties                                       *
 ****************************************************************/
template<typename TYPE, class Allocator>
inline size_t VectorDataDefault<TYPE, Allocator>::numberOfDataBlocks() const
{
    return 1;
}
template<typename TYPE, class Allocator>
inline size_t VectorDataDefault<TYPE, Allocator>::sizeOfDataBlock( size_t i ) const
{
    if ( i > 0 )
        return 0;
    return this->d_localSize;
}
template<typename TYPE, class Allocator>
uint64_t VectorDataDefault<TYPE, Allocator>::getDataID() const
{
    return reinterpret_cast<uint64_t>( this->d_data );
}
template<typename TYPE, class Allocator>
typeID VectorDataDefault<TYPE, Allocator>::getType( size_t ) const
{
    constexpr auto type = getTypeID<TYPE>();
    return type;
}
template<typename TYPE, class Allocator>
size_t VectorDataDefault<TYPE, Allocator>::sizeofDataBlockType( size_t ) const
{
    return sizeof( TYPE );
}

template<typename TYPE, class Allocator>
AMP::Utilities::MemoryType VectorDataDefault<TYPE, Allocator>::getMemoryLocation() const
{
    return AMP::Utilities::getAllocatorMemoryType<Allocator>();
}

/****************************************************************
 * Access the raw data blocks                                    *
 ****************************************************************/
template<typename TYPE, class Allocator>
inline void *VectorDataDefault<TYPE, Allocator>::getRawDataBlockAsVoid( size_t i )
{
    if ( i != 0 ) {
        return 0;
    }
    AMP_DEBUG_ASSERT( AMP::Utilities::getAllocatorMemoryType<Allocator>() ==
                      AMP::Utilities::getMemoryType( this->d_data ) );
    return this->d_data;
}
template<typename TYPE, class Allocator>
inline const void *VectorDataDefault<TYPE, Allocator>::getRawDataBlockAsVoid( size_t i ) const
{
    if ( i != 0 ) {
        return 0;
    }
    AMP_DEBUG_ASSERT( AMP::Utilities::getAllocatorMemoryType<Allocator>() ==
                      AMP::Utilities::getMemoryType( this->d_data ) );
    return this->d_data;
}


/****************************************************************
 * Access individual values                                      *
 ****************************************************************/
template<typename TYPE, class Allocator>
inline TYPE &VectorDataDefault<TYPE, Allocator>::operator[]( size_t i )
{
    return this->d_data[i];
}

template<typename TYPE, class Allocator>
inline const TYPE &VectorDataDefault<TYPE, Allocator>::operator[]( size_t i ) const
{
    return this->d_data[i];
}
template<typename TYPE, class Allocator>
inline void VectorDataDefault<TYPE, Allocator>::setValuesByLocalID( size_t num,
                                                                    const size_t *indices,
                                                                    const void *vals,
                                                                    const typeID &id )
{
    PROFILE( "VectorDataDefault::setValuesByLocalID" );

#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    for ( size_t i = 0; i < num; i++ )
        AMP_ASSERT( indices[i] < this->d_localSize );
#endif
    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        for ( size_t i = 0; i < num; i++ )
            this->d_data[indices[i]] = data[i];
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            this->d_data[indices[i]] = static_cast<TYPE>( data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename TYPE, class Allocator>
inline void VectorDataDefault<TYPE, Allocator>::addValuesByLocalID( size_t num,
                                                                    const size_t *indices,
                                                                    const void *vals,
                                                                    const typeID &id )
{
    PROFILE( "VectorDataDefault::addValuesByLocalID" );

#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    for ( size_t i = 0; i < num; i++ )
        AMP_ASSERT( indices[i] < this->d_localSize );
#endif
    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        for ( size_t i = 0; i < num; i++ )
            this->d_data[indices[i]] += data[i];
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            this->d_data[indices[i]] += static_cast<TYPE>( data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename TYPE, class Allocator>
inline void VectorDataDefault<TYPE, Allocator>::getValuesByLocalID( size_t num,
                                                                    const size_t *indices,
                                                                    void *vals,
                                                                    const typeID &id ) const
{
    PROFILE( "VectorDataDefault::getValuesByLocalID" );

#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    for ( size_t i = 0; i < num; i++ )
        AMP_ASSERT( indices[i] < this->d_localSize );
#endif
    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<TYPE *>( vals );
        for ( size_t i = 0; i < num; i++ )
            data[i] = this->d_data[indices[i]];
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<double *>( vals );
        for ( size_t i = 0; i < num; ++i )
            data[i] = this->d_data[indices[i]];
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


/****************************************************************
 * Copy raw data                                                 *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDefault<TYPE, Allocator>::putRawData( const void *in, const typeID &id )
{
    PROFILE( "VectorDataDefault::putRawData" );

    if ( id == getTypeID<TYPE>() ) {
        memcpy( this->d_data, in, this->d_localSize * sizeof( TYPE ) );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( in );
        for ( size_t i = 0; i < this->d_localSize; ++i )
            this->d_data[i] = static_cast<TYPE>( data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename TYPE, class Allocator>
void VectorDataDefault<TYPE, Allocator>::getRawData( void *out, const typeID &id ) const
{
    PROFILE( "VectorDataDefault::getRawData" );

    if ( id == getTypeID<TYPE>() ) {
        memcpy( out, this->d_data, this->d_localSize * sizeof( TYPE ) );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<double *>( out );
        for ( size_t i = 0; i < this->d_localSize; ++i )
            data[i] = static_cast<double>( this->d_data[i] );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}


/****************************************************************
 * Swap raw data                                                 *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDefault<TYPE, Allocator>::swapData( VectorData &rhs )
{
    PROFILE( "VectorDataDefault::swapData" );

    auto rhs2 = dynamic_cast<VectorDataDefault<TYPE, Allocator> *>( &rhs );
    AMP_INSIST( rhs2, "Cannot swap with arbitrary VectorData" );
    std::swap( this->d_CommList, rhs2->d_CommList );
    std::swap( this->d_UpdateState, rhs2->d_UpdateState );
    std::swap( this->d_ghostSize, rhs2->d_ghostSize );
    std::swap( this->d_Ghosts, rhs2->d_Ghosts );
    std::swap( this->d_AddBuffer, rhs2->d_AddBuffer );
    std::swap( this->d_data, rhs2->d_data );
    std::swap( this->d_localSize, rhs2->d_localSize );
    std::swap( this->d_globalSize, rhs2->d_globalSize );
    std::swap( this->d_localStart, rhs2->d_localStart );
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDefault<TYPE, Allocator>::registerChildObjects(
    AMP::IO::RestartManager *manager ) const
{
    GhostDataHelper<TYPE, Allocator>::registerChildObjects( manager );
}
template<typename TYPE, class Allocator>
void VectorDataDefault<TYPE, Allocator>::writeRestart( int64_t fid ) const
{
    GhostDataHelper<TYPE, Allocator>::writeRestart( fid );
    AMP::Array<TYPE> data( this->d_localSize );
    getRawData( data.data(), getTypeID<TYPE>() );
    IO::writeHDF5( fid, "data", data );
}
template<typename TYPE, class Allocator>
VectorDataDefault<TYPE, Allocator>::VectorDataDefault( int64_t fid,
                                                       AMP::IO::RestartManager *manager )
    : GhostDataHelper<TYPE, Allocator>( fid, manager )
{
    AMP::Array<TYPE> data;
    IO::readHDF5( fid, "data", data );
    d_data = this->d_alloc.allocate( this->d_localSize );
    putRawData( data.data(), getTypeID<TYPE>() );
}


} // namespace AMP::LinearAlgebra

#endif
