#ifndef included_AMP_VectorDataDevice_hpp
#define included_AMP_VectorDataDevice_hpp

#include "AMP/IO/RestartManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/data/device/DeviceDataHelpers.h"
#include "AMP/vectors/data/device/VectorDataDevice.h"
#include <cstring>


namespace AMP::LinearAlgebra {


// Suppresses implicit instantiation below
extern template class VectorDataDevice<double>;
extern template class VectorDataDevice<float>;


/****************************************************************
 * Get the class type                                            *
 ****************************************************************/
template<typename TYPE, class Allocator>
std::string VectorDataDevice<TYPE, Allocator>::VectorDataName() const
{
    constexpr typeID id = getTypeID<TYPE>();
    constexpr AMP::Utilities::MemoryType allocMemType =
        AMP::Utilities::getAllocatorMemoryType<Allocator>();

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::host ) {
        return "VectorDataDevice<" + std::string( id.name ) + ">";
    }

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::managed ) {
        return "VectorDataDevice<" + std::string( id.name ) + ",AMP::ManagedAllocator>";
    }

    if constexpr ( allocMemType == AMP::Utilities::MemoryType::device ) {
        return "VectorDataDevice<" + std::string( id.name ) + ",AMP::DeviceAllocator>";
    }

    return "VectorDataDevice<" + std::string( id.name ) + ",UnknownAllocator>";
}


/****************************************************************
 * Allocate the data                                             *
 ****************************************************************/
template<typename TYPE, class Allocator>
VectorDataDevice<TYPE, Allocator>::VectorDataDevice( size_t start,
                                                     size_t localSize,
                                                     size_t globalSize )
    : VectorDataDefault<TYPE, Allocator>( start, localSize, globalSize )
{
}

template<typename TYPE, class Allocator>
VectorDataDevice<TYPE, Allocator>::~VectorDataDevice()
{
}

template<typename T>
bool inDeviceMemory( T *v )
{
    return ( AMP::Utilities::getMemoryType( v ) >= AMP::Utilities::MemoryType::managed );
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::setScratchSpace( const size_t N ) const
{
    if ( N > this->d_scratchSize ) {
        d_idx_alloc.deallocate( this->d_idx_scratch, this->d_scratchSize );
        d_scalar_alloc.deallocate( this->d_scalar_scratch, this->d_scratchSize );
        this->d_scratchSize    = N;
        this->d_idx_scratch    = d_idx_alloc.allocate( this->d_scratchSize );
        this->d_scalar_scratch = d_scalar_alloc.allocate( this->d_scratchSize );
    }
    AMP_ASSERT( d_idx_scratch && d_scalar_scratch );
}

template<typename TYPE, class Allocator>
std::tuple<bool, size_t *, void *> VectorDataDevice<TYPE, Allocator>::copyToScratchSpace(
    size_t num, const size_t *indices_, const void *vals_, const typeID &id ) const
{
    bool scratchUsed = false;
    size_t *indices  = const_cast<size_t *>( indices_ );
    void *vals       = const_cast<void *>( vals_ );

    if ( ( !inDeviceMemory( indices ) ) || ( !inDeviceMemory( vals ) ) ) {
        this->setScratchSpace( num );
        indices = this->d_idx_scratch;
        vals    = this->d_scalar_scratch;
        AMP::Utilities::copy<size_t, size_t>( num, indices_, indices );

        if ( id == getTypeID<TYPE>() ) {
            AMP::Utilities::copy<TYPE, TYPE>(
                num, reinterpret_cast<const TYPE *>( vals_ ), this->d_scalar_scratch );
        } else if ( id == getTypeID<double>() ) {
            AMP::Utilities::copy<double, TYPE>(
                num, reinterpret_cast<const double *>( vals_ ), this->d_scalar_scratch );
        } else if ( id == getTypeID<float>() ) {
            AMP::Utilities::copy<float, TYPE>(
                num, reinterpret_cast<const float *>( vals_ ), this->d_scalar_scratch );
        } else {
            AMP_ERROR( "Conversion not supported yet" );
        }
        scratchUsed = true;
    }

    return std::make_tuple( scratchUsed, indices, vals );
}

template<typename TYPE, class Allocator>
inline void VectorDataDevice<TYPE, Allocator>::setValuesByLocalID( size_t num,
                                                                   const size_t *indices_,
                                                                   const void *vals_,
                                                                   const typeID &id )
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup               = copyToScratchSpace( num, indices_, vals_, id );
    const auto scratchUsed = std::get<0>( tup );
    const auto *indices    = std::get<1>( tup );
    const auto *vals       = std::get<2>( tup );

    if ( id == getTypeID<TYPE>() || scratchUsed ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        DeviceDataHelpers<double, TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<float>() ) {
        auto data = reinterpret_cast<const float *>( vals );
        DeviceDataHelpers<float, TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename TYPE, class Allocator>
inline void VectorDataDevice<TYPE, Allocator>::addValuesByLocalID( size_t num,
                                                                   const size_t *indices_,
                                                                   const void *vals_,
                                                                   const typeID &id )
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup               = copyToScratchSpace( num, indices_, vals_, id );
    const auto scratchUsed = std::get<0>( tup );
    const auto *indices    = std::get<1>( tup );
    const auto *vals       = std::get<2>( tup );

    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<const double *>( vals );
        DeviceDataHelpers<double, TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<float>() ) {
        auto data = reinterpret_cast<const float *>( vals );
        DeviceDataHelpers<float, TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}
template<typename TYPE, class Allocator>
inline void VectorDataDevice<TYPE, Allocator>::getValuesByLocalID( size_t num,
                                                                   const size_t *indices_,
                                                                   void *vals_,
                                                                   const typeID &id ) const
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup         = copyToScratchSpace( num, indices_, vals_, id );
    auto scratchUsed = std::get<0>( tup );
    auto *indices    = std::get<1>( tup );
    auto *vals       = std::get<2>( tup );

    if ( id == getTypeID<TYPE>() || scratchUsed ) {
        auto data = reinterpret_cast<TYPE *>( vals );
        DeviceDataHelpers<TYPE>::getValuesByIndex( num, indices, this->d_data, data );
    } else if ( id == getTypeID<double>() ) {
        auto data = reinterpret_cast<double *>( vals );
        DeviceDataHelpers<TYPE, double>::getValuesByIndex( num, indices, this->d_data, data );
    } else if ( id == getTypeID<float>() ) {
        auto data = reinterpret_cast<float *>( vals );
        DeviceDataHelpers<TYPE, float>::getValuesByIndex( num, indices, this->d_data, data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( scratchUsed ) {
        auto data = reinterpret_cast<TYPE *>( vals );
        if ( id == getTypeID<TYPE>() ) {
            AMP::Utilities::copy<TYPE, TYPE>( num, data, reinterpret_cast<TYPE *>( vals_ ) );
        } else if ( id == getTypeID<double>() ) {
            AMP::Utilities::copy<TYPE, double>( num, data, reinterpret_cast<double *>( vals_ ) );
        } else if ( id == getTypeID<float>() ) {
            AMP::Utilities::copy<TYPE, float>( num, data, reinterpret_cast<float *>( vals_ ) );
        }
    }
}


/****************************************************************
 * Copy raw data                                                 *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::putRawData( const void *in, const typeID &id )
{
    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<const TYPE *>( in );
        AMP::Utilities::Algorithms<TYPE>::copy_n( data, this->d_localSize, this->d_data );
    } else if ( id == getTypeID<double>() ) {
        const auto *data_in = reinterpret_cast<const double *>( in );
        AMP::Utilities::copy<double, TYPE>( this->d_localSize, data_in, this->d_data );
    } else if ( id == getTypeID<float>() ) {
        const auto *data_in = reinterpret_cast<const float *>( in );
        AMP::Utilities::copy<float, TYPE>( this->d_localSize, data_in, this->d_data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getRawData( void *out, const typeID &id ) const
{
    if ( id == getTypeID<TYPE>() ) {
        auto data = reinterpret_cast<TYPE *>( out );
        AMP::Utilities::Algorithms<TYPE>::copy_n( this->d_data, this->d_localSize, data );
    } else if ( id == getTypeID<double>() ) {
        auto *data_out = reinterpret_cast<double *>( out );
        AMP::Utilities::copy<TYPE, double>( this->d_localSize, this->d_data, data_out );
    } else if ( id == getTypeID<float>() ) {
        auto *data_out = reinterpret_cast<float *>( out );
        AMP::Utilities::copy<TYPE, float>( this->d_localSize, this->d_data, data_out );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename TYPE, class Allocator>
std::shared_ptr<VectorData>
VectorDataDevice<TYPE, Allocator>::cloneData( const std::string & ) const
{
    auto retVal = std::make_shared<VectorDataDevice<TYPE, Allocator>>(
        this->d_localStart, this->d_localSize, this->d_globalSize );
    auto comm = this->getCommunicationList();
    if ( comm )
        retVal->setCommunicationList( comm );

    if ( this->hasGhosts() ) {
        retVal->copyGhostValues( *this );
        AMP::Utilities::Algorithms<TYPE>::copy_n(
            this->d_AddBuffer, this->d_ghostSize, retVal->d_AddBuffer );
    }

    return retVal;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::registerChildObjects(
    AMP::IO::RestartManager *manager ) const
{
    VectorDataDefault<TYPE, Allocator>::registerChildObjects( manager );
}
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::writeRestart( int64_t fid ) const
{
    VectorDataDefault<TYPE, Allocator>::writeRestart( fid );
}
template<typename TYPE, class Allocator>
VectorDataDevice<TYPE, Allocator>::VectorDataDevice( int64_t fid, AMP::IO::RestartManager *manager )
    : VectorDataDefault<TYPE, Allocator>( fid, manager )
{
}

// Functions overloaded from GhostDataHelpers and in turn VectorData
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::fillGhosts( const Scalar &val_in )
{
    const auto val = static_cast<TYPE>( val_in );
    AMP::Utilities::Algorithms<TYPE>::fill_n( this->d_Ghosts, this->d_ghostSize, val );
    AMP::Utilities::Algorithms<TYPE>::fill_n(
        this->d_AddBuffer, this->d_ghostSize, static_cast<TYPE>( 0.0 ) );
}

template<typename TYPE, class Allocator>
bool VectorDataDevice<TYPE, Allocator>::containsGlobalElement( size_t i ) const
{
    if ( ( i >= this->d_CommList->getStartGID() ) &&
         ( i < this->d_CommList->getStartGID() + this->d_CommList->numLocalRows() ) )
        return true;
    return DeviceDataHelpers<TYPE>::containsIndex( this->d_ghostSize, this->d_ReceiveDOFList, i );
}

template<typename TYPE, class Allocator>
bool VectorDataDevice<TYPE, Allocator>::allGhostIndices( size_t N, const size_t *ndx ) const
{
    return DeviceDataHelpers<TYPE>::allGhostIndices(
        N, ndx, this->d_localStart, this->d_localStart + this->d_localSize );
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::setGhostValuesByGlobalID( size_t N,
                                                                  const size_t *ndx_,
                                                                  const void *vals_,
                                                                  const typeID &id )
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup               = copyToScratchSpace( N, ndx_, vals_, id );
    const auto scratchUsed = std::get<0>( tup );
    const auto *ndx        = std::get<1>( tup );
    const auto *vals       = std::get<2>( tup );

    if ( id == AMP::getTypeID<TYPE>() ) {
        AMP_ASSERT( *( this->d_UpdateState ) != UpdateState::ADDING );
        *( this->d_UpdateState ) = UpdateState::SETTING;
        AMP_DEBUG_INSIST( allGhostIndices( N, ndx ), "Non ghost index encountered" );
        auto data = reinterpret_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::setGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           ndx,
                                                           data,
                                                           this->d_ghostSize,
                                                           this->d_Ghosts );
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::addGhostValuesByGlobalID( size_t N,
                                                                  const size_t *ndx_,
                                                                  const void *vals_,
                                                                  const typeID &id )
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup               = copyToScratchSpace( N, ndx_, vals_, id );
    const auto scratchUsed = std::get<0>( tup );
    const auto *ndx        = std::get<1>( tup );
    const auto *vals       = std::get<2>( tup );

    if ( id == AMP::getTypeID<TYPE>() ) {
        AMP_ASSERT( *( this->d_UpdateState ) != UpdateState::SETTING );
        *( this->d_UpdateState ) = UpdateState::ADDING;
        AMP_DEBUG_INSIST( this->allGhostIndices( N, ndx ), "Non ghost index encountered" );
        auto data = reinterpret_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::addGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           ndx,
                                                           data,
                                                           this->d_ghostSize,
                                                           this->d_AddBuffer );
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getGhostValuesByGlobalID( size_t N,
                                                                  const size_t *ndx_,
                                                                  void *vals_,
                                                                  const typeID &id ) const
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup         = copyToScratchSpace( N, ndx_, vals_, id );
    auto scratchUsed = std::get<0>( tup );
    auto *ndx        = std::get<1>( tup );
    auto *vals       = std::get<2>( tup );

    if ( id != AMP::getTypeID<TYPE>() ) {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    } else {

        auto data       = reinterpret_cast<TYPE *>( vals );
        size_t *indices = const_cast<size_t *>( ndx );
        AMP_DEBUG_INSIST( this->allGhostIndices( N, indices ), "Non ghost index encountered" );

        DeviceDataHelpers<TYPE>::getGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           indices,
                                                           this->d_ghostSize,
                                                           this->d_Ghosts,
                                                           this->d_AddBuffer,
                                                           data );

        if ( scratchUsed ) {
            AMP::Utilities::copy<TYPE, TYPE>( N, data, reinterpret_cast<TYPE *>( vals_ ) );
        }
    }
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getGhostAddValuesByGlobalID( size_t N,
                                                                     const size_t *ndx_,
                                                                     void *vals_,
                                                                     const typeID &id ) const
{
    AMP_ASSERT( inDeviceMemory( this->d_data ) );
    auto tup         = copyToScratchSpace( N, ndx_, vals_, id );
    auto scratchUsed = std::get<0>( tup );
    auto *ndx        = std::get<1>( tup );
    auto *vals       = std::get<2>( tup );

    if ( id != AMP::getTypeID<TYPE>() ) {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    } else {
        AMP_DEBUG_INSIST( this->allGhostIndices( N, ndx ), "Non ghost index encountered" );
        auto data = reinterpret_cast<TYPE *>( vals );
        DeviceDataHelpers<TYPE>::getGhostAddValuesByGlobalID( this->d_ghostSize,
                                                              this->d_ReceiveDOFList,
                                                              N,
                                                              ndx,
                                                              this->d_ghostSize,
                                                              this->d_AddBuffer,
                                                              data );
        if ( scratchUsed ) {
            AMP::Utilities::copy<TYPE, TYPE>( N, data, reinterpret_cast<TYPE *>( vals_ ) );
        }
    }
}
} // namespace AMP::LinearAlgebra

#endif
