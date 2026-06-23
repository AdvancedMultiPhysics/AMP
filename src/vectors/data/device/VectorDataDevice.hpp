#ifndef included_AMP_VectorDataDevice_hpp
#define included_AMP_VectorDataDevice_hpp

#include "AMP/IO/RestartManager.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/data/device/DeviceDataHelpers.h"
#include "AMP/vectors/data/device/VectorDataDevice.h"
#include <cstring>

#include "ProfilerApp.h"

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
VectorDataDevice<TYPE, Allocator>::VectorDataDevice( std::shared_ptr<CommunicationList> commList,
                                                     TYPE *data )
    : VectorDataDefault<TYPE, Allocator>( commList, data )
{
}

template<typename TYPE, class Allocator>
VectorDataDevice<TYPE, Allocator>::~VectorDataDevice()
{
    if ( this->d_idx_map_scratch ) {
        d_idx_alloc.deallocate( this->d_idx_map_scratch, this->d_map_scratch_size );
        this->d_idx_map_scratch = nullptr;
    }
    if ( this->d_idx_req_scratch ) {
        d_idx_alloc.deallocate( this->d_idx_req_scratch, this->d_scratch_size );
        this->d_idx_req_scratch = nullptr;
    }
    if ( this->d_scalar_scratch ) {
        d_scalar_alloc.deallocate( this->d_scalar_scratch, this->d_scratch_size );
        this->d_scalar_scratch = nullptr;
    }
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::setMapScratchSpace( const size_t N ) const
{
    if ( N > this->d_map_scratch_size ) {
        d_idx_alloc.deallocate( this->d_idx_map_scratch, this->d_map_scratch_size );
        this->d_map_scratch_size = N;
        this->d_idx_map_scratch  = d_idx_alloc.allocate( this->d_map_scratch_size );
        AMP::Utilities::Algorithms::zero_n( this->d_idx_map_scratch, N, d_memory_location );
    }
    AMP_ASSERT( d_idx_map_scratch );
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::setScratchSpace( const size_t N ) const
{
    if ( N > this->d_scratch_size || !this->d_idx_req_scratch ) {
        d_idx_alloc.deallocate( this->d_idx_req_scratch, this->d_scratch_size );
        d_scalar_alloc.deallocate( this->d_scalar_scratch, this->d_scratch_size );
        this->d_scratch_size    = N;
        this->d_idx_req_scratch = d_idx_alloc.allocate( this->d_scratch_size );
        this->d_scalar_scratch  = d_scalar_alloc.allocate( this->d_scratch_size );
    }
    AMP_ASSERT( d_idx_req_scratch && d_scalar_scratch );
}

template<typename TYPE, class Allocator>
std::tuple<bool, size_t *, void *>
VectorDataDevice<TYPE, Allocator>::copyToScratchSpace( size_t num,
                                                       const size_t *indices_,
                                                       const void *vals_,
                                                       const typeID &id,
                                                       AMP::Utilities::MemoryType buf_loc ) const
{
    const bool scratchUsed = buf_loc <= AMP::Utilities::MemoryType::host;

    if ( !scratchUsed ) {
        return std::make_tuple(
            scratchUsed, const_cast<size_t *>( indices_ ), const_cast<void *>( vals_ ) );
    } else {
        this->setScratchSpace( num );
        AMP::Utilities::Algorithms::copy_n(
            this->d_idx_req_scratch, d_memory_location, indices_, buf_loc, num );

        if ( id == getTypeID<TYPE>() ) {
            auto tvals = static_cast<const TYPE *>( vals_ );
            AMP::Utilities::Algorithms::copy_n(
                this->d_scalar_scratch, d_memory_location, tvals, buf_loc, num );
        } else if ( id == getTypeID<double>() ) {
            auto dvals = static_cast<const double *>( vals_ );
            AMP::Utilities::Algorithms::copyCast(
                this->d_scalar_scratch, d_memory_location, dvals, buf_loc, num );
        } else if ( id == getTypeID<float>() ) {
            auto fvals = static_cast<const float *>( vals_ );
            AMP::Utilities::Algorithms::copyCast(
                this->d_scalar_scratch, d_memory_location, fvals, buf_loc, num );
        } else {
            AMP_ERROR( "Conversion not supported yet" );
        }
        return std::make_tuple( scratchUsed, this->d_idx_req_scratch, this->d_scalar_scratch );
    }
}

template<typename TYPE, class Allocator>
inline void
VectorDataDevice<TYPE, Allocator>::setValuesByLocalID( size_t num,
                                                       const size_t *indices_,
                                                       const void *vals_,
                                                       const typeID &id,
                                                       AMP::Utilities::MemoryType buf_loc )
{
    auto [scratchUsed, indices, vals] = copyToScratchSpace( num, indices_, vals_, id, buf_loc );

    if ( id == getTypeID<TYPE>() || scratchUsed ) {
        auto data = static_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<double>() ) {
        auto data = static_cast<const double *>( vals );
        DeviceDataHelpers<double, TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<float>() ) {
        auto data = static_cast<const float *>( vals );
        DeviceDataHelpers<float, TYPE>::setValuesByIndex( num, indices, data, this->d_data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}

template<typename TYPE, class Allocator>
inline void
VectorDataDevice<TYPE, Allocator>::addValuesByLocalID( size_t num,
                                                       const size_t *indices_,
                                                       const void *vals_,
                                                       const typeID &id,
                                                       AMP::Utilities::MemoryType buf_loc )
{
    auto [scratchUsed, indices, vals] = copyToScratchSpace( num, indices_, vals_, id, buf_loc );

    if ( id == getTypeID<TYPE>() ) {
        auto data = static_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<double>() ) {
        auto data = static_cast<const double *>( vals );
        DeviceDataHelpers<double, TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else if ( id == getTypeID<float>() ) {
        auto data = static_cast<const float *>( vals );
        DeviceDataHelpers<float, TYPE>::addValuesByIndex( num, indices, data, this->d_data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( *( this->d_UpdateState ) == UpdateState::UNCHANGED )
        *( this->d_UpdateState ) = UpdateState::LOCAL_CHANGED;
}
template<typename TYPE, class Allocator>
inline void
VectorDataDevice<TYPE, Allocator>::getValuesByLocalID( size_t num,
                                                       const size_t *indices_,
                                                       void *vals_,
                                                       const typeID &id,
                                                       AMP::Utilities::MemoryType buf_loc ) const
{
    auto [scratchUsed, indices, vals] = copyToScratchSpace( num, indices_, vals_, id, buf_loc );

    if ( id == getTypeID<TYPE>() || scratchUsed ) {
        auto data = static_cast<TYPE *>( vals );
        DeviceDataHelpers<TYPE>::getValuesByIndex( num, indices, this->d_data, data );
    } else if ( id == getTypeID<double>() ) {
        auto data = static_cast<double *>( vals );
        DeviceDataHelpers<TYPE, double>::getValuesByIndex( num, indices, this->d_data, data );
    } else if ( id == getTypeID<float>() ) {
        auto data = static_cast<float *>( vals );
        DeviceDataHelpers<TYPE, float>::getValuesByIndex( num, indices, this->d_data, data );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }

    if ( scratchUsed ) {
        auto data = static_cast<TYPE *>( vals );
        if ( id == getTypeID<TYPE>() ) {
            auto tvals = static_cast<TYPE *>( vals_ );
            AMP::Utilities::Algorithms::copy_n( tvals, buf_loc, data, d_memory_location, num );
        } else if ( id == getTypeID<double>() ) {
            auto dvals = static_cast<double *>( vals_ );
            AMP::Utilities::Algorithms::copyCast( dvals, buf_loc, data, d_memory_location, num );
        } else if ( id == getTypeID<float>() ) {
            auto fvals = static_cast<float *>( vals_ );
            AMP::Utilities::Algorithms::copyCast( fvals, buf_loc, data, d_memory_location, num );
        }
    }
}


/****************************************************************
 * Copy raw data                                                 *
 ****************************************************************/
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::putRawData( const void *in,
                                                    const typeID &id,
                                                    AMP::Utilities::MemoryType buf_loc )
{
    if ( id == getTypeID<TYPE>() ) {
        auto data = static_cast<const TYPE *>( in );
        AMP::Utilities::Algorithms::copy_n(
            this->d_data, d_memory_location, data, buf_loc, this->d_localSize );
    } else if ( id == getTypeID<double>() ) {
        const auto *data_in = static_cast<const double *>( in );
        AMP::Utilities::Algorithms::copyCast(
            this->d_data, d_memory_location, data_in, buf_loc, this->d_localSize );
    } else if ( id == getTypeID<float>() ) {
        const auto *data_in = static_cast<const float *>( in );
        AMP::Utilities::Algorithms::copyCast(
            this->d_data, d_memory_location, data_in, buf_loc, this->d_localSize );
    } else {
        AMP_ERROR( "Conversion not supported yet" );
    }
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getRawData( void *out,
                                                    const typeID &id,
                                                    AMP::Utilities::MemoryType buf_loc ) const
{
    if ( id == getTypeID<TYPE>() ) {
        auto data = static_cast<TYPE *>( out );
        AMP::Utilities::Algorithms::copy_n(
            data, buf_loc, this->d_data, d_memory_location, this->d_localSize );
    } else if ( id == getTypeID<double>() ) {
        auto *data_out = static_cast<double *>( out );
        AMP::Utilities::Algorithms::copyCast(
            data_out, buf_loc, this->d_data, d_memory_location, this->d_localSize );
    } else if ( id == getTypeID<float>() ) {
        auto *data_out = static_cast<float *>( out );
        AMP::Utilities::Algorithms::copyCast(
            data_out, buf_loc, this->d_data, d_memory_location, this->d_localSize );
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
        AMP::Utilities::Algorithms::copy_n(
            retVal->d_AddBuffer, this->d_AddBuffer, this->d_ghostSize, d_memory_location );
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
    AMP::Utilities::Algorithms::fill_n( this->d_Ghosts, this->d_ghostSize, val, d_memory_location );
    AMP::Utilities::Algorithms::zero_n( this->d_AddBuffer, this->d_ghostSize, d_memory_location );
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
void VectorDataDevice<TYPE, Allocator>::setGhostValuesByGlobalID(
    size_t N,
    const size_t *ndx_,
    const void *vals_,
    const typeID &id,
    AMP::Utilities::MemoryType buf_loc )
{
    setMapScratchSpace( N );
    auto [scratchUsed, ndxReq, vals] = copyToScratchSpace( N, ndx_, vals_, id, buf_loc );

    if ( id == AMP::getTypeID<TYPE>() ) {
        AMP_ASSERT( *( this->d_UpdateState ) != UpdateState::ADDING );
        *( this->d_UpdateState ) = UpdateState::SETTING;
        AMP_DEBUG_INSIST( allGhostIndices( N, ndx_ ), "Non ghost index encountered" );
        auto data = static_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::setGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           ndxReq,
                                                           this->d_idx_map_scratch,
                                                           data,
                                                           this->d_ghostSize,
                                                           this->d_Ghosts );
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::addGhostValuesByGlobalID(
    size_t N,
    const size_t *ndx_,
    const void *vals_,
    const typeID &id,
    AMP::Utilities::MemoryType buf_loc )
{
    setMapScratchSpace( N );
    auto [scratchUsed, ndxReq, vals] = copyToScratchSpace( N, ndx_, vals_, id, buf_loc );

    if ( id == AMP::getTypeID<TYPE>() ) {
        AMP_ASSERT( *( this->d_UpdateState ) != UpdateState::SETTING );
        *( this->d_UpdateState ) = UpdateState::ADDING;
        AMP_DEBUG_INSIST( this->allGhostIndices( N, ndx_ ), "Non ghost index encountered" );
        auto data = static_cast<const TYPE *>( vals );
        DeviceDataHelpers<TYPE>::addGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           ndxReq,
                                                           this->d_idx_map_scratch,
                                                           data,
                                                           this->d_ghostSize,
                                                           this->d_AddBuffer );
    } else {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    }
}
template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getGhostValuesByGlobalID(
    size_t N,
    const size_t *ndx_,
    void *vals_,
    const typeID &id,
    AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorDataDevice::getGhostValuesByGlobalID" );
    setMapScratchSpace( N );
    auto [scratchUsed, ndxReq, vals] = copyToScratchSpace( N, ndx_, vals_, id, buf_loc );

    if ( id != AMP::getTypeID<TYPE>() ) {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    } else {

        auto data = static_cast<TYPE *>( vals );
        AMP_DEBUG_INSIST( this->allGhostIndices( N, ndx_ ), "Non ghost index encountered" );

        DeviceDataHelpers<TYPE>::getGhostValuesByGlobalID( this->d_ghostSize,
                                                           this->d_ReceiveDOFList,
                                                           N,
                                                           ndxReq,
                                                           this->d_idx_map_scratch,
                                                           this->d_ghostSize,
                                                           this->d_Ghosts,
                                                           this->d_AddBuffer,
                                                           data );

        if ( scratchUsed ) {
            AMP::Utilities::Algorithms::copy_n(
                static_cast<TYPE *>( vals_ ), buf_loc, data, d_memory_location, N );
        }
    }
}

template<typename TYPE, class Allocator>
void VectorDataDevice<TYPE, Allocator>::getGhostAddValuesByGlobalID(
    size_t N,
    const size_t *ndx_,
    void *vals_,
    const typeID &id,
    AMP::Utilities::MemoryType buf_loc ) const
{
    setMapScratchSpace( N );
    auto [scratchUsed, ndxReq, vals] = copyToScratchSpace( N, ndx_, vals_, id, buf_loc );

    if ( id != AMP::getTypeID<TYPE>() ) {
        AMP_ERROR( "Ghosts other than same type are not supported yet" );
    } else {
        AMP_DEBUG_INSIST( this->allGhostIndices( N, ndx_ ), "Non ghost index encountered" );
        auto data = static_cast<TYPE *>( vals );
        DeviceDataHelpers<TYPE>::getGhostAddValuesByGlobalID( this->d_ghostSize,
                                                              this->d_ReceiveDOFList,
                                                              N,
                                                              ndxReq,
                                                              this->d_idx_map_scratch,
                                                              this->d_ghostSize,
                                                              this->d_AddBuffer,
                                                              data );
        if ( scratchUsed ) {
            AMP::Utilities::Algorithms::copy_n(
                static_cast<TYPE *>( vals_ ), buf_loc, data, d_memory_location, N );
        }
    }
}
} // namespace AMP::LinearAlgebra

#endif
