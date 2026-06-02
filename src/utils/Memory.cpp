#include "AMP/utils/Memory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Utilities.h"

#include <cstring>


#ifndef AMP_USE_DEVICE
    #define deviceMemcpy( ... ) AMP_ERROR( "Device memcpy without device" )
    #define deviceMemset( ... ) AMP_ERROR( "Device memset without device" )
    #define deviceSynchronize() AMP_ERROR( "Device synchronize without device" )
#endif


namespace AMP::Utilities {


/****************************************************************************
 *  Get pointer location                                                     *
 ****************************************************************************/
MemoryType getMemoryType( [[maybe_unused]] const void *ptr )
{
    [[maybe_unused]] auto type = MemoryType::host;
#ifdef AMP_USE_CUDA
    type = getCudaMemoryType( ptr );
    if ( type != MemoryType::unregistered )
        return type;
#endif
#ifdef AMP_USE_HIP
    type = getHipMemoryType( ptr );
    if ( type != MemoryType::unregistered )
        return type;
#endif
    return MemoryType::host;
}


/****************************************************************************
 *  Get the string for the memory location                                   *
 ****************************************************************************/
std::string_view getString( MemoryType type )
{
    if ( type == MemoryType::unregistered )
        return "unregistered";
    else if ( type == MemoryType::host )
        return "host";
    else if ( type == MemoryType::device )
        return "device";
    else if ( type == MemoryType::managed )
        return "managed";
    else if ( type == MemoryType::none )
        return "none";
    else
        AMP_ERROR( "Unknown pointer type" );
}
MemoryType memoryLocationFromString( [[maybe_unused]] std::string_view name )
{
#ifdef AMP_USE_DEVICE
    if ( name == "managed" || name == "Managed" ) {
        return MemoryType::managed;
    } else if ( name == "device" || name == "Device" ) {
        return MemoryType::device;
    }
#endif
    return MemoryType::host;
}


/****************************************************************************
 *  Helper functions to check compatibility of memory spaces with eachother *
 *  and if they can run on device                                           *
 ****************************************************************************/
bool memoryLocationsDeviceAccessible( const MemoryType t )
{
    // Trivial version of following functions. This simply
    // asserts that a space is registered and returns true
    // if device accessible

    AMP_INSIST( t > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t1 unregistered" );
    return t >= MemoryType::managed;
}

bool memoryLocationsDeviceAccessible( const MemoryType t1,
                                      const MemoryType t2,
                                      const bool check_strict )
{
    // Check that t1 and t2 are identical if user requests strict checking
    if ( check_strict ) {
        AMP_INSIST( t1 == t2,
                    "AMP::Utilities::memoryLocationsDeviceAccessible: Mismatched memory spaces "
                    "with strict checking enabled" );
    }

    // do assert always that spaces are not unregistered
    AMP_INSIST( t1 > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t1 unregistered" );
    AMP_INSIST( t2 > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t2 unregistered" );

    // non-strictly both must accessible from the same space
    // if both are device accessible return true,
    // else if both are host accessible return false,
    // finally if one is host-only and other is device-only error out
    if ( t1 >= MemoryType::managed && t2 >= MemoryType::managed ) {
        return true;
    } else if ( t1 <= MemoryType::managed && t2 <= MemoryType::managed ) {
        return false;
    } else {
        AMP_ERROR(
            "AMP::Utilities::memoryLocationsDeviceAccessible: memory spaces are incompatible" );
        return false;
    }
}

bool memoryLocationsDeviceAccessible( const MemoryType t1,
                                      const MemoryType t2,
                                      const MemoryType t3,
                                      const bool check_strict )
{
    // Check that t1 == t2 == t3 if user requests strict checking
    if ( check_strict ) {
        AMP_INSIST( t1 == t2,
                    "AMP::Utilities::memoryLocationsDeviceAccessible: Mismatched memory spaces "
                    "with strict checking enabled" );
        AMP_INSIST( t1 == t3,
                    "AMP::Utilities::memoryLocationsDeviceAccessible: Mismatched memory spaces "
                    "with strict checking enabled" );
    }

    // do assert always that spaces are not unregistered
    AMP_INSIST( t1 > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t1 unregistered" );
    AMP_INSIST( t2 > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t2 unregistered" );
    AMP_INSIST( t3 > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t3 unregistered" );

    // as above, but now all three need to be in compatible space
    // return true if all are device-accessible, false if all host-accessible,
    // error if accessibility does not overlap
    if ( t1 >= MemoryType::managed && t2 >= MemoryType::managed && t3 >= MemoryType::managed ) {
        return true;
    } else if ( t1 <= MemoryType::managed && t2 <= MemoryType::managed &&
                t3 <= MemoryType::managed ) {
        return false;
    } else {
        AMP_ERROR(
            "AMP::Utilities::memoryLocationsDeviceAccessible: memory spaces are incompatible" );
        return false;
    }
}

/****************************************************************************
 *  Helper enum / function to determine type of copy operation               *
 ****************************************************************************/
enum class MemoryDirection { HOST, DEVICE, MANAGED, HOST_TO_DEVICE, DEVICE_TO_HOST };
MemoryDirection getMemoryOp( const void *src, void *dst )
{
    const auto t1 = getMemoryType( src );
    const auto t2 = getMemoryType( dst );
    if ( t1 == MemoryType::managed && t2 == MemoryType::managed ) {
        // managed-managed operations can use device or CPU
#ifdef AMP_USE_DEVICE
        return MemoryDirection::DEVICE;
#else
        return MemoryDirection::HOST;
#endif
    } else if ( t1 <= MemoryType::managed && t2 <= MemoryType::managed ) {
        // host-host
        return MemoryDirection::HOST;
    } else if ( t1 <= MemoryType::host ) {
        // host to device
        return MemoryDirection::HOST_TO_DEVICE;
    } else if ( t2 <= MemoryType::host ) {
        // device to host
        return MemoryDirection::DEVICE_TO_HOST;
    } else {
        // device to device
        return MemoryDirection::DEVICE;
    }
}


/****************************************************************************
 *  Copy / Fill memory                                                       *
 ****************************************************************************/
void memcpy( void *dst, const void *src, std::size_t count )
{
    auto op = getMemoryOp( src, dst );
    if ( op == MemoryDirection::HOST ) {
        std::memcpy( dst, src, count );
    } else if ( op == MemoryDirection::DEVICE_TO_HOST ) {
        deviceMemcpy( dst, src, count, deviceMemcpyDeviceToHost );
        deviceSynchronize();
    } else if ( op == MemoryDirection::HOST_TO_DEVICE ) {
        deviceMemcpy( dst, src, count, deviceMemcpyHostToDevice );
        deviceSynchronize();
    } else {
        deviceMemcpy( dst, src, count, deviceMemcpyDeviceToDevice );
        deviceSynchronize();
    }
}
void memset( void *dst, int ch, std::size_t count )
{
    const auto t = getMemoryType( dst );
    if ( t == MemoryType::managed ) {
        // managed memory operations can use device or CPU
#ifdef AMP_USE_DEVICE
        deviceMemset( dst, ch, count );
#else
        std::memset( dst, ch, count );
#endif
    } else if ( t < MemoryType::managed ) {
        // host memset
        std::memset( dst, ch, count );
    } else {
        // device memset
        deviceMemset( dst, ch, count );
    }
}
void zero( void *dst, std::size_t count ) { AMP::Utilities::memset( dst, 0, count ); }
template<class T1, class T2>
void copy( size_t N, const T1 *src, T2 *dst )
{
    static_assert( std::is_trivially_copyable_v<T1> );
    static_assert( std::is_trivially_copyable_v<T2> );
    if constexpr ( std::is_same_v<T1, T2> ) {
        // The types are the same and trivial, use memcpy
        AMP::Utilities::memcpy( dst, src, N * sizeof( T1 ) );
    } else {
        // Types are not the same
        auto op = getMemoryOp( src, dst );
        if ( op == MemoryDirection::HOST ) {
            for ( size_t i = 0; i < N; i++ )
                dst[i] = src[i];
        } else if ( op == MemoryDirection::DEVICE_TO_HOST ) {
            auto tmp = new T1[N];
            AMP::Utilities::memcpy( tmp, src, N * sizeof( T1 ) );
            for ( size_t i = 0; i < N; i++ )
                dst[i] = tmp[i];
            delete[] tmp;
        } else if ( op == MemoryDirection::HOST_TO_DEVICE ) {
            auto tmp = new T2[N];
            for ( size_t i = 0; i < N; i++ )
                tmp[i] = src[i];
            AMP::Utilities::memcpy( dst, tmp, N * sizeof( T2 ) );
            delete[] tmp;
        } else {
#ifdef AMP_USE_DEVICE
            if constexpr ( std::is_integral_v<T1> || std::is_integral_v<T2> ) {
                AMP_ERROR( "Converting device vector int/float conversion is not supported" );
            } else {
                copyCast<T1, T2, Backend::Hip_Cuda>( N, src, dst );
            }
#else
            AMP_ERROR( "No backend" );
#endif
        }
    }
}


/****************************************************************************
 *  Explicit instantiations                                                  *
 ****************************************************************************/
#define INSTANTIATE( T1, T2 ) template void copy<T1, T2>( size_t N, const T1 *, T2 * )
INSTANTIATE( int, int );
INSTANTIATE( int, long long );
INSTANTIATE( int, unsigned long );
INSTANTIATE( int, float );
INSTANTIATE( int, double );
INSTANTIATE( long long, int );
INSTANTIATE( long long, long long );
INSTANTIATE( long long, unsigned long );
INSTANTIATE( unsigned long, int );
INSTANTIATE( size_t, size_t );
INSTANTIATE( size_t, long long );
INSTANTIATE( float, int );
INSTANTIATE( float, float );
INSTANTIATE( float, double );
INSTANTIATE( double, int );
INSTANTIATE( double, float );
INSTANTIATE( double, double );


} // namespace AMP::Utilities
