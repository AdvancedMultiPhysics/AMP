#include "AMP/utils/Memory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"

#include <cstring>

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
    if ( type == MemoryType::unregistered ) {
        AMP_WARN_ONCE( "********** Unregistered memory!! ***********" );
    }

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
 *  and if they can run on device. check_strict ensures compatibility.      *
 *  Returns pair of bools:                                                  *
 *     first: all device accesible                                          *
 *    second: all strictly of device type and not managed                   *
 ****************************************************************************/
std::tuple<bool, bool> memoryLocationsDeviceAccessible( const MemoryType t )
{
    // Trivial version of following functions. This simply
    // asserts that a space is registered and returns true
    // if device accessible

    AMP_INSIST( t > MemoryType::unregistered,
                "AMP::Utilities::memoryLocationsDeviceAccessible: t1 unregistered" );
    bool dev_acc = t >= MemoryType::managed;
    bool all_dev = t == MemoryType::device;
    return std::make_tuple( dev_acc, all_dev );
}

std::tuple<bool, bool>
memoryLocationsDeviceAccessible( const MemoryType t1, const MemoryType t2, const bool check_strict )
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

    bool dev_acc = t1 >= MemoryType::managed && t2 >= MemoryType::managed;
    bool all_dev = t1 == MemoryType::device && t2 == MemoryType::device;
    if ( check_strict ) {
        bool host_acc = t1 <= MemoryType::managed && t2 <= MemoryType::managed;
        AMP_INSIST(
            dev_acc || host_acc,
            "AMP::Utilities::memoryLocationsDeviceAccessible: memory spaces are incompatible" );
    }
    return std::make_tuple( dev_acc, all_dev );
}

std::tuple<bool, bool> memoryLocationsDeviceAccessible( const MemoryType t1,
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

    bool dev_acc =
        t1 >= MemoryType::managed && t2 >= MemoryType::managed && t3 >= MemoryType::managed;
    bool all_dev = t1 == MemoryType::device && t2 == MemoryType::device && t3 == MemoryType::device;
    if ( check_strict ) {
        bool host_acc =
            t1 <= MemoryType::managed && t2 <= MemoryType::managed && t3 <= MemoryType::managed;
        AMP_INSIST(
            dev_acc || host_acc,
            "AMP::Utilities::memoryLocationsDeviceAccessible: memory spaces are incompatible" );
    }
    return std::make_tuple( dev_acc, all_dev );
}

} // namespace AMP::Utilities
