#ifndef AMP_HipHelpers
#define AMP_HipHelpers

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>

#include "AMP/utils/UtilityMacros.h"
#include "StackTrace/source_location.h"

#define deviceMemcpyHostToDevice hipMemcpyHostToDevice
#define deviceMemcpyDeviceToHost hipMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define deviceSynchronize() checkHipErrors( hipDeviceSynchronize() )
#define deviceMalloc( ... ) checkHipErrors( hipMalloc( __VA_ARGS__ ) )
#define deviceMemcpy( ... ) checkHipErrors( hipMemcpy( __VA_ARGS__ ) )
#define deviceFree( ... ) checkHipErrors( hipFree( __VA_ARGS__ ) )


namespace AMP::Utilities {
enum class MemoryType : uint8_t;
}

// Get the pointer type from hip
AMP::Utilities::MemoryType getHipMemoryType( const void *ptr );

// Get the name of a return code
template<typename T>
const char *hipGetName( T result );

// Check the return code
template<typename T>
void checkHipErrors( T result,
                     const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );

// Get the last hip error
void getLastHipError( const char *errorMessage,
                      const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );

static void inline setKernelDims( size_t n, dim3 &BlockDim, dim3 &GridDim )
{
    // Parameters for an NVIDIA k20x.  Consider using occupancy API
    const int warpSize    = 32;
    const int maxGridSize = 112;

    int warpCount    = ( n / warpSize ) + ( ( ( n % warpSize ) == 0 ) ? 0 : 1 );
    int warpPerBlock = std::max( 1, std::min( 4, warpCount ) );
    int threadCount  = warpSize * warpPerBlock;
    int blockCount   = std::min( maxGridSize, std::max( 1, warpCount / warpPerBlock ) );
    BlockDim         = dim3( threadCount, 1, 1 );
    GridDim          = dim3( blockCount, 1, 1 );
    return;
}

#endif
