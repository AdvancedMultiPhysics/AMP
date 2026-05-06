#ifndef AMP_HipHelpers
#define AMP_HipHelpers

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>

#include "AMP/utils/UtilityMacros.h"
#include "StackTrace/source_location.h"

#define hostDeviceId hipCpuDeviceId

#define deviceMemAttachGlobal hipMemAttachGlobal

#define deviceMemcpyHostToDevice hipMemcpyHostToDevice
#define deviceMemcpyDeviceToHost hipMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define deviceInit( ... ) checkHipErrors( hipInit( __VA_ARGS__ ) )
#define deviceGetCount( ... ) checkHipErrors( hipGetDeviceCount( __VA_ARGS__ ) )
#define deviceBind( ... ) checkHipErrors( hipSetDevice( __VA_ARGS__ ) )
#define deviceId( ... ) checkHipErrors( hipGetDevice( __VA_ARGS__ ) )
#define deviceSynchronize() checkHipErrors( hipDeviceSynchronize() )
#define deviceMalloc( ... ) checkHipErrors( hipMalloc( __VA_ARGS__ ) )
#define deviceMallocManaged( ... ) checkHipErrors( hipMallocManaged( __VA_ARGS__ ) )
#define deviceMemcpy( ... ) checkHipErrors( hipMemcpy( __VA_ARGS__ ) )
#define deviceMemset( ... ) checkHipErrors( hipMemset( __VA_ARGS__ ) )
#define deviceFree( ... ) checkHipErrors( hipFree( __VA_ARGS__ ) )
#define deviceMemPrefetchAsync( ... ) checkHipErrors( hipMemPrefetchAsync( __VA_ARGS__ ) )

namespace AMP::Utilities {
enum class MemoryType : int8_t;
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
void getLastDeviceError( const char *errorMessage,
                         const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );


template<typename FUNC>
static void inline setKernelDims( const int n, FUNC func, dim3 &BlockDim, dim3 &GridDim )
{
    int minGridSize = 0, blockSize = 0;
    checkHipErrors( hipOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, func, 0, 0 ) );
    const int gridSize = ( n + blockSize - 1 ) / blockSize;
    BlockDim           = dim3( blockSize, 1, 1 );
    GridDim            = dim3( gridSize, 1, 1 );
    return;
}

#endif
