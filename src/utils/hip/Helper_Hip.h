#ifndef AMP_HipHelpers
#define AMP_HipHelpers

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <hip/hip_runtime.h>

#include "AMP/utils/UtilityMacros.h"
#include "StackTrace/source_location.h"

#define hostDeviceId hipCpuDeviceId

#define deviceMemcpyHostToDevice hipMemcpyHostToDevice
#define deviceMemcpyDeviceToHost hipMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define deviceId( ... ) checkHipErrors( hipGetDevice( __VA_ARGS__ ) )
#define deviceSynchronize() checkHipErrors( hipDeviceSynchronize() )
#define deviceMalloc( ... ) checkHipErrors( hipMalloc( __VA_ARGS__ ) )
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
void getLastHipError( const char *errorMessage,
                      const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );

static void inline setKernelDims( size_t n, dim3 &BlockDim, dim3 &GridDim )
{
    // ORNL Ref:
    // https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf
    // AMD Specs
    // https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
    // Parameters for AMD MI250/300 series.
    // This should change to using occupancy API at
    // https://rocm.docs.amd.com/projects/HIP/en/docs-develop/reference/hip_runtime_api/modules/occupancy.html
    constexpr int waveFrontSize = 64;
    // MI250 CUs (SMs) = 104, max wavefronts/CU = 8
    // MI300 CUs (SMs) = 228, max wavefronts/CU = 10
    // For now go with MI 300 values
    constexpr int maxGridSize = 228 * 10;

    int warpCount    = ( n / waveFrontSize ) + ( ( ( n % waveFrontSize ) == 0 ) ? 0 : 1 );
    int warpPerBlock = std::max( 1, std::min( 4, warpCount ) );
    int threadCount  = waveFrontSize * warpPerBlock;
    int blockCount   = std::min( maxGridSize, std::max( 1, warpCount / warpPerBlock ) );
    BlockDim         = dim3( threadCount, 1, 1 );
    GridDim          = dim3( blockCount, 1, 1 );
    return;
}

#endif
