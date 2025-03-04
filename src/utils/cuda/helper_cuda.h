#ifndef AMP_CudaHelpers
#define AMP_CudaHelpers

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/cuda/helper_string.h"

#include "StackTrace/source_location.h"

#include <cuda.h>
#include <cuda_runtime.h>

#define deviceMemcpyHostToDevice cudaMemcpyHostToDevice
#define deviceMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define deviceMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define deviceSynchronize() checkCudaErrors( cudaDeviceSynchronize() )
#define deviceMalloc( ... ) checkCudaErrors( cudaMalloc( __VA_ARGS__ ) )
#define deviceMemcpy( ... ) checkCudaErrors( cudaMemcpy( __VA_ARGS__ ) )
#define deviceFree( ... ) checkCudaErrors( cudaFree( __VA_ARGS__ ) )


#ifndef EXIT_WAIVED
    #define EXIT_WAIVED 2
#endif


#ifdef __DRIVER_TYPES_H__
    #ifndef DEVICE_RESET
        #define DEVICE_RESET cudaDeviceReset();
    #endif
#else
    #ifndef DEVICE_RESET
        #define DEVICE_RESET
    #endif
#endif


namespace AMP::Utilities {
enum class MemoryType : uint8_t;
}

// Get the pointer type from cuda
AMP::Utilities::MemoryType getCudaMemoryType( const void *ptr );


// Get the name of a return code
template<typename T>
const char *cudaGetName( T result );

// Check the return code
template<typename T>
void checkCudaErrors( T result,
                      const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );

// Get the last cuda error
void getLastCudaError( const char *errorMessage,
                       const StackTrace::source_location &source = SOURCE_LOCATION_CURRENT() );

#ifndef MAX
    #define MAX( a, b ) ( a > b ? a : b )
#endif

// Float To Int conversion
inline int ftoi( float value )
{
    return ( value >= 0 ? (int) ( value + 0.5 ) : (int) ( value - 0.5 ) );
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores( int major, int minor )
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = { { 0x20, 32 },  // Fermi Generation (SM 2.0) GF100 class
                                        { 0x21, 48 },  // Fermi Generation (SM 2.1) GF10x class
                                        { 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
                                        { 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
                                        { 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
                                        { 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
                                        { 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
                                        { 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
                                        { -1, -1 } };

    int index = 0;

    while ( nGpuArchCoresPerSM[index].SM != -1 ) {
        if ( nGpuArchCoresPerSM[index].SM == ( ( major << 4 ) + minor ) ) {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf( "MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n",
            major,
            minor,
            nGpuArchCoresPerSM[index - 1].Cores );
    return nGpuArchCoresPerSM[index - 1].Cores;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
int gpuDeviceInit( int devID );

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId();


// Initialization code to find the best CUDA Device
int findCudaDevice( int argc, const char **argv );

// General check for CUDA GPU SM Capabilities
bool checkCudaCapabilities( int major_version, int minor_version );


#endif

static void inline setKernelDims( size_t n, dim3 &BlockDim, dim3 &GridDim )
{
    // Parameters for an NVIDIA k20x.  Consider using occupancy API
    const int warpSize    = 32;
    const int maxGridSize = 112; // 8 blocks per MP of tesla k20x,
                                 //  this number might need to be tuned
                                 //  consider querying for device info
    int warpCount    = ( n / warpSize ) + ( ( ( n % warpSize ) == 0 ) ? 0 : 1 );
    int warpPerBlock = std::max( 1, std::min( 4, warpCount ) );
    int threadCount  = warpSize * warpPerBlock;
    int blockCount   = std::min( maxGridSize, std::max( 1, warpCount / warpPerBlock ) );
    BlockDim         = dim3( threadCount, 1, 1 );
    GridDim          = dim3( blockCount, 1, 1 );
    return;
}

// end of CUDA Helper Functions


#endif
