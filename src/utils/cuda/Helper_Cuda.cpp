#include "AMP/utils/cuda/Helper_Cuda.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <cuda.h>
#include <cuda_runtime.h>


AMP::Utilities::MemoryType getCudaMemoryType( const void *ptr )
{
    cudaPointerAttributes attributes;
    auto err = cudaPointerGetAttributes( &attributes, ptr );
    checkCudaErrors( err );
    if ( attributes.type == cudaMemoryTypeUnregistered )
        return AMP::Utilities::MemoryType::unregistered;
    else if ( attributes.type == cudaMemoryTypeHost )
        return AMP::Utilities::MemoryType::host;
    else if ( attributes.type == cudaMemoryTypeDevice )
        return AMP::Utilities::MemoryType::device;
    else if ( attributes.type == cudaMemoryTypeManaged )
        return AMP::Utilities::MemoryType::managed;
    else
        AMP_ERROR( "Unknown pointer type" );
    return AMP::Utilities::MemoryType::unregistered;
}

// Check
template<typename T>
void checkCudaErrors( T result, const StackTrace::source_location &source )
{
    if ( result ) {
        fprintf( stderr,
                 "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                 source.file_name(),
                 source.line(),
                 static_cast<int>( result ),
                 cudaGetName( result ),
                 source.function_name() );
        // Make sure we call CUDA Device Reset before exiting
        DEVICE_RESET
        exit( EXIT_FAILURE );
    }
}
void getLastCudaError( const char *errorMessage, const StackTrace::source_location &source )
{
#ifdef __DRIVER_TYPES_H__
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr,
                 "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                 source.file_name(),
                 source.line(),
                 errorMessage,
                 (int) err,
                 cudaGetErrorString( err ) );
        DEVICE_RESET
        exit( EXIT_FAILURE );
    }
#endif
}


// CUDA Runtime error messages
template<>
const char *cudaGetName<cudaError_t>( cudaError_t error )
{
    return cudaGetErrorName( error );
}
template void checkCudaErrors<cudaError_t>( cudaError_t, const StackTrace::source_location & );


// CUDA Driver API errors
template<>
const char *cudaGetName<CUresult>( CUresult error )
{
    const char *str = nullptr;
    cuGetErrorName( error, &str );
    return str;
}
template void checkCudaErrors<CUresult>( CUresult, const StackTrace::source_location & );


// cuBLAS API errors
#ifdef CUBLAS_API_H_
template<>
const char *cudaGetName<cublasStatus_t>( cublasStatus_t error )
{
    switch ( error ) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}
template void checkCudaErrors<cublasStatus_t>( cublasStatus_t,
                                               const StackTrace::source_location & );
#endif


// cuFFT API errors
#ifdef _CUFFT_H_
template<>
const char *cudaGetName<cufftResult>( cufftResult error )
{
    switch ( error ) {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
        return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE:
        return "CUFFT_INVALID_DEVICE";
    case CUFFT_PARSE_ERROR:
        return "CUFFT_PARSE_ERROR";
    case CUFFT_NO_WORKSPACE:
        return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
        return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_LICENSE_ERROR:
        return "CUFFT_LICENSE_ERROR";
    }
    return "<unknown>";
}
template void checkCudaErrors<cufftResult>( cufftResult, const StackTrace::source_location & );
#endif


// cuSPARSE API errors
#ifdef CUSPARSEAPI
template<>
const char *cudaGetName<cusparseStatus_t>( cusparseStatus_t error )
{
    switch ( error ) {
    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    return "<unknown>";
}
template void checkCudaErrors<cusparseStatus_t>( cufftResult, const StackTrace::source_location & );
#endif


// cuSOLVER API errors
#ifdef CUSOLVER_COMMON_H_
template<>
const char *cudaGetName<cusolverStatus_t>( cusolverStatus_t error )
{
    switch ( error ) {
    case CUSOLVER_STATUS_SUCCESS:
        return "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
        return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
        return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
        return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
        return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
        return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
        return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
        return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
        return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
        return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
        return "CUSOLVER_STATUS_INVALID_LICENSE";
    }

    return "<unknown>";
}
template void checkCudaErrors<cusolverStatus_t>( cufftResult, const StackTrace::source_location & );
#endif


// cuRAND API errors
#ifdef CURAND_H_
template<>
const char *cudaGetName<curandStatus_t>( curandStatus_t error )
{
    switch ( error ) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "<unknown>";
}
template void checkCudaErrors<curandStatus_t>( cufftResult, const StackTrace::source_location & );
#endif


// NPP API errors
#ifdef NV_NPPIDEFS_H
template<>
const char *cudaGetName<NppStatus>( NppStatus error )
{
    switch ( error ) {
    case NPP_NOT_SUPPORTED_MODE_ERROR:
        return "NPP_NOT_SUPPORTED_MODE_ERROR";
    case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
        return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
    case NPP_RESIZE_NO_OPERATION_ERROR:
        return "NPP_RESIZE_NO_OPERATION_ERROR";
    case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
        return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
    case NPP_BAD_ARGUMENT_ERROR:
        return "NPP_BAD_ARGUMENT_ERROR";
    case NPP_COEFFICIENT_ERROR:
        return "NPP_COEFFICIENT_ERROR";
    case NPP_RECTANGLE_ERROR:
        return "NPP_RECTANGLE_ERROR";
    case NPP_QUADRANGLE_ERROR:
        return "NPP_QUADRANGLE_ERROR";
    case NPP_MEMORY_ALLOCATION_ERR:
        return "NPP_MEMORY_ALLOCATION_ERROR";
    case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
        return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";
    case NPP_INVALID_HOST_POINTER_ERROR:
        return "NPP_INVALID_HOST_POINTER_ERROR";
    case NPP_INVALID_DEVICE_POINTER_ERROR:
        return "NPP_INVALID_DEVICE_POINTER_ERROR";
    case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
        return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";
    case NPP_TEXTURE_BIND_ERROR:
        return "NPP_TEXTURE_BIND_ERROR";
    case NPP_WRONG_INTERSECTION_ROI_ERROR:
        return "NPP_WRONG_INTERSECTION_ROI_ERROR";
    case NPP_NOT_EVEN_STEP_ERROR:
        return "NPP_NOT_EVEN_STEP_ERROR";
    case NPP_INTERPOLATION_ERROR:
        return "NPP_INTERPOLATION_ERROR";
    case NPP_RESIZE_FACTOR_ERROR:
        return "NPP_RESIZE_FACTOR_ERROR";
    case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
        return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
    case NPP_MEMFREE_ERROR:
        return "NPP_MEMFREE_ERROR";
    case NPP_MEMSET_ERROR:
        return "NPP_MEMSET_ERROR";
    case NPP_MEMCPY_ERROR:
        return "NPP_MEMCPY_ERROR";
    case NPP_MIRROR_FLIP_ERROR:
        return "NPP_MIRROR_FLIP_ERROR";
    case NPP_ALIGNMENT_ERROR:
        return "NPP_ALIGNMENT_ERROR";
    case NPP_STEP_ERROR:
        return "NPP_STEP_ERROR";
    case NPP_SIZE_ERROR:
        return "NPP_SIZE_ERROR";
    case NPP_NULL_POINTER_ERROR:
        return "NPP_NULL_POINTER_ERROR";
    case NPP_CUDA_KERNEL_EXECUTION_ERROR:
        return "NPP_CUDA_KERNEL_EXECUTION_ERROR";
    case NPP_NOT_IMPLEMENTED_ERROR:
        return "NPP_NOT_IMPLEMENTED_ERROR";
    case NPP_ERROR:
        return "NPP_ERROR";
    case NPP_SUCCESS:
        return "NPP_SUCCESS";
    case NPP_WRONG_INTERSECTION_QUAD_WARNING:
        return "NPP_WRONG_INTERSECTION_QUAD_WARNING";
    case NPP_MISALIGNED_DST_ROI_WARNING:
        return "NPP_MISALIGNED_DST_ROI_WARNING";
    case NPP_AFFINE_QUAD_INCORRECT_WARNING:
        return "NPP_AFFINE_QUAD_INCORRECT_WARNING";
    case NPP_DOUBLE_SIZE_WARNING:
        return "NPP_DOUBLE_SIZE_WARNING";
    case NPP_WRONG_INTERSECTION_ROI_WARNING:
        return "NPP_WRONG_INTERSECTION_ROI_WARNING";
    case NPP_LUT_PALETTE_BITSIZE_ERROR:
        return "NPP_LUT_PALETTE_BITSIZE_ERROR";
    case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
        return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
    case NPP_QUALITY_INDEX_ERROR:
        return "NPP_QUALITY_INDEX_ERROR";
    case NPP_CHANNEL_ORDER_ERROR:
        return "NPP_CHANNEL_ORDER_ERROR";
    case NPP_ZERO_MASK_VALUE_ERROR:
        return "NPP_ZERO_MASK_VALUE_ERROR";
    case NPP_NUMBER_OF_CHANNELS_ERROR:
        return "NPP_NUMBER_OF_CHANNELS_ERROR";
    case NPP_COI_ERROR:
        return "NPP_COI_ERROR";
    case NPP_DIVISOR_ERROR:
        return "NPP_DIVISOR_ERROR";
    case NPP_CHANNEL_ERROR:
        return "NPP_CHANNEL_ERROR";
    case NPP_STRIDE_ERROR:
        return "NPP_STRIDE_ERROR";
    case NPP_ANCHOR_ERROR:
        return "NPP_ANCHOR_ERROR";
    case NPP_MASK_SIZE_ERROR:
        return "NPP_MASK_SIZE_ERROR";
    case NPP_MOMENT_00_ZERO_ERROR:
        return "NPP_MOMENT_00_ZERO_ERROR";
    case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
        return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
    case NPP_THRESHOLD_ERROR:
        return "NPP_THRESHOLD_ERROR";
    case NPP_CONTEXT_MATCH_ERROR:
        return "NPP_CONTEXT_MATCH_ERROR";
    case NPP_FFT_FLAG_ERROR:
        return "NPP_FFT_FLAG_ERROR";
    case NPP_FFT_ORDER_ERROR:
        return "NPP_FFT_ORDER_ERROR";
    case NPP_SCALE_RANGE_ERROR:
        return "NPP_SCALE_RANGE_ERROR";
    case NPP_DATA_TYPE_ERROR:
        return "NPP_DATA_TYPE_ERROR";
    case NPP_OUT_OFF_RANGE_ERROR:
        return "NPP_OUT_OFF_RANGE_ERROR";
    case NPP_DIVIDE_BY_ZERO_ERROR:
        return "NPP_DIVIDE_BY_ZERO_ERROR";
    case NPP_RANGE_ERROR:
        return "NPP_RANGE_ERROR";
    case NPP_NO_MEMORY_ERROR:
        return "NPP_NO_MEMORY_ERROR";
    case NPP_ERROR_RESERVED:
        return "NPP_ERROR_RESERVED";
    case NPP_NO_OPERATION_WARNING:
        cudaGetName return "NPP_NO_OPERATION_WARNING";
    case NPP_DIVIDE_BY_ZERO_WARNING:
        return "NPP_DIVIDE_BY_ZERO_WARNING";
    }
    return "<unknown>";
}
template void checkCudaErrors<curandStatus_t>( NppStatus, const StackTrace::source_location & );
#endif


#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
int gpuDeviceInit( int devID )
{
    int device_count;
    checkCudaErrors( cudaGetDeviceCount( &device_count ) );

    if ( device_count == 0 ) {
        fprintf( stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n" );
        exit( EXIT_FAILURE );
    }

    if ( devID < 0 ) {
        devID = 0;
    }

    if ( devID > device_count - 1 ) {
        fprintf( stderr, "\n" );
        fprintf( stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count );
        fprintf( stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID );
        fprintf( stderr, "\n" );
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties( &deviceProp, devID ) );

    if ( deviceProp.computeMode == cudaComputeModeProhibited ) {
        fprintf( stderr,
                 "Error: device is running in <Compute Mode Prohibited>, no threads can "
                 "use ::cudaSetDevice().\n" );
        return -1;
    }

    if ( deviceProp.major < 1 ) {
        fprintf( stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n" );
        exit( EXIT_FAILURE );
    }

    checkCudaErrors( cudaSetDevice( devID ) );
    printf( "gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name );

    return devID;
}


// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId()
{
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0, best_SM_arch = 0;
    int devices_prohibited = 0;

    unsigned long long max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount( &device_count );

    checkCudaErrors( cudaGetDeviceCount( &device_count ) );

    if ( device_count == 0 ) {
        fprintf( stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n" );
        exit( EXIT_FAILURE );
    }

    // Find the best major SM Architecture GPU device
    while ( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if ( deviceProp.computeMode != cudaComputeModeProhibited ) {
            if ( deviceProp.major > 0 && deviceProp.major < 9999 ) {
                best_SM_arch = MAX( best_SM_arch, deviceProp.major );
            }
        } else {
            devices_prohibited++;
        }

        current_device++;
    }

    if ( devices_prohibited == device_count ) {
        fprintf(
            stderr,
            "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n" );
        exit( EXIT_FAILURE );
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while ( current_device < device_count ) {
        cudaGetDeviceProperties( &deviceProp, current_device );

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if ( deviceProp.computeMode != cudaComputeModeProhibited ) {
            if ( deviceProp.major == 9999 && deviceProp.minor == 9999 ) {
                sm_per_multiproc = 1;
            } else {
                sm_per_multiproc = _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor );
            }

            unsigned long long compute_perf = (unsigned long long) deviceProp.multiProcessorCount *
                                              sm_per_multiproc * deviceProp.clockRate;

            if ( compute_perf > max_compute_perf ) {
                // If we find GPU with SM major > 2, search only these
                if ( best_SM_arch > 2 ) {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if ( deviceProp.major == best_SM_arch ) {
                        max_compute_perf = compute_perf;
                        max_perf_device  = current_device;
                    }
                } else {
                    max_compute_perf = compute_perf;
                    max_perf_device  = current_device;
                }
            }
        }

        ++current_device;
    }
    return max_perf_device;
}


// Initialization code to find the best CUDA Device
int findCudaDevice( int argc, const char **argv )
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if ( checkCmdLineFlag( argc, argv, "device" ) ) {
        devID = getCmdLineArgumentInt( argc, argv, "device=" );

        if ( devID < 0 ) {
            printf( "Invalid command line parameter\n " );
            exit( EXIT_FAILURE );
        } else {
            devID = gpuDeviceInit( devID );

            if ( devID < 0 ) {
                printf( "exiting...\n" );
                exit( EXIT_FAILURE );
            }
        }
    } else {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkCudaErrors( cudaSetDevice( devID ) );
        checkCudaErrors( cudaGetDeviceProperties( &deviceProp, devID ) );
        printf( "GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
                devID,
                deviceProp.name,
                deviceProp.major,
                deviceProp.minor );
    }

    return devID;
}


// General check for CUDA GPU SM Capabilities
bool checkCudaCapabilities( int major_version, int minor_version )
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors( cudaGetDevice( &dev ) );
    checkCudaErrors( cudaGetDeviceProperties( &deviceProp, dev ) );

    if ( ( deviceProp.major > major_version ) ||
         ( deviceProp.major == major_version && deviceProp.minor >= minor_version ) ) {
        printf( "  Device %d: <%16s >, Compute SM %d.%d detected\n",
                dev,
                deviceProp.name,
                deviceProp.major,
                deviceProp.minor );
        return true;
    } else {
        printf( "  No GPU device was found that can support CUDA compute capability %d.%d.\n",
                major_version,
                minor_version );
        return false;
    }
}
#endif
