#include "AMP/utils/hip/helper_hip.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"
#include <hip/hip_runtime.h>

AMP::Utilities::MemoryType getHipMemoryType( const void *ptr )
{
    hipPointerAttribute_t attributes;
    auto err = hipPointerGetAttributes( &attributes, ptr );
    checkHipErrors( err );
    if ( attributes.type == hipMemoryTypeUnregistered )
        return AMP::Utilities::MemoryType::unregistered;
    else if ( attributes.type == hipMemoryTypeHost )
        return AMP::Utilities::MemoryType::host;
    else if ( attributes.type == hipMemoryTypeDevice )
        return AMP::Utilities::MemoryType::device;
    else if ( attributes.type == hipMemoryTypeManaged )
        return AMP::Utilities::MemoryType::managed;
    else
        AMP_ERROR( "Unknown pointer type" );
    return AMP::Utilities::MemoryType::unregistered;
}

// Check
template<typename T>
void checkHipErrors( T result, const StackTrace::source_location &source )
{
    if ( result ) {
        fprintf( stderr,
                 "HIP error at %s:%d code=%d(%s) \"%s\" \n",
                 source.file_name(),
                 source.line(),
                 static_cast<int>( result ),
                 hipGetName( result ),
                 source.function_name() );
        // Make sure we call HIP Device Reset before exiting
        (void) hipDeviceReset();
        exit( EXIT_FAILURE );
    }
}
void getLastHipError( const char *errorMessage, const StackTrace::source_location &source )
{
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err ) {
        fprintf( stderr,
                 "%s(%i) : getLastHipError() HIP error : %s : (%d) %s.\n",
                 source.file_name(),
                 source.line(),
                 errorMessage,
                 (int) err,
                 hipGetErrorString( err ) );
        (void) hipDeviceReset();
        exit( EXIT_FAILURE );
    }
}

// HIP Runtime error messages
template<>
const char *hipGetName<hipError_t>( hipError_t error )
{
    return hipGetErrorName( error );
}
template void checkHipErrors<hipError_t>( hipError_t, const StackTrace::source_location & );
