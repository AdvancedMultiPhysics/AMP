#include "AMP/utils/Backend.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/memory.h"

#include <algorithm>
#include <string>


namespace AMP::Utilities {


Backend getDefaultBackend( const MemoryType memory_location )
{
    if ( memory_location == MemoryType::unregistered || memory_location == MemoryType::host ) {
#ifdef AMP_USE_OPENMP
        return Backend::OpenMP;
#else
        return Backend::Serial;
#endif
    } else if ( memory_location == MemoryType::managed || memory_location == MemoryType::device ) {
        return Backend::Hip_Cuda;
    }
    return Backend::Serial;
}

std::string getString( const Backend backend )
{
    if ( backend == Backend::Serial ) {
        return "Serial";
    } else if ( backend == Backend::Hip_Cuda ) {
        return "Hip_Cuda";
    } else if ( backend == Backend::Kokkos ) {
        return "Kokkos";
    } else if ( backend == Backend::OpenMP ) {
        return "OpenMP";
    } else if ( backend == Backend::OpenACC ) {
        return "OpenACC";
    } else if ( backend == Backend::OpenCL ) {
        return "OpenCL";
    } else if ( backend == Backend::RAJA ) {
        return "RAJA";
    }
    AMP_ERROR( "Unknown backend" );
}

Backend backendFromString( const std::string &name )
{
    auto lcname( name );
    std::transform( lcname.begin(), lcname.end(), lcname.begin(), []( unsigned char c ) {
        return std::tolower( c );
    } );

    if ( lcname == "serial" ) {
        return Backend::Serial;
    } else if ( name == "hip_cuda" ) {
#ifdef AMP_USE_DEVICE
        return Backend::Hip_Cuda;
#else
        AMP_ERROR( "HIP or CUDA need to be loaded to be able to use hip_cuda backend." );
#endif
    } else if ( lcname == "kokkos" ) {
#ifdef AMP_USE_KOKKOS
        return Backend::Kokkos;
#else
        AMP_ERROR( "KOKKOS need to be loaded to be able to use kokkos backend." );
#endif
    } else if ( lcname == "openmp" ) {
#ifdef AMP_USE_OPENMP
        return Backend::OpenMP;
#else
        AMP_ERROR( "OpenMP need to be loaded to be able to use OpenMP backend." );
#endif
    } else if ( lcname == "openacc" ) {
#ifdef AMP_USE_OPENACC
        return Backend::OpenACC;
#else
        AMP_ERROR( "OpenACC need to be loaded to be able to use OpenACC backend." );
#endif
    } else if ( lcname == "opencl" ) {
#ifdef AMP_USE_OPENCL
        return Backend::OpenCL;
#else
        AMP_ERROR( "OpenCL need to be loaded to be able to use OpenCL backend." );
#endif
    } else if ( lcname == "raja" ) {
#ifdef AMP_USE_OPENCL
        return Backend::RAJA;
#else
        AMP_ERROR( "RAJA need to be loaded to be able to use RAJA backend." );
#endif
    }
    AMP_ERROR( "Unknown backend" );
}

} // namespace AMP::Utilities
