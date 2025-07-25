#ifndef included_AMP_Backend
#define included_AMP_Backend

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/memory.h"

#include <algorithm>
#include <string>


namespace AMP::Utilities {


//! Enum to store the backend used for gpu acceleration
enum class Backend : int8_t {
    Serial   = 0,
    Hip_Cuda = 1,
    Kokkos   = 2,
    OpenMP   = 3,
    OpenACC  = 4,
    OpenCL   = 5,
    RAJA     = 6
};


//! Structs for each backend
namespace AccelerationBackend {
struct Serial {
};
#ifdef USE_DEVICE
struct Hip_Cuda {
};
#endif
#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )
struct Kokkos {
};
#endif
#ifdef USE_OPENMP
struct OpenMP {
};
#endif
#ifdef USE_OPENACC
struct OpenACC {
};
#endif
#ifdef USE_OPENCL
struct OpenCL {
};
#endif
#ifdef USE_RAJA
struct RAJA {
};
#endif
} // namespace AccelerationBackend


Backend getDefaultBackend( const MemoryType memory_location );

std::string getString( const Backend backend );

Backend backendFromString( const std::string &name );


} // namespace AMP::Utilities

#endif
