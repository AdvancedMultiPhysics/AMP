#ifndef included_AMP_Backend
#define included_AMP_Backend

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Memory.h"

#include <algorithm>
#include <string_view>


namespace AMP::Utilities {

//! Enum to store the backend used for execution
enum class Backend : int8_t {
    Serial   = 0,
    Hip_Cuda = 1,
    Kokkos   = 2,
    OpenMP   = 3,
    OpenACC  = 4,
    OpenCL   = 5,
    RAJA     = 6
};

Backend getDefaultBackend( const MemoryType memory_location );
std::string_view getString( const Backend backend );
Backend backendFromString( const std::string_view name );
bool backendMemoryTypeCompatible( const Backend backend, const MemoryType memory_location );

} // namespace AMP::Utilities

#endif
