#ifndef included_AMP_Backend
#define included_AMP_Backend

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Memory.h"

#include <algorithm>
#include <string_view>


namespace AMP::Utilities {

// specifies where code should be executed
enum class ExecutionSpace : int8_t { unspecified, cpu, device };

ExecutionSpace getDefaultExecutionSpace( const MemoryType memory_location );
std::string_view getString( const ExecutionSpace exec_space );
ExecutionSpace executionSpaceFromString( const std::string_view name );

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


Backend getDefaultBackend( const MemoryType memory_location );
std::string_view getString( const Backend backend );
Backend backendFromString( const std::string_view name );

} // namespace AMP::Utilities

#endif
