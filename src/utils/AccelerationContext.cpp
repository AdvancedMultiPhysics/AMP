#include "AMP/utils/AccelerationContext.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/device/Device.h"

#include <optional>

namespace AMP::Utilities {

AccelerationContext::AccelerationContext()
    : d_stream( nullptr ),
      d_manage_stream_deletion( false ),
#ifdef AMP_USE_KOKKOS
      d_kokkos_exec_default( std::nullopt ),
      d_kokkos_exec_host( std::nullopt )
#endif
{
}

AccelerationContext::AccelerationContext( const ComputeStream stream,
                                          const bool manage_stream_deletion )
    : d_stream( stream ),
      d_manage_stream_deletion( manage_stream_deletion ),
#ifdef AMP_USE_KOKKOS
    #ifdef AMP_USE_DEVICE
      d_kokkos_exec_default( stream ),
    #else
      d_kokkos_exec_default(),
    #endif
      d_kokkos_exec_host()
#endif
{
}

AccelerationContext::~AccelerationContext()
{
#ifdef AMP_USE_DEVICE
    if ( d_manage_stream_deletion ) {
        deviceStreamDestroy( d_stream );
    }
#endif
}

void AccelerationContext::setComputeStream( const ComputeStream stream,
                                            const bool manage_stream_deletion )
{
#ifdef AMP_USE_DEVICE
    // destroy current stream if we own it and it is non-null
    if ( d_manage_stream_deletion && d_stream ) {
        deviceStreamDestroy( d_stream );
    }
#endif
    // update stream and ownership
    d_stream                 = stream;
    d_manage_stream_deletion = manage_stream_deletion;
    // update Kokkos execution space to match
#if defined( AMP_USE_KOKKOS ) && defined( AMP_USE_DEVICE )
    d_kokkos_exec_default = std::make_optional<Kokkos::DefaultExecutionSpace>( d_stream );
#endif
}

const ComputeStream AccelerationContext::getStream() const { return d_stream; }

void AccelerationContext::synchronizeStream() const
{
#ifdef AMP_USE_DEVICE
    if ( d_stream ) {
        deviceStreamSynchronize( d_stream );
    }
#endif
}

#ifdef AMP_USE_KOKKOS
const Kokkos::DefaultExecutionSpace &AccelerationContext::getKokkosExecDefault() const
{
    AMP_DEBUG_ASSERT( d_kokkos_exec_default );
    return d_kokkos_exec_default.value();
}

const Kokkos::DefaultHostExecutionSpace &AccelerationContext::getKokkosExecHost() const
{
    AMP_DEBUG_ASSERT( d_kokkos_exec_host );
    return d_kokkos_exec_host.value();
}
#endif

} // namespace AMP::Utilities
