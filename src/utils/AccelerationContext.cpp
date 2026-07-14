#include "AMP/utils/AccelerationContext.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/device/Device.h"

#include <optional>

namespace AMP::Utilities {

AccelerationContext::AccelerationContext() : d_stream( nullptr ), d_manage_stream_deletion( false )
{
}

AccelerationContext::AccelerationContext( const ComputeStream stream,
                                          const bool manage_stream_deletion )
    : d_stream( stream ), d_manage_stream_deletion( manage_stream_deletion )
{
}

AccelerationContext::~AccelerationContext()
{
#ifdef AMP_USE_KOKKOS
    // Kokkos execution spaces with streams need to deallocate
    // internal memory tied to that stream *before* that stream
    // is destroyed. Trigger their destructors by writing in nullopts
    d_kokkos_exec_default = std::nullopt;
    d_kokkos_exec_host    = std::nullopt;
#endif
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
    // synchronize on current stream if it exists
    if ( d_stream ) {
        deviceStreamSynchronize( d_stream );
        // then delete it if this context owns it
        if ( d_manage_stream_deletion ) {
            deviceStreamDestroy( d_stream );
        }
    }
#endif
    // update stream and ownership
    d_stream                 = stream;
    d_manage_stream_deletion = manage_stream_deletion;
}

void AccelerationContext::createStream()
{
#ifdef AMP_USE_DEVICE
    if ( !d_stream ) {
        ComputeStream stream;
        deviceStreamCreate( &stream );
        setComputeStream( stream, true );
    }
#endif
}

ComputeStream AccelerationContext::getStream()
{
    createStream();
    return d_stream;
}

void AccelerationContext::synchronizeStream() const
{
#ifdef AMP_USE_DEVICE
    if ( d_stream ) {
        deviceStreamSynchronize( d_stream );
    }
#endif
}

#ifdef AMP_USE_KOKKOS
const Kokkos::DefaultExecutionSpace &AccelerationContext::getKokkosExecDefault()
{
    if ( !d_kokkos_exec_default ) {
        // create execution space lazily and ensure stream exists if needed
    #ifdef AMP_USE_DEVICE
        createStream();
        d_kokkos_exec_default = std::make_optional<Kokkos::DefaultExecutionSpace>( d_stream );
    #else
        d_kokkos_exec_default = std::make_optional<Kokkos::DefaultExecutionSpace>();
    #endif
    }
    return d_kokkos_exec_default.value();
}

const Kokkos::DefaultHostExecutionSpace &AccelerationContext::getKokkosExecHost()
{
    if ( !d_kokkos_exec_host ) {
        // create execution space lazily
        d_kokkos_exec_host = std::make_optional<Kokkos::DefaultHostExecutionSpace>();
    }
    return d_kokkos_exec_host.value();
}
#endif

} // namespace AMP::Utilities
