#ifndef included_AMP_AccelerationContext
#define included_AMP_AccelerationContext

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/device/Device.h"

#ifdef AMP_USE_KOKKOS
    #include <Kokkos_Core.hpp>
#endif

#include <optional>

namespace AMP::Utilities {

/*!
 * @brief Class AccelerationContext holds information for accelerators (HIP/Cuda,Kokkos...)
 */
class AccelerationContext final
{
public:
    AccelerationContext();

    AccelerationContext( const ComputeStream stream, const bool manage_stream_deletion );

    ~AccelerationContext();

    void setComputeStream( const ComputeStream stream, const bool manage_stream_deletion );

    const ComputeStream getStream() const;

    void synchronizeStream() const;

#ifdef AMP_USE_KOKKOS
    const Kokkos::DefaultExecutionSpace &getKokkosExecDefault() const;
    const Kokkos::DefaultHostExecutionSpace &getKokkosExecHost() const;
#endif

private:
    /*!
     * GPU compute stream to enqueue work into.
     * This is defined but useless in host-only builds.
     * If d_manage_stream_deletion is true, and we have a GPU-enabled build,
     * then this stream will be destroyed in the AccelerationContext destructor
     */
    ComputeStream d_stream;

    //! Flag for ownership of compute stream
    bool d_manage_stream_deletion;

#ifdef AMP_USE_KOKKOS
    /*!
     * Default Kokkos execution space, matched to d_compute_stream when GPU-enabled.
     * This class must be static constructible but Kokkos execution spaces can not
     * be constructed before Kokkos::initialize is called. To get around this they
     * are stored in optionals that default to empty, then get replaced after
     * initialization is done.
     */
    std::optional<Kokkos::DefaultExecutionSpace> d_kokkos_exec_default;

    //! Default Kokkos host execution space, same as d_kokkos_exec_default for host-only builds
    std::optional<Kokkos::DefaultHostExecutionSpace> d_kokkos_exec_host;
#endif
};


} // namespace AMP::Utilities

#endif
