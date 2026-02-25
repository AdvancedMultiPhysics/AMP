#include "AMP/utils/copycast/CopyCastHelper.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/copycast/kokkos/CopyCastHelper.hpp"


namespace AMP::Utilities {


template struct copyCast_<double, float, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<float, double, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<float, float, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::HostAllocator<void>>;

#ifdef AMP_USE_DEVICE
template struct copyCast_<double,
                          float,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::ManagedAllocator<void>>;

template struct copyCast_<double,
                          float,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::DeviceAllocator<void>>;
#endif


} // namespace AMP::Utilities
