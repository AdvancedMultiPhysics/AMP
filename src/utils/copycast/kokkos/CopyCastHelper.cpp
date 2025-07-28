#include "AMP/utils/copycast/CopyCastHelper.h"
#include "CopyCastHelper.hpp"

namespace AMP::Utilities {

template struct copyCast_<double, float, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<float, double, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<float, float, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Kokkos,
                          AMP::HostAllocator<void>>;

#ifdef USE_DEVICE
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
