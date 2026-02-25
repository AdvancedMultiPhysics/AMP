#include "AMP/utils/copycast/CopyCastHelper.h"
#include "CopyCastHelper.hpp"

namespace AMP::Utilities {

template struct copyCast_<double,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::HostAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::HostAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::HostAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::HostAllocator<void>>;

template struct copyCast_<double,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::ManagedAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::ManagedAllocator<void>>;

template struct copyCast_<double,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::DeviceAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::Backend::Hip_Cuda,
                          AMP::DeviceAllocator<void>>;

} // namespace AMP::Utilities
