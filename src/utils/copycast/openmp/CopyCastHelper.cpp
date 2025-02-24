#include "AMP/utils/memory.h"

#include "AMP/utils/copycast/CopyCastHelper.h"
#include "CopyCastHelper.hpp"

namespace AMP::Utilities {
    
template struct copyCast_<double,
                          float,
                          AMP::Utilities::PortabilityBackend::OpenMP,
                          AMP::HostAllocator<void>>;
template struct copyCast_<float,
                          double,
                          AMP::Utilities::PortabilityBackend::OpenMP,
                          AMP::HostAllocator<void>>;
template struct copyCast_<double,
                          double,
                          AMP::Utilities::PortabilityBackend::OpenMP,
                          AMP::HostAllocator<void>>;
template struct copyCast_<float,
                          float,
                          AMP::Utilities::PortabilityBackend::OpenMP,
                          AMP::HostAllocator<void>>;

} // namespace AMP::Utilities