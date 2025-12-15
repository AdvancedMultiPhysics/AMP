#include "AMP/vectors/data/device/VectorDataDevice.h"
#include "AMP/vectors/data/device/VectorDataDevice.hpp"

template class AMP::LinearAlgebra::VectorDataDevice<double, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDevice<float, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDevice<double, AMP::ManagedAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDevice<float, AMP::ManagedAllocator<void>>;
