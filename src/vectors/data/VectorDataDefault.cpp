#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/data/VectorDataDefault.hpp"


// Explicit instantiations
template class AMP::LinearAlgebra::VectorDataDefault<double>;
template class AMP::LinearAlgebra::VectorDataDefault<float>;

#ifdef USE_DEVICE
template class AMP::LinearAlgebra::VectorDataDefault<double, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDefault<float, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDefault<double, AMP::ManagedAllocator<void>>;
template class AMP::LinearAlgebra::VectorDataDefault<float, AMP::ManagedAllocator<void>>;
#endif
