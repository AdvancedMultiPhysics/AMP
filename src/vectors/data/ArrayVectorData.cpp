#include "AMP/vectors/data/ArrayVectorData.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/memory.h"


// Explicit instantiations
template class AMP::LinearAlgebra::ArrayVectorData<double>;
template class AMP::LinearAlgebra::ArrayVectorData<float>;

#ifdef AMP_USE_DEVICE
    #include "AMP/utils/device/GPUFunctionTable.h"
template class AMP::LinearAlgebra::
    ArrayVectorData<double, AMP::GPUFunctionTable, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::
    ArrayVectorData<float, AMP::GPUFunctionTable, AMP::DeviceAllocator<void>>;
template class AMP::LinearAlgebra::
    ArrayVectorData<double, AMP::GPUFunctionTable, AMP::ManagedAllocator<void>>;
template class AMP::LinearAlgebra::
    ArrayVectorData<float, AMP::GPUFunctionTable, AMP::ManagedAllocator<void>>;
#endif
