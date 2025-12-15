#include "AMP/vectors/data/ArrayVectorData.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Memory.h"


// Explicit instantiations
template class AMP::LinearAlgebra::ArrayVectorData<double>;
template class AMP::LinearAlgebra::ArrayVectorData<float>;

#ifdef AMP_USE_DEVICE
    #include "AMP/utils/device/GPUFunctionTable.h"
    #define INSTANTIATE( T, ALLOC ) \
        template class AMP::LinearAlgebra::ArrayVectorData<T, AMP::GPUFunctionTable<T>, ALLOC<T>>
INSTANTIATE( float, AMP::DeviceAllocator );
INSTANTIATE( float, AMP::ManagedAllocator );
INSTANTIATE( double, AMP::DeviceAllocator );
INSTANTIATE( double, AMP::ManagedAllocator );
#endif
