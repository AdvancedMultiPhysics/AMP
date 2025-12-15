#include "AMP/utils/Array.hpp"
#include "AMP/utils/cuda/CudaAllocator.h"
#include "AMP/utils/cuda/GPUFunctionTable.h"


/********************************************************
 *  Explicit instantiations of Array                     *
 ********************************************************/
#define INSTANTIATE( T, ALLOC ) template class AMP::Array<T, AMP::GPUFunctionTable<T>, ALLOC<T>>
INSTANTIATE( float, AMP::CudaDevAllocator );
INSTANTIATE( float, AMP::CudaManagedAllocator );
INSTANTIATE( double, AMP::CudaDevAllocator );
INSTANTIATE( double, AMP::CudaManagedAllocator );
