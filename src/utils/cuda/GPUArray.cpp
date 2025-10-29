#include "AMP/utils/Array.hpp"
#include "AMP/utils/cuda/CudaAllocator.h"
#include "AMP/utils/cuda/GPUFunctionTable.hpp"


namespace AMP {


/********************************************************
 *  Explicit instantiations of Array                     *
 ********************************************************/
template class Array<double, AMP::GPUFunctionTable, CudaDevAllocator<double>>;
template class Array<float, AMP::GPUFunctionTable, CudaDevAllocator<float>>;
template class Array<double, AMP::GPUFunctionTable, CudaManagedAllocator<double>>;
template class Array<float, AMP::GPUFunctionTable, CudaManagedAllocator<float>>;


} // namespace AMP
