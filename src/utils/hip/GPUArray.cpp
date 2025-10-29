#include "AMP/utils/Array.hpp"
#include "AMP/utils/hip/GPUFunctionTable.hpp"
#include "AMP/utils/hip/HipAllocator.h"

namespace AMP {

/********************************************************
 *  Explicit instantiations of Array                     *
 ********************************************************/
template class Array<double, AMP::GPUFunctionTable, HipDevAllocator<double>>;
template class Array<double, AMP::GPUFunctionTable, HipManagedAllocator<double>>;
template class Array<float, AMP::GPUFunctionTable, HipDevAllocator<float>>;
template class Array<float, AMP::GPUFunctionTable, HipManagedAllocator<float>>;

} // namespace AMP
