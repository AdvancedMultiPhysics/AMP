#ifndef DEVICE_H_INCLUDED_
#define DEVICE_H_INCLUDED_

#include "AMP/AMP_TPLs.h"

#ifdef AMP_USE_DEVICE

    #ifdef AMP_USE_CUDA
        #include "AMP/utils/cuda/Helper_Cuda.h"
    #endif

    #ifdef AMP_USE_HIP
        #include "AMP/utils/hip/Helper_Hip.h"
    #endif

    #include <thrust/binary_search.h>
    #include <thrust/copy.h>
    #include <thrust/device_ptr.h>
    #include <thrust/device_vector.h>
    #include <thrust/execution_policy.h>
    #include <thrust/extrema.h>
    #include <thrust/fill.h>
    #include <thrust/for_each.h>
    #include <thrust/functional.h>
    #include <thrust/gather.h>
    #include <thrust/host_vector.h>
    #include <thrust/inner_product.h>
    #include <thrust/iterator/constant_iterator.h>
    #include <thrust/iterator/counting_iterator.h>
    #include <thrust/iterator/permutation_iterator.h>
    #include <thrust/iterator/transform_iterator.h>
    #include <thrust/iterator/zip_iterator.h>
    #include <thrust/logical.h>
    #include <thrust/mr/allocator.h>
    #include <thrust/random.h>
    #include <thrust/scan.h>
    #include <thrust/scatter.h>
    #include <thrust/sort.h>
    #include <thrust/transform.h>
    #include <thrust/transform_reduce.h>
    #include <thrust/tuple.h>
    #include <thrust/unique.h>

#else

// helpers for hip and cuda give a compute stream type
// with no device support just stub one in
typedef void *computeStream_t;

#endif

namespace AMP::Utilities {

//! struct for any persistent device information
struct DeviceContext {
    //! Compute stream
    computeStream_t stream;
};

// static instance of context struct as default option
extern DeviceContext device_context_default;

} // namespace AMP::Utilities

#endif
