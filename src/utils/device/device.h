#ifndef _DEVICE_H_INCLUDED_
#define _DEVICE_H_INCLUDED_

#ifdef USE_CUDA
    #include "AMP/utils/cuda/helper_cuda.h"
#endif

#ifdef USE_HIP
    #include "AMP/utils/hip/helper_hip.h"
#endif

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>

#endif
