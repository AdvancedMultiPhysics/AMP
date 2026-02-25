#ifndef _DEVICE_H_INCLUDED_
#define _DEVICE_H_INCLUDED_

#include "AMP/AMP_TPLs.h"

#ifdef AMP_USE_CUDA
    #include "AMP/utils/cuda/Helper_Cuda.h"
#endif

#ifdef AMP_USE_HIP
    #include "AMP/utils/hip/Helper_Hip.h"
#endif

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>

#endif
