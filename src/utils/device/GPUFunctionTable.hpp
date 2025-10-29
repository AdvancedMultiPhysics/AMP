#include "AMP/AMP_TPLs.h"

#ifdef AMP_USE_CUDA
    #include "AMP/utils/cuda/GPUFunctionTable.hpp"
#endif
#ifdef AMP_USE_HIP
    #include "AMP/utils/hip/GPUFunctionTable.hpp"
#endif
