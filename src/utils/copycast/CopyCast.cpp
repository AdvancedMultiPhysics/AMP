#include "AMP/utils/copycast/CopyCast.hpp"
#include "AMP/AMP_TPLs.h"

namespace AMP::Utilities {


#define INSTANTIATE_CC_M( t1, t2, backend, allocator ) \
    template void copyCast<t1, t2, backend, allocator>( size_t len, const t1 *vec_in, t2 *vec_out );


#define INSTANTIATE_CC( t1, t2, backend ) \
    template void copyCast<t1, t2, backend>( size_t len, const t1 *vec_in, t2 *vec_out );

#ifdef AMP_USE_DEVICE
    #define INSTANTIATE_ALLOCATOR( t1, t2, backend )                     \
        INSTANTIATE_CC_M( t1, t2, backend, AMP::HostAllocator<void> )    \
        INSTANTIATE_CC_M( t1, t2, backend, AMP::ManagedAllocator<void> ) \
        INSTANTIATE_CC_M( t1, t2, backend, AMP::DeviceAllocator<void> )  \
        INSTANTIATE_CC( t1, t2, backend )
#else
    #define INSTANTIATE_ALLOCATOR( t1, t2, backend )                  \
        INSTANTIATE_CC_M( t1, t2, backend, AMP::HostAllocator<void> ) \
        INSTANTIATE_CC( t1, t2, backend )
#endif

#define INSTANTIATE_TYPES( backend )                \
    INSTANTIATE_ALLOCATOR( float, double, backend ) \
    INSTANTIATE_ALLOCATOR( double, float, backend ) \
    INSTANTIATE_ALLOCATOR( float, float, backend )  \
    INSTANTIATE_ALLOCATOR( double, double, backend )

INSTANTIATE_TYPES( AMP::Utilities::Backend::Serial )
#ifdef AMP_USE_OPENMP
INSTANTIATE_TYPES( AMP::Utilities::Backend::OpenMP )
#endif
#ifdef AMP_USE_KOKKOS
INSTANTIATE_TYPES( AMP::Utilities::Backend::Kokkos )
#endif
#ifdef AMP_USE_DEVICE
INSTANTIATE_TYPES( AMP::Utilities::Backend::Hip_Cuda )
#endif


} // namespace AMP::Utilities

