#ifndef included_AMP_Algorithms
#define included_AMP_Algorithms

#include "AMP/utils/Memory.h"

namespace AMP {
namespace Utilities {
namespace Algorithms {

template<typename TYPE>
void fill_n( TYPE *x,
             const size_t N,
             const TYPE alpha,
             const MemoryType mem_loc,
             const computeStream_t stream );

template<typename TYPE>
void zero_n( TYPE *x, const size_t N, const MemoryType mem_loc, const computeStream_t stream );

template<typename TYPE>
void copy_n( TYPE *dst,
             const TYPE *src,
             const size_t N,
             const MemoryType mem_loc,
             const computeStream_t stream );

template<typename TYPE>
void copy_n( TYPE *dst,
             const MemoryType dst_loc,
             const TYPE *src,
             const MemoryType src_loc,
             const size_t N,
             const computeStream_t stream );

template<class TDst, class TSrc>
void copyCast( TDst *dst,
               const MemoryType dst_loc,
               const TSrc *src,
               const MemoryType src_loc,
               size_t N,
               const computeStream_t stream );

template<typename TYPE>
void exclusive_scan( const TYPE *x,
                     const size_t N,
                     TYPE *y,
                     const TYPE alpha,
                     const MemoryType mem_loc,
                     const computeStream_t stream );

template<typename TYPE>
void inclusive_scan( const TYPE *x,
                     const size_t N,
                     TYPE *y,
                     const MemoryType mem_loc,
                     const computeStream_t stream );

template<typename TYPE>
void sort( TYPE *x, const size_t N, const MemoryType mem_loc, const computeStream_t stream );

template<typename TYPE>
TYPE min_element( const TYPE *x,
                  const size_t N,
                  const MemoryType mem_loc,
                  const computeStream_t stream );

template<typename TYPE>
TYPE max_element( const TYPE *x,
                  const size_t N,
                  const MemoryType mem_loc,
                  const computeStream_t stream );

template<typename TYPE>
TYPE accumulate( const TYPE *x,
                 const size_t N,
                 TYPE alpha,
                 const MemoryType mem_loc,
                 const computeStream_t stream );

template<typename TYPE>
size_t unique( TYPE *x, const size_t N, const MemoryType mem_loc, const computeStream_t stream );

} // namespace Algorithms
} // namespace Utilities
} // namespace AMP

#endif
