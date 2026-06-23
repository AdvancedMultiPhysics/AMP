#ifndef included_AMP_Algorithms
#define included_AMP_Algorithms

#include "AMP/utils/Memory.h"

namespace AMP {
namespace Utilities {

class Algorithms
{
public:
    template<typename TYPE>
    static void fill_n( TYPE *x, const size_t N, const TYPE alpha, const MemoryType mem_loc );

    template<typename TYPE>
    static void zero_n( TYPE *x, const size_t N, const MemoryType mem_loc );

    template<typename TYPE>
    static void copy_n( TYPE *dst, const TYPE *src, const size_t N, const MemoryType mem_loc );

    template<typename TYPE>
    static void copy_n( TYPE *dst,
                        const MemoryType dst_loc,
                        const TYPE *src,
                        const MemoryType src_loc,
                        const size_t N );

    template<class TDst, class TSrc>
    static void copyCast(
        TDst *dst, const MemoryType dst_loc, const TSrc *src, const MemoryType src_loc, size_t N );

    template<typename TYPE>
    static void exclusive_scan(
        const TYPE *x, const size_t N, TYPE *y, const TYPE alpha, const MemoryType mem_loc );

    template<typename TYPE>
    static void inclusive_scan( const TYPE *x, const size_t N, TYPE *y, const MemoryType mem_loc );

    template<typename TYPE>
    static void sort( TYPE *x, const size_t N, const MemoryType mem_loc );

    template<typename TYPE>
    static TYPE min_element( const TYPE *x, const size_t N, const MemoryType mem_loc );

    template<typename TYPE>
    static TYPE max_element( const TYPE *x, const size_t N, const MemoryType mem_loc );

    template<typename TYPE>
    static TYPE accumulate( const TYPE *x, const size_t N, TYPE alpha, const MemoryType mem_loc );

    template<typename TYPE>
    static size_t unique( TYPE *x, const size_t N, const MemoryType mem_loc );
};

} // namespace Utilities
} // namespace AMP

#endif
