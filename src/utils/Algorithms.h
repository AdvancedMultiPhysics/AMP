#ifndef included_AMP_Algorithms
#define included_AMP_Algorithms

#include <cstddef>

namespace AMP {
namespace Utilities {

template<typename TYPE>
class Algorithms
{
public:
    static void fill_n( TYPE *x, const size_t N, const TYPE alpha );
    static void copy_n( const TYPE *x, const size_t N, TYPE *y );
    static void exclusive_scan( TYPE *x, const size_t N, TYPE *y, const TYPE alpha );
    static void inclusive_scan( TYPE *x, const size_t N, TYPE *y );
    static TYPE max_element( TYPE *x, const size_t N );
    static TYPE accumulate( TYPE *x, const size_t N, TYPE alpha );
};

} // namespace Utilities
} // namespace AMP

#endif
