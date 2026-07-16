#ifndef included_AMP_DeviceOperationsHelpers_h
#define included_AMP_DeviceOperationsHelpers_h

#include "AMP/utils/device/Device.h"

namespace AMP {
namespace LinearAlgebra {

/**
 * \brief  A default set of helper functions for vector operations
 * \details OperationsHelpers impliments a default set of
 *    vector operations on the GPU.
 */
template<typename TYPE>
class DeviceOperationsHelpers
{
public:
    //  functions that operate on VectorData
    static void setRandomValues( size_t N, TYPE *x, const AMP::Utilities::ComputeStream stream );

    static void scale(
        TYPE alpha, size_t N, const TYPE *x, TYPE *y, const AMP::Utilities::ComputeStream stream );

    static void scale( TYPE alpha, size_t N, TYPE *x, const AMP::Utilities::ComputeStream stream );

    static void add( size_t N,
                     const TYPE *x,
                     const TYPE *y,
                     TYPE *z,
                     const AMP::Utilities::ComputeStream stream );

    static void subtract( size_t N,
                          const TYPE *x,
                          const TYPE *y,
                          TYPE *z,
                          const AMP::Utilities::ComputeStream stream );

    static void multiply( size_t N,
                          const TYPE *x,
                          const TYPE *y,
                          TYPE *z,
                          const AMP::Utilities::ComputeStream stream );

    static void divide( size_t N,
                        const TYPE *x,
                        const TYPE *y,
                        TYPE *z,
                        const AMP::Utilities::ComputeStream stream );

    static void
    reciprocal( size_t N, const TYPE *x, TYPE *y, const AMP::Utilities::ComputeStream stream );

    static void linearSum( const TYPE alpha,
                           size_t N,
                           const TYPE *x,
                           const TYPE beta,
                           const TYPE *y,
                           TYPE *z,
                           const AMP::Utilities::ComputeStream stream );

    static void abs( size_t N, const TYPE *x, TYPE *z, const AMP::Utilities::ComputeStream stream );

    static void addScalar( size_t N,
                           const TYPE *x,
                           TYPE alpha_in,
                           TYPE *y,
                           const AMP::Utilities::ComputeStream stream );

    static void setMax( size_t N, TYPE val, TYPE *x, const AMP::Utilities::ComputeStream stream );

    static void setMin( size_t N, TYPE val, TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localMin( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localMax( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localSum( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localL1Norm( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localL2Norm2( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE localMaxNorm( size_t N, const TYPE *x, const AMP::Utilities::ComputeStream stream );

    static TYPE
    localDot( size_t N, const TYPE *x, const TYPE *y, const AMP::Utilities::ComputeStream stream );

    static TYPE localMinQuotient( size_t N,
                                  const TYPE *x,
                                  const TYPE *y,
                                  const AMP::Utilities::ComputeStream stream );

    static TYPE localWrmsNorm( size_t N,
                               const TYPE *x,
                               const TYPE *y,
                               const AMP::Utilities::ComputeStream stream );
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
