#ifndef included_AMP_DeviceOperationsHelpers_h
#define included_AMP_DeviceOperationsHelpers_h

#include "AMP/utils/AccelerationContext.h"

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
    static void setRandomValues( size_t N, TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static void
    scale( TYPE alpha, size_t N, const TYPE *x, TYPE *y, AMP::Utilities::AccelerationContext &ctx );

    static void scale( TYPE alpha, size_t N, TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static void add(
        size_t N, const TYPE *x, const TYPE *y, TYPE *z, AMP::Utilities::AccelerationContext &ctx );

    static void subtract(
        size_t N, const TYPE *x, const TYPE *y, TYPE *z, AMP::Utilities::AccelerationContext &ctx );

    static void multiply(
        size_t N, const TYPE *x, const TYPE *y, TYPE *z, AMP::Utilities::AccelerationContext &ctx );

    static void divide(
        size_t N, const TYPE *x, const TYPE *y, TYPE *z, AMP::Utilities::AccelerationContext &ctx );

    static void
    reciprocal( size_t N, const TYPE *x, TYPE *y, AMP::Utilities::AccelerationContext &ctx );

    static void linearSum( const TYPE alpha,
                           size_t N,
                           const TYPE *x,
                           const TYPE beta,
                           const TYPE *y,
                           TYPE *z,
                           AMP::Utilities::AccelerationContext &ctx );

    static void abs( size_t N, const TYPE *x, TYPE *z, AMP::Utilities::AccelerationContext &ctx );

    static void addScalar(
        size_t N, const TYPE *x, TYPE alpha_in, TYPE *y, AMP::Utilities::AccelerationContext &ctx );

    static void setMax( size_t N, TYPE val, TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static void setMin( size_t N, TYPE val, TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localMin( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localMax( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localSum( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localL1Norm( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localL2Norm2( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localMaxNorm( size_t N, const TYPE *x, AMP::Utilities::AccelerationContext &ctx );

    static TYPE
    localDot( size_t N, const TYPE *x, const TYPE *y, AMP::Utilities::AccelerationContext &ctx );

    static TYPE localMinQuotient( size_t N,
                                  const TYPE *x,
                                  const TYPE *y,
                                  AMP::Utilities::AccelerationContext &ctx );

    static TYPE localWrmsNorm( size_t N,
                               const TYPE *x,
                               const TYPE *y,
                               AMP::Utilities::AccelerationContext &ctx );
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
