#ifndef included_AMP_DeviceOperationsHelpers_h
#define included_AMP_DeviceOperationsHelpers_h


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
    static void scale( TYPE alpha, size_t N, const TYPE *x, TYPE *y );
    static void scale( TYPE alpha, size_t N, TYPE *x );
    static void add( size_t N, const TYPE *x, const TYPE *y, TYPE *z );
    static void subtract( size_t N, const TYPE *x, const TYPE *y, TYPE *z );
    static void multiply( size_t N, const TYPE *x, const TYPE *y, TYPE *z );
    static void divide( size_t N, const TYPE *x, const TYPE *y, TYPE *z );
    static void reciprocal( size_t N, const TYPE *x, TYPE *y );
    static void linearSum( TYPE alpha, size_t N, const TYPE *x, TYPE beta, const TYPE *y, TYPE *z );
    static void abs( size_t N, const TYPE *x, TYPE *z );
    static void addScalar( size_t N, const TYPE *x, TYPE alpha_in, TYPE *y );
    static void setMax( size_t N, TYPE val, TYPE *x );
    static void setMin( size_t N, TYPE val, TYPE *x );
    static TYPE localMin( size_t N, const TYPE *x );
    static TYPE localMax( size_t N, const TYPE *x );
    static TYPE localSum( size_t N, const TYPE *x );
    static TYPE localL1Norm( size_t N, const TYPE *x );
    static TYPE localL2Norm( size_t N, const TYPE *x );
    static TYPE localMaxNorm( size_t N, const TYPE *x );
    static TYPE localDot( size_t N, const TYPE *x, const TYPE *y );
    static TYPE localMinQuotient( size_t N, const TYPE *x, const TYPE *y );
    static TYPE localWrmsNorm( size_t N, const TYPE *x, const TYPE *y );
};


} // namespace LinearAlgebra
} // namespace AMP


#endif
