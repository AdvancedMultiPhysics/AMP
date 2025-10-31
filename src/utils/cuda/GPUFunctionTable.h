#ifndef included_AMP_GPUFunctionTable
#define included_AMP_GPUFunctionTable


#include "AMP/utils/ArraySize.h"

#include <memory>


namespace AMP {


/*!
 * Class GPUFunctionTable is an accelerated function table class that defines
 *   a series of operations that can be performed on the Array class.
 *   The class implements the same interface as the serial FunctionTable class
 *   Meant to be used with a GPU Allocator class.
 */
template<class TYPE>
class GPUFunctionTable final
{
public:
    /*!
     * Initialize the array with random values
     * @param[in] N         The length of the array
     * @param[in] x         The array to operate on
     */
    static void rand( size_t N, TYPE *x );

    /*!
     * Perform a reduce operator y = f(x)
     * @param[in] op            The function operation
     *                          Note: the operator is a template parameter to improve performance
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[in] initialValue  The initial value for the reduction (0 for sum, +/- inf for min/max)
     * @return                  The reduction
     */
    template<typename LAMBDA>
    static TYPE reduce( LAMBDA &op, size_t N, const TYPE *A, TYPE initialValue )
    {
        AMP_ERROR( "Not implemented for GPU" );
    }

    /*!
     * Perform a reduce operator z = f(x,y)
     * @param[in] op            The function operation
     *                          Note: the operator is a template parameter to improve performance
     * @param[in] N             The length of the array
     * @param[in] A             The first array to operate on
     * @param[in] B             The second array to operate on
     * @param[in] initialValue  The initial value for the reduction (0 for sum, +/- inf for min/max)
     * @return                  The reduction
     */
    template<typename LAMBDA>
    static TYPE reduce( LAMBDA &op, size_t N, const TYPE *A, const TYPE *B, TYPE initialValue )
    {
        AMP_ERROR( "Not implemented for GPU" );
    }

    /*!
     * Perform a element-wise operation y = f(x)
     * @param[in] fun           The function operation
     *                          Note: the function is a template parameter to improve performance
     * @param[in] N             The length of the array
     * @param[in,out] x         The array to operate on
     * @param[out] y            The output array
     */
    template<typename LAMBDA>
    static void transform( LAMBDA &fun, size_t N, const TYPE *x, TYPE *y )
    {
        AMP_ERROR( "Not implemented for GPU" );
    }

    /*!
     * Perform a element-wise operation z = f(x,y)
     * @param[in] fun           The function operation
     *                          Note: the function is a template parameter to improve performance
     * @param[in] N             The length of the array
     * @param[in] x             The first array
     * @param[in] y             The second array
     * @param[out] z            The output array
     */
    template<typename LAMBDA>
    static void transform( LAMBDA &fun, size_t N, const TYPE *x, const TYPE *y, TYPE *z )
    {
        AMP_ERROR( "Not implemented for GPU" );
    }

    /*!
     * Return the minimum value
     * @param[in] N             The length of the array
     * @param[in] x             The first array
     */
    static TYPE min( size_t N, const TYPE *x );

    /*!
     * Return the maximum value
     * @param[in] N             The length of the array
     * @param[in] x             The first array
     */
    static TYPE max( size_t N, const TYPE *x );

    /*!
     * Return the sum
     * @param[in] N             The length of the array
     * @param[in] x             The first array
     */
    static TYPE sum( size_t N, const TYPE *x );

    /*!
     * Multiply two arrays
     * @param[in] sa            The size of the a array
     * @param[in] a             The first array
     * @param[in] sb            The size of the b array
     * @param[in] b             The second array
     * @param[in] sc            The size of the c array
     * @param[out] c            The output array
     */
    static void multiply( const ArraySize &sa,
                          const TYPE *a,
                          const ArraySize &sb,
                          const TYPE *b,
                          const ArraySize &sc,
                          TYPE *c );

    /*!
     * Perform addition operation ( y += x )
     * @param[in] N             The length of the array
     * @param[in] x             The scalar value alpha
     * @param[in,out] y         The output array y
     */
    static void px( size_t N, TYPE x, TYPE *y );

    /*!
     * Perform addition operation ( y += x )
     * @param[in] N             The length of the array
     * @param[in] x             The scalar value alpha
     * @param[in,out] y         The output array y
     */
    static void px( size_t N, const TYPE *x, TYPE *y );

    /*!
     * Perform subtraction operation ( y += x )
     * @param[in] N             The length of the array
     * @param[in] x             The scalar value alpha
     * @param[in,out] y         The output array y
     */
    static void mx( size_t N, TYPE x, TYPE *y );

    /*!
     * Perform subtraction operation ( y += x )
     * @param[in] x             The scalar value alpha
     * @param[in] N             The length of the array
     * @param[in,out] y         The output array y
     */
    static void mx( size_t N, const TYPE *x, TYPE *y );

    /*!
     * Perform axpy equavalent operation ( y = alpha*x + y )
     * @param[in] alpha         The scalar value alpha
     * @param[in] N             The length of the array
     * @param[in] x             The input array x
     * @param[in,out] y         The output array y
     */
    static void axpy( TYPE alpha, size_t N, const TYPE *x, TYPE *y );

    /*!
     * Perform axpby equavalent operation ( y = alpha*x + beta*y )
     * @param[in] alpha         The scalar value alpha
     * @param[in] N             The length of the array
     * @param[in] x             The input array x
     * @param[in] beta         The scalar value alpha
     * @param[in,out] y         The output array y
     */
    static void axpby( TYPE alpha, size_t N, const TYPE *x, TYPE beta, TYPE *y );

    /*!
     * Check if two arrays are approximately equal
     * @param[in] N             The length of the array
     * @param[in] A             The first array
     * @param[in] B             The second array
     * @param[in] tol           The tolerance
     */
    static bool equals( size_t N, const TYPE *A, const TYPE *B, TYPE tol );


    /* Specialized Functions */

    /*!
     * Perform a element-wise operation y = max(x , 0)
     * @param[in] N             The length of the array
     * @param[in] A             The input array
     * @param[out] B            The output array
     */
    static void transformReLU( size_t N, const TYPE *A, TYPE *B );

    /*!
     * Perform a element-wise operation B = |A|
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    static void transformAbs( size_t N, const TYPE *A, TYPE *B );

    /*!
     * Perform a element-wise operation B = tanh(A)
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    static void transformTanh( size_t N, const TYPE *A, TYPE *B );

    /*!
     * Perform a element-wise operation B = max(-1 , min(1 , A) )
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    static void transformHardTanh( size_t N, const TYPE *A, TYPE *B );

    /*!
     * Perform a element-wise operation B = 1 / (1 + exp(-A))
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    static void transformSigmoid( size_t N, const TYPE *A, TYPE *B );

    /*!
     * Perform a element-wise operation B = log(exp(A) + 1)
     * @param[in] N             The length of the array
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    static void transformSoftPlus( size_t N, const TYPE *A, TYPE *B );

private:
    GPUFunctionTable() = delete;
};
} // namespace AMP


#endif
