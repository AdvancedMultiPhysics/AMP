#ifndef included_AMP_GPUFunctionTable_H_
#define included_AMP_GPUFunctionTable_H_

#include "AMP/utils/ArraySize.h"

#include <memory>


namespace AMP {


/*!
 * Class GPUFunctionTable is an accelerated function table class that defines
 *   a series of operations that can be performed on the Array class.
 *   The class implements the same interface as the serial FunctionTable class
 *   Meant to be used with a GPU Allocator class.
 */
class GPUFunctionTable final
{
public:
    /*!
     * Initialize the array with random values
     * @param[in] x         The array to operate on
     */
    template<class TYPE, class FUN, class ALLOC>
    static void rand( Array<TYPE, FUN, ALLOC> &x );

    /*! NOT IMPLEMENTED
     * Perform a reduce operator y = f(x)
     * @param[in] op            The function operation
     *                          Note: the operator is a template parameter to improve performance
     * @param[in] A             The array to operate on
     * @param[in] initialValue  The initial value for the reduction (0 for sum, +/- inf for min/max,
     * ...)
     * @return                  The reduction
     */
    template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
    static TYPE reduce( LAMBDA &op, const Array<TYPE, FUN, ALLOC> &A, const TYPE &initialValue );

    /*! NOT IMPLEMENTED
     * Perform a reduce operator z = f(x,y)
     * @param[in] op            The function operation
     *                          Note: the operator is a template parameter to improve performance
     * @param[in] A             The first array to operate on
     * @param[in] B             The second array to operate on
     * @param[in] initialValue  The initial value for the reduction (0 for sum, +/- inf for min/max,
     * ...)
     * @return                  The reduction
     */
    template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
    static TYPE reduce( LAMBDA &op,
                        const Array<TYPE, FUN, ALLOC> &A,
                        const Array<TYPE, FUN, ALLOC> &B,
                        const TYPE &initialValue );

    /*! NOT IMPLEMENTED
     * Perform a element-wise operation y = f(x)
     * @param[in] fun           The function operation
     *                          Note: the function is a template parameter to improve performance
     * @param[in,out] x         The array to operate on
     * @param[out] y            The output array
     */
    template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
    static void
    transform( LAMBDA &fun, const Array<TYPE, FUN, ALLOC> &x, Array<TYPE, FUN, ALLOC> &y );

    /*! NOT IMPLEMENTED
     * Perform a element-wise operation z = f(x,y)
     * @param[in] fun           The function operation
     *                          Note: the function is a template parameter to improve performance
     * @param[in] x             The first array
     * @param[in] y             The second array
     * @param[out] z            The output array
     */
    template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
    static void transform( LAMBDA &fun,
                           const Array<TYPE, FUN, ALLOC> &x,
                           const Array<TYPE, FUN, ALLOC> &y,
                           Array<TYPE, FUN, ALLOC> &z );

    /*! NOT IMPLEMENTED
     * Multiply two arrays
     * @param[in] a             The first array
     * @param[in] b             The second array
     * @param[out] c            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void multiply( const Array<TYPE, FUN, ALLOC> &a,
                          const Array<TYPE, FUN, ALLOC> &b,
                          Array<TYPE, FUN, ALLOC> &c );

    /*!
     * Check if two arrays are approximately equal
     * @param[in] A             The first array
     * @param[in] B             The second array
     * @param[in] tol           The tolerance
     */
    template<class TYPE, class FUN, class ALLOC>
    static bool
    equals( const Array<TYPE, FUN, ALLOC> &A, const Array<TYPE, FUN, ALLOC> &B, TYPE tol );

    /*!
     * Perform a element-wise operation y = max(x , 0)
     * @param[in] A             The input array
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformReLU( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Perform a element-wise operation B = |A|
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformAbs( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Perform a element-wise operation B = tanh(A)
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformTanh( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Perform a element-wise operation B = max(-1 , min(1 , A) )
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformHardTanh( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Perform a element-wise operation B = 1 / (1 + exp(-A))
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformSigmoid( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Perform a element-wise operation B = log(exp(A) + 1)
     * @param[in] A             The array to operate on
     * @param[out] B            The output array
     */
    template<class TYPE, class FUN, class ALLOC>
    static void transformSoftPlus( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B );

    /*!
     * Sum the elements of the Array
     * @param[in] A             The array to sum
     */
    template<class TYPE, class FUN, class ALLOC>
    static TYPE sum( const Array<TYPE, FUN, ALLOC> &A );


    GPUFunctionTable(){};

private:
    template<class TYPE>
    static void rand( size_t N, TYPE *x );
};
} // namespace AMP


#endif
