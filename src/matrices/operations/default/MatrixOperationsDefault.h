#ifndef included_MatrixOperationsDefault_H_
#define included_MatrixOperationsDefault_H_

#include "AMP/vectors/Scalar.h"

namespace AMP::LinearAlgebra {

class MatrixData;
struct MatrixOperationsDefault {
    /** \brief  Compute the product of two matrices
     * \param[in] A  Left multiplicand
     * \param[in] B  Right multiplicand
     * \param[in] C  The product \f$\mathbf{AB}\f$.
     */
    static void matMatMult( MatrixData const &A, MatrixData const &B, MatrixData &C );

    /** \brief  Compute the linear combination of two matrices
     * \param[in]  alpha  scalar
     * \param[in]  X      matrix
     * \param[out] Y      The output matrix
     * \details  Compute \f$\mathbf{Y} = \alpha\mathbf{X} + \mathbf{Y}\f$
     */
    static void axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y );

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     * \param[in] X matrix data to copy from
     * \param[in] Y matrix data to copy to
     */
    static void copy( const MatrixData &X, MatrixData &Y );
};

} // namespace AMP::LinearAlgebra

#endif
