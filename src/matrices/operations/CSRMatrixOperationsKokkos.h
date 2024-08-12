#ifndef included_CSRMatrixOperationsKokkos_H_
#define included_CSRMatrixOperationsKokkos_H_

#include "AMP/matrices/operations/MatrixOperations.h"

namespace AMP::LinearAlgebra {

template<typename Policy, typename Allocator, class ExecSpace>
class CSRMatrixOperationsKokkos : public MatrixOperations
{

    /** \brief  Matrix-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( std::shared_ptr<const Vector> x,
               MatrixData const &A,
               std::shared_ptr<Vector> y ) override;

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( std::shared_ptr<const Vector> in,
                        MatrixData const &A,
                        std::shared_ptr<Vector> out ) override;

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Compute the product of two matrices
     * \param[in] A  A multiplicand
     * \param[in] B  A multiplicand
     * \return The product \f$\mathbf{AB}\f$.
     */
    void matMultiply( MatrixData const &A, MatrixData const &B, MatrixData &C ) override;

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y ) override;

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     */
    void setScalar( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Set the non-zeros of the matrix to zero
     * \details  May not deallocate space.
     */
    void zero( MatrixData &A ) override;

    /** \brief  Set the diagonal to the values in a vector
     * \param[in] in The values to set the diagonal to
     */
    void setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A ) override;

    /** \brief  Set the matrix to the identity matrix
     */
    void setIdentity( MatrixData &A ) override;

    /** \brief Compute the maximum column sum
     * \return  The L1 norm of the matrix
     */
    AMP::Scalar L1Norm( const MatrixData &X ) const override;

protected:
    ExecSpace d_exec_space;
};

} // namespace AMP::LinearAlgebra

#endif
