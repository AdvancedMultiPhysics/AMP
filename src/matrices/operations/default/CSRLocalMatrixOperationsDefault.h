#ifndef included_CSRLocalMatrixOperationsDefault_H_
#define included_CSRLocalMatrixOperationsDefault_H_

#include "AMP/matrices/operations/MatrixOperations.h"

namespace AMP::LinearAlgebra {

template<typename Policy,
         class Allocator,
         class LocalMatrixData = CSRLocalMatrixData<Policy, Allocator>>
class CSRLocalMatrixOperationsDefault
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

public:
    /** \brief  Matrix-vector multiplication
     * \param[in]  x  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] y  The resulting vectory
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( const scalar_t *in, std::shared_ptr<LocalMatrixData> A, scalar_t *out );

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( const scalar_t *in,
                        std::shared_ptr<LocalMatrixData> A,
                        std::vector<scalar_t> &vvals,
                        std::vector<size_t> &rcols );

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( scalar_t alpha, std::shared_ptr<LocalMatrixData> A );

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \param[out] Y The output matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void
    axpy( scalar_t alpha, std::shared_ptr<LocalMatrixData> X, std::shared_ptr<LocalMatrixData> Y );

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     * \param[out] A The input matrix A
     */
    void setScalar( scalar_t alpha, std::shared_ptr<LocalMatrixData> A );

    /** \brief  Set the non-zeros of the matrix to zero
     * \details  May not deallocate space.
     * \param[in] A The input matrix A
     */
    void zero( std::shared_ptr<LocalMatrixData> A );

    /** \brief  Set the diagonal to the values in a vector
     * \param[in] in The values to set the diagonal to
     * \param[out] A The matrix to set
     */
    void setDiagonal( const scalar_t *in, std::shared_ptr<LocalMatrixData> A );

    /** \brief Extract the diagonal values into a vector
     * \param[in] in The values to set the diagonal to
     * \param[in] A The matrix to set
     */
    void extractDiagonal( std::shared_ptr<LocalMatrixData> A, scalar_t *buf );

    /** \brief  Set the matrix to the identity matrix
     * \param[out] A The matrix to set
     */
    void setIdentity( std::shared_ptr<LocalMatrixData> A );

    /** \brief Compute the maximum row sum
     * \return  The L-infinity norm of the matrix
     * \param[in] A Data for the input matrix
     */
    void LinfNorm( std::shared_ptr<LocalMatrixData> A, scalar_t *rowSums ) const;
};

} // namespace AMP::LinearAlgebra

#endif
