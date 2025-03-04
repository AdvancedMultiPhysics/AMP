#ifndef included_CSRMatrixOperationsDefault_H_
#define included_CSRMatrixOperationsDefault_H_

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/matrices/operations/default/CSRLocalMatrixOperationsDefault.h"
#include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.h"
#include "AMP/vectors/Vector.h"

#include <map>

namespace AMP::LinearAlgebra {

template<typename Policy,
         class Allocator,
         class DiagMatrixData = CSRLocalMatrixData<Policy, Allocator>>
class CSRMatrixOperationsDefault : public MatrixOperations
{
public:
    CSRMatrixOperationsDefault()
        : d_localops_diag( std::make_shared<
                           CSRLocalMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>>() ),
          d_localops_offd( std::make_shared<CSRLocalMatrixOperationsDefault<Policy, Allocator>>() )
    {
    }

    /** \brief  Matrix-vector multiplication
     * \param[in]  x  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] y  The resulting vectory
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( std::shared_ptr<const Vector> x,
               MatrixData const &A,
               std::shared_ptr<Vector> y ) override;

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( std::shared_ptr<const Vector> in,
                        MatrixData const &A,
                        std::shared_ptr<Vector> out ) override;

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Compute the product of two matrices
     * \param[in] A  A multiplicand
     * \param[in] B  A multiplicand
     * \param[in] C  The product \f$\mathbf{AB}\f$.
     */
    void matMultiply( MatrixData const &A, MatrixData const &B, MatrixData &C ) override;

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \param[out] Y The output matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y ) override;

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients after up/down casting
     * \param[in] x matrix data to copy from
     * \param[in] y matrix data to copy to after up/down casting the coefficients
     */
    void copyCast( const MatrixData &X, MatrixData &Y ) override;

    template<typename PolicyIn>
    static void
    copyCast( CSRMatrixData<PolicyIn, Allocator, CSRLocalMatrixData<PolicyIn, Allocator>> *X,
              CSRMatrixData<Policy, Allocator, DiagMatrixData> *Y );

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     * \param[out] A The input matrix A
     */
    void setScalar( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Set the non-zeros of the matrix to zero
     * \details  May not deallocate space.
     * \param[in] A The input matrix A
     */
    void zero( MatrixData &A ) override;

    /** \brief  Set the diagonal to the values in a vector
     * \param[in] in The values to set the diagonal to
     * \param[out] A The matrix to set
     */
    void setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A ) override;

    /** \brief Extract the diagonal values into a vector
     * \param[in] in The values to set the diagonal to
     * \param[in] A The matrix to set
     */
    void extractDiagonal( MatrixData const &A, std::shared_ptr<Vector> buf ) override;

    /** \brief  Set the matrix to the identity matrix
     * \param[out] A The matrix to set
     */
    void setIdentity( MatrixData &A ) override;

    /** \brief Compute the maximum row sum
     * \return  The L-infinity norm of the matrix
     * \param[in] X Data for the input matrix
     */
    AMP::Scalar LinfNorm( const MatrixData &X ) const override;

protected:
    std::shared_ptr<CSRLocalMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>>
        d_localops_diag;
    std::shared_ptr<CSRLocalMatrixOperationsDefault<Policy, Allocator>> d_localops_offd;
    std::map<std::pair<CSRMatrixData<Policy, Allocator, DiagMatrixData> *,
                       CSRMatrixData<Policy, Allocator, DiagMatrixData> *>,
             CSRMatrixSpGEMMHelperDefault<Policy, Allocator, DiagMatrixData>>
        d_SpGEMMHelpers;
};

} // namespace AMP::LinearAlgebra

#endif
