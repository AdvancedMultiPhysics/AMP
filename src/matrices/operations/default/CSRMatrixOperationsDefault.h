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

template<typename Config>
class CSRMatrixOperationsDefault : public MatrixOperations
{
public:
    using allocator_type = typename Config::allocator_type;
    static_assert( std::is_same_v<typename allocator_type::value_type, void> );

    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    using localops_t = CSRLocalMatrixOperationsDefault<Config>;

    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    CSRMatrixOperationsDefault()
        : d_localops_diag( std::make_shared<localops_t>() ),
          d_localops_offd( std::make_shared<localops_t>() )
    {
    }

    /** \brief  Matrix-vector multiplication
     * \param[in]  x  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] y  The resulting vector
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( std::shared_ptr<const Vector> x,
               MatrixData const &A,
               std::shared_ptr<Vector> y ) override;

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in   The vector to multiply
     * \param[in]  A    The input matrix A
     * \param[out] out  The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( std::shared_ptr<const Vector> in,
                        MatrixData const &A,
                        std::shared_ptr<Vector> out ) override;

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \param[in,out] A  The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Scale the matrix by a scalar and diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D      Vector representing the diagonal matrix
     * \param[in] A      The input matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha, std::shared_ptr<const Vector> D, MatrixData &A ) override;

    /** \brief  Scale the matrix by a scalar and inverse of diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D      Vector representing the diagonal matrix
     * \param[in] A      The input matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}^{-1}\mathbf{A}\f$
     */
    void scaleInv( AMP::Scalar alpha, std::shared_ptr<const Vector> D, MatrixData &A ) override;

    /** \brief  Compute the product of two matrices
     * \param[in]  A  Left multiplicand
     * \param[in]  B  Right multiplicand
     * \param[out] C  The product \f$\mathbf{AB}\f$.
     */
    void matMatMult( std::shared_ptr<MatrixData> A,
                     std::shared_ptr<MatrixData> B,
                     std::shared_ptr<MatrixData> C ) override;

    /** \brief  Compute the linear combination of two matrices
     * \param[in]  alpha  scalar
     * \param[in]  X      matrix
     * \param[out] Y      The output matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y ) override;

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     * \param[out] A      The input matrix A
     */
    void setScalar( AMP::Scalar alpha, MatrixData &A ) override;

    /** \brief  Set the non-zeros of the matrix to zero
     * \param[in] A The input matrix A
     * \details  Does not deallocate space.
     */
    void zero( MatrixData &A ) override;

    /** \brief  Set the diagonal to the values in a vector
     * \param[in]  in  The values to set the diagonal to
     * \param[out] A   The matrix to set
     */
    void setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A ) override;

    /** \brief Extract the diagonal values into a vector
     * \param[in]  A    The matrix to read from
     * \param[out] buf  Buffer to write diagonal into
     */
    void extractDiagonal( MatrixData const &A, std::shared_ptr<Vector> buf ) override;

    /** \brief Extract the row sums into a vector
     * \param[in]  A    The matrix to read from
     * \param[out] buf  Buffer to write row sums into
     */
    void getRowSums( MatrixData const &A, std::shared_ptr<Vector> buf ) override;

    /** \brief Extract the absolute row sums into a vector
     * \param[in]  A             The matrix to read from
     * \param[out] buf           Buffer to write row sums into
     * \param[in]  remove_zeros  If true zero values in sum are replaced with ones
     */
    void getRowSumsAbsolute( MatrixData const &A,
                             std::shared_ptr<Vector> buf,
                             const bool remove_zeros = false ) override;

    /** \brief  Set the matrix to the identity matrix
     * \param[out] A The matrix to set
     */
    void setIdentity( MatrixData &A ) override;

    /** \brief Compute the maximum row sum
     * \param[in] X Data for the input matrix
     * \return  The L-infinity norm of the matrix
     */
    AMP::Scalar LinfNorm( const MatrixData &X ) const override;

    /** \brief  Set Y matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     * \param[in] X  matrix data to copy from
     * \param[in] Y  matrix data to copy to
     */
    void copy( const MatrixData &X, MatrixData &Y ) override;

    /** \brief  Set Y matrix with the same non-zero and distributed structure
     * as x and copy the coefficients after up/down casting
     * \param[in] X  matrix data to copy from
     * \param[in] Y  matrix data to copy to after up/down casting the coefficients
     */
    void copyCast( const MatrixData &X, MatrixData &Y ) override;

    template<typename ConfigIn>
    static void
    copyCast( CSRMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>> *X,
              matrixdata_t *Y );

    std::string type() const override { return "CSRMatrixOperationsDefault"; }

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;

    CSRMatrixOperationsDefault( int64_t, AMP::IO::RestartManager * )
        : d_localops_diag( std::make_shared<localops_t>() ),
          d_localops_offd( std::make_shared<localops_t>() )
    {
    }

protected:
    std::shared_ptr<localops_t> d_localops_diag;
    std::shared_ptr<localops_t> d_localops_offd;
    std::map<std::pair<std::shared_ptr<matrixdata_t>, std::shared_ptr<matrixdata_t>>,
             CSRMatrixSpGEMMDefault<Config>>
        d_SpGEMMHelpers;
};

} // namespace AMP::LinearAlgebra

#endif
