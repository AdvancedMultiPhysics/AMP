#ifndef included_CSRMatrixOperationsKokkos_H_
#define included_CSRMatrixOperationsKokkos_H_

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/matrices/operations/kokkos/CSRLocalMatrixOperationsKokkos.h"
#include "AMP/vectors/Vector.h"

#include <type_traits>

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Config,
         class ExecSpace = typename std::conditional<
             std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>>,
             Kokkos::DefaultHostExecutionSpace,
             Kokkos::DefaultExecutionSpace>::type,
         class ViewSpace = typename std::conditional<
             std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>>,
             Kokkos::HostSpace,
             Kokkos::SharedSpace>::type>
class CSRMatrixOperationsKokkos : public MatrixOperations
{
public:
    static_assert( std::is_same_v<typename Config::allocator_type::value_type, void> );

    using config_type       = Config;
    using allocator_type    = typename Config::allocator_type;
    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;

    using localops_t = CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>;

    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    CSRMatrixOperationsKokkos()
        : d_exec_space(),
          d_localops_diag( std::make_shared<localops_t>( d_exec_space ) ),
          d_localops_offd( std::make_shared<localops_t>( d_exec_space ) )
    {
    }

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

    /** \brief  Scale the matrix by a scalar and diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D  A vector representing the diagonal matrix
     * \param[in] A The input matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}\mathbf{A}\f$
     */
    void scale( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & ) override
    {
        AMP_ERROR( "Not implemented" );
    }

    /** \brief  Scale the matrix by a scalar and inverse of diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D  A vector representing the diagonal matrix
     * \param[in] A The input matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}^{-1}\mathbf{A}\f$
     */
    void scaleInv( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & ) override
    {
        AMP_ERROR( "Not implemented" );
    }

    /** \brief  Compute the product of two matrices
     * \param[in] A  A multiplicand
     * \param[in] B  A multiplicand
     * \return The product \f$\mathbf{AB}\f$.
     */
    void matMatMult( std::shared_ptr<MatrixData> A,
                     std::shared_ptr<MatrixData> B,
                     std::shared_ptr<MatrixData> C ) override;

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

    /** \brief Extract the diagonal values into a vector
     * \param[in] A The matrix to read from
     * \param[out] buf Buffer to write diagonal into
     */
    void extractDiagonal( MatrixData const &A, std::shared_ptr<Vector> buf ) override;

    /** \brief Extract the row sums into a vector
     * \param[in] A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     */
    void getRowSums( MatrixData const &, std::shared_ptr<Vector> ) override
    {
        AMP_ERROR( "Not implemented" );
    }

    /** \brief Extract the absolute row sums into a vector
     * \param[in] A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     */
    void getRowSumsAbsolute( MatrixData const &, std::shared_ptr<Vector>, const bool ) override
    {
        AMP_ERROR( "Not implemented" );
    }

    /** \brief Compute the maximum column sum
     * \return  The L1 norm of the matrix
     */
    AMP::Scalar LinfNorm( const MatrixData &X ) const override;

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     * \param[in] x matrix data to copy from
     * \param[in] y matrix data to copy to
     */
    void copy( const MatrixData &X, MatrixData &Y ) override;

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients after up/down casting
     * \param[in] x matrix data to copy from
     * \param[in] y matrix data to copy to after up/down casting the coefficients
     */
    void copyCast( const MatrixData &X, MatrixData &Y ) override;

    template<typename ConfigIn>
    static void
    copyCast( CSRMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>> *X,
              CSRMatrixData<Config> *Y );

protected:
    ExecSpace d_exec_space;
    std::shared_ptr<localops_t> d_localops_diag;
    std::shared_ptr<localops_t> d_localops_offd;
};

} // namespace AMP::LinearAlgebra

#endif
#endif
