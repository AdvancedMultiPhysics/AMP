#ifndef included_CSRLocalMatrixOperationsKokkos_H_
#define included_CSRLocalMatrixOperationsKokkos_H_

#include "AMP/matrices/operations/MatrixOperations.h"

#include <type_traits>

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Config, class ExecSpace, class ViewSpace>
class CSRLocalMatrixOperationsKokkos
{
public:
    static_assert( std::is_same_v<typename Config::allocator_type::value_type, void> );

    using config_type       = Config;
    using allocator_type    = typename Config::allocator_type;
    using localmatrixdata_t = CSRLocalMatrixData<Config>;

    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    CSRLocalMatrixOperationsKokkos( const ExecSpace &exec_space ) : d_exec_space( exec_space ) {}

    /** \brief  Matrix-vector multiplication
     * \param[in]  x  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] y  The resulting vectory
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( const scalar_t *in,
               const scalar_t alpha,
               std::shared_ptr<localmatrixdata_t> A,
               const scalar_t beta,
               scalar_t *out );

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[in]  A  The input matrix A
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( const scalar_t *in, std::shared_ptr<localmatrixdata_t> A, scalar_t *out );

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( scalar_t alpha, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D  Vector holding diagonal matrix entries
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( scalar_t alpha, const scalar_t *D, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and inverse of diagonal matrix
     * \param[in] alpha  The value to scale by
     * \param[in] D  Vector holding diagonal matrix entries
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scaleInv( scalar_t alpha, const scalar_t *D, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Compute the product of two matrices
     * \param[in] A  A multiplicand
     * \param[in] B  A multiplicand
     * \param[in] C  The product \f$\mathbf{AB}\f$.
     */
    void matMatMult( std::shared_ptr<localmatrixdata_t>,
                     std::shared_ptr<localmatrixdata_t>,
                     std::shared_ptr<localmatrixdata_t> );

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \param[out] Y The output matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( scalar_t alpha,
               std::shared_ptr<localmatrixdata_t> X,
               std::shared_ptr<localmatrixdata_t> Y );

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     * \param[out] A The input matrix A
     */
    void setScalar( scalar_t alpha, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set the non-zeros of the matrix to zero
     * \details  May not deallocate space.
     * \param[in] A The input matrix A
     */
    void zero( std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set the diagonal to the values in a vector
     * \param[in] in The values to set the diagonal to
     * \param[out] A The matrix to set
     */
    void setDiagonal( const scalar_t *in, std::shared_ptr<localmatrixdata_t> A );

    /** \brief Extract the diagonal values into a vector
     * \param[in] in The values to set the diagonal to
     * \param[in] A The matrix to set
     */
    void extractDiagonal( std::shared_ptr<localmatrixdata_t> A, scalar_t *buf );

    /** \brief Extract the row sums into a vector
     * \param[in] A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     */
    void
    getRowSums( std::shared_ptr<localmatrixdata_t> A, scalar_t *buf, const bool zero_first ) const;

    /** \brief Extract the absolute row sums into a vector
     * \param[in] A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     */
    void getRowSumsAbsolute( std::shared_ptr<localmatrixdata_t> A,
                             scalar_t *buf,
                             const bool zero_first ) const;

    /** \brief  Set the matrix to the identity matrix
     * \param[out] A The matrix to set
     */
    void setIdentity( std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     * \param[in] X matrix data to copy from
     * \param[in] Y matrix data to copy to
     */
    void copy( std::shared_ptr<const localmatrixdata_t> X, std::shared_ptr<localmatrixdata_t> Y );

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients after up/down casting
     * \param[in] X matrix data to copy from
     * \param[in] Y matrix data to copy to after up/down casting the coefficients
     */
    template<typename ConfigIn>
    static void
    copyCast( std::shared_ptr<
                  CSRLocalMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>>> X,
              std::shared_ptr<localmatrixdata_t> Y );

protected:
    ExecSpace d_exec_space;
};

} // namespace AMP::LinearAlgebra

#endif

#endif
