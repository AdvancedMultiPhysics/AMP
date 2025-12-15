#ifndef included_CSRLocalMatrixOperationsDefault_H_
#define included_CSRLocalMatrixOperationsDefault_H_

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRLocalMatrixOperationsDefault
{
public:
    using config_type       = Config;
    using allocator_type    = typename Config::allocator_type;
    using localmatrixdata_t = CSRLocalMatrixData<Config>;
    using gidx_t            = typename Config::gidx_t;
    using lidx_t            = typename Config::lidx_t;
    using scalar_t          = typename Config::scalar_t;

    static_assert( std::is_same_v<typename allocator_type::value_type, void> );

    /** \brief  Matrix-vector multiplication
     * \param[in]  in   The vector to multiply
     * \param[in]  A    The input matrix A
     * \param[out] out  The resulting vector
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    static void mult( const scalar_t *in, std::shared_ptr<localmatrixdata_t> A, scalar_t *out );

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in     The vector to multiply
     * \param[in]  A      The input matrix A
     * \param[out] vvals  Values accumulated into from product
     * \param[out] rcols  Indices where vvals correspond to
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    static void multTranspose( const scalar_t *in,
                               std::shared_ptr<localmatrixdata_t> A,
                               std::vector<scalar_t> &vvals,
                               std::vector<size_t> &rcols );

    /** \brief  Scale the matrix by a scalar
     * \param[in]     alpha  The value to scale by
     * \param[in,out] A      The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    static void scale( scalar_t alpha, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and diagonal matrix
     * \param[in]     alpha  The value to scale by
     * \param[in]     D      Vector holding diagonal matrix entries
     * \param[in,out] A      The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}\mathbf{A}\f$
     */
    static void scale( scalar_t alpha, const scalar_t *D, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and inverse of diagonal matrix
     * \param[in]     alpha  The value to scale by
     * \param[in]     D      Vector holding diagonal matrix entries
     * \param[in,out] A      The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}^{-1}\mathbf{A}\f$
     */
    static void scaleInv( scalar_t alpha, const scalar_t *D, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Compute the linear combination of two matrices
     * \param[in]  alpha  scalar
     * \param[in]  X      Other matrix
     * \param[out] Y      The output matrix
     * \details  Compute \f$\mathbf{Y} = \alpha\mathbf{X} + \mathbf{Y}\f$
     */
    static void axpy( scalar_t alpha,
                      std::shared_ptr<localmatrixdata_t> X,
                      std::shared_ptr<localmatrixdata_t> Y );

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     * \param[out] A      The input matrix A
     */
    static void setScalar( scalar_t alpha, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set the non-zeros of the matrix to zero
     * \param[in] A The input matrix A
     * \details  May not deallocate space.
     */
    static void zero( std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set the diagonal to the values in a vector
     * \param[in]  in  The values to set the diagonal to
     * \param[out] A   The matrix to set
     */
    static void setDiagonal( const scalar_t *in, std::shared_ptr<localmatrixdata_t> A );

    /** \brief Extract the diagonal values into a vector
     * \param[in]  A    The matrix to read from
     * \param[out] buf  Buffer to write diagonal into
     */
    static void extractDiagonal( std::shared_ptr<localmatrixdata_t> A, scalar_t *buf );

    /** \brief Extract the row sums into a vector
     * \param[in]  A    The matrix to read from
     * \param[out] buf  Buffer to write row sums into
     */
    static void getRowSums( std::shared_ptr<localmatrixdata_t> A, scalar_t *buf );

    /** \brief Extract the absolute row sums into a vector
     * \param[in]  A    The matrix to read from
     * \param[out] buf  Buffer to write row sums into
     */
    static void getRowSumsAbsolute( std::shared_ptr<localmatrixdata_t> A, scalar_t *buf );

    /** \brief  Set the matrix to the identity matrix
     * \param[out] A  The matrix to set
     */
    static void setIdentity( std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     * \param[in] X  matrix data to copy from
     * \param[in] Y  matrix data to copy to
     */
    static void copy( std::shared_ptr<const localmatrixdata_t> X,
                      std::shared_ptr<localmatrixdata_t> Y );

    /** \brief  Set Y matrix with the same non-zero and distributed structure
     * as X and copy the coefficients after up/down casting
     * \param[in] X  matrix data to copy from
     * \param[in] Y  matrix data to copy to after up/down casting the coefficients
     */
    template<typename ConfigIn>
    static void
    copyCast( std::shared_ptr<
                  CSRLocalMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>>> X,
              std::shared_ptr<localmatrixdata_t> Y );
};

} // namespace AMP::LinearAlgebra

#endif
