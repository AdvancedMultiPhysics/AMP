#ifndef included_CSRLocalMatrixOperationsKokkos_H_
#define included_CSRLocalMatrixOperationsKokkos_H_

#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/utils/Memory.h"

#include <type_traits>
#include <variant>

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Config, class ExecSpace>
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

    // give names for explicit Kokkos memory spaces we use
    using kokkos_host_space_t = Kokkos::HostSpace;
    #ifdef AMP_USE_DEVICE
        #ifdef AMP_USE_CUDA
    using kokkos_managed_space_t = Kokkos::CudaUVMSpace;
    using kokkos_device_space_t  = Kokkos::CudaSpace;
        #else
    using kokkos_managed_space_t = Kokkos::HIPManagedSpace;
    using kokkos_device_space_t  = Kokkos::HIPSpace;
        #endif
    #endif

    // convert allocator_type from config into equivalent Kokkos memory space
    #ifndef AMP_USE_DEVICE
    // only one memory space, choice is easy
    using csr_memspace_t = kokkos_host_space_t;
    #else
    // pick between host, managed, and device
    using csr_memspace_t = typename std::conditional<
        alloc_info<Config::allocator>::mem_loc == AMP::Utilities::MemoryType::host,
        kokkos_host_space_t,
        typename std::conditional<alloc_info<Config::allocator>::mem_loc ==
                                      AMP::Utilities::MemoryType::managed,
                                  kokkos_managed_space_t,
                                  kokkos_device_space_t>::type>::type;
    #endif

    // tuples for wrapping CSR fields in views
    using csr_tuple_t =
        std::tuple<Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>,
                   Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>,
                   Kokkos::View<scalar_t *, Kokkos::LayoutRight, csr_memspace_t>>;
    using csr_const_tuple_t =
        std::tuple<Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>,
                   Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>,
                   Kokkos::View<const scalar_t *, Kokkos::LayoutRight, csr_memspace_t>>;

    CSRLocalMatrixOperationsKokkos( const ExecSpace &exec_space ) : d_exec_space( exec_space ) {}

    /** \brief  Matrix-vector multiplication
     * \param[in]  in The vector to multiply
     * \param[in]  in_loc Memory space of input vector
     * \param[in]  A The input matrix A
     * \param[out] out The resulting vector
     * \param[in]  out_loc Memory space of output vector
     * \details  Compute \f$\mathbf{Ax} = \mathbf{y}\f$.
     */
    void mult( const scalar_t *in,
               const AMP::Utilities::MemoryType in_loc,
               const scalar_t alpha,
               std::shared_ptr<localmatrixdata_t> A,
               const scalar_t beta,
               scalar_t *out,
               const AMP::Utilities::MemoryType out_loc );

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in The vector to multiply
     * \param[in]  in_loc Memory space of input vector
     * \param[in]  A The input matrix A
     * \param[out] out The resulting vector
     * \param[in]  out_loc Memory space of output vector
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( const scalar_t *in,
                        const AMP::Utilities::MemoryType in_loc,
                        std::shared_ptr<localmatrixdata_t> A,
                        scalar_t *out,
                        const AMP::Utilities::MemoryType out_loc );

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( scalar_t alpha, std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and diagonal matrix
     * \param[in]     alpha The value to scale by
     * \param[in]     D Vector holding diagonal matrix entries
     * \param[in]     D_loc Memory space of D
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( scalar_t alpha,
                const scalar_t *D,
                const AMP::Utilities::MemoryType D_loc,
                std::shared_ptr<localmatrixdata_t> A );

    /** \brief  Scale the matrix by a scalar and inverse of diagonal matrix
     * \param[in]     alpha The value to scale by
     * \param[in]     D Vector holding diagonal matrix entries
     * \param[in]     D_loc Memory space of D
     * \param[in,out] A The matrix A
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scaleInv( scalar_t alpha,
                   const scalar_t *D,
                   const AMP::Utilities::MemoryType D_loc,
                   std::shared_ptr<localmatrixdata_t> A );

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
     * \param[in]     D The values to set the diagonal to
     * \param[in]     D_loc Memory space of D
     * \param[in,out] A The matrix to set
     */
    void setDiagonal( const scalar_t *D,
                      const AMP::Utilities::MemoryType D_loc,
                      std::shared_ptr<localmatrixdata_t> A );

    /** \brief Extract the diagonal values into a vector
     * \param[in]  A The matrix to set
     * \param[out] D Buffer to write diagonal to
     * \param[in]  D_loc Memory space of buffer
     */
    void extractDiagonal( std::shared_ptr<localmatrixdata_t> A,
                          scalar_t *D,
                          const AMP::Utilities::MemoryType D_loc );

    /** \brief Extract the row sums into a vector
     * \param[in]  A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     * \param[in]  buf_loc Memory space of buffer
     */
    void getRowSums( std::shared_ptr<localmatrixdata_t> A,
                     scalar_t *buf,
                     const AMP::Utilities::MemoryType buf_loc,
                     const bool zero_first ) const;

    /** \brief Extract the absolute row sums into a vector
     * \param[in]  A The matrix to read from
     * \param[out] buf Buffer to write row sums into
     * \param[in]  buf_loc Memory space of buffer
     * \param[in]  zero_first Flag to zero out buffer before summing
     * \param[in]  remove_zeros Flag to replace zeros with ones after summing
     */
    void getRowSumsAbsolute( std::shared_ptr<localmatrixdata_t> A,
                             scalar_t *buf,
                             const AMP::Utilities::MemoryType buf_loc,
                             const bool zero_first,
                             const bool remove_zeros ) const;

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
    static void copyCast( std::shared_ptr<CSRLocalMatrixData<ConfigIn>> X,
                          std::shared_ptr<localmatrixdata_t> Y );

    //! Helper function for wrapping csr data into kokkos views
    static csr_const_tuple_t wrapCSRDataKokkos( std::shared_ptr<const localmatrixdata_t> A );

    //! Helper function for wrapping csr data into kokkos views
    static csr_tuple_t wrapCSRDataKokkos( std::shared_ptr<localmatrixdata_t> A );

    #ifndef AMP_USE_DEVICE
    //! Helper to wrap incoming data into view
    template<typename T, typename... ViewArgs>
    static auto WrapVector( T *ptr, lidx_t num, AMP::Utilities::MemoryType )
    {
        using ViewVariant =
            std::variant<Kokkos::View<T *, Kokkos::LayoutRight, kokkos_host_space_t, ViewArgs...>>;
        ViewVariant erased_view =
            Kokkos::View<T *, Kokkos::LayoutRight, kokkos_host_space_t, ViewArgs...>( ptr, num );
        return erased_view;
    }
    #else
    //! Helper to wrap incoming data into view matching its runtime known memory space
    template<typename T, typename... ViewArgs>
    static auto WrapVector( T *ptr, lidx_t num, AMP::Utilities::MemoryType mem_loc )
    {
        using ViewVariant = std::variant<
            Kokkos::View<T *, Kokkos::LayoutRight, kokkos_host_space_t, ViewArgs...>,
            Kokkos::View<T *, Kokkos::LayoutRight, kokkos_managed_space_t, ViewArgs...>,
            Kokkos::View<T *, Kokkos::LayoutRight, kokkos_device_space_t, ViewArgs...>>;
        ViewVariant erased_view;
        if ( mem_loc == AMP::Utilities::MemoryType::host ) {
            erased_view = Kokkos::View<T *, Kokkos::LayoutRight, kokkos_host_space_t, ViewArgs...>(
                ptr, num );
        } else if ( mem_loc == AMP::Utilities::MemoryType::managed ) {
            erased_view =
                Kokkos::View<T *, Kokkos::LayoutRight, kokkos_managed_space_t, ViewArgs...>( ptr,
                                                                                             num );
        } else if ( mem_loc == AMP::Utilities::MemoryType::device ) {
            erased_view =
                Kokkos::View<T *, Kokkos::LayoutRight, kokkos_device_space_t, ViewArgs...>( ptr,
                                                                                            num );
        } else {
            AMP_ERROR( "Unrecognized memory space requested" );
        }
        return erased_view;
    }
    #endif
protected:
    ExecSpace d_exec_space;
};

} // namespace AMP::LinearAlgebra

#endif

#endif
