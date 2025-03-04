#ifndef included_CSRLocalMatrixOperationsDevice_HPP_
#define included_CSRLocalMatrixOperationsDevice_HPP_

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/device/CSRLocalMatrixOperationsDevice.h"
#include "AMP/matrices/operations/device/DeviceMatrixOperations.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::mult(
    const typename Policy::scalar_t *in,
    std::shared_ptr<LocalMatrixData> A,
    typename Policy::scalar_t *out )
{
    PROFILE( "CSRLocalMatrixOperationsDevice::mult" );
    AMP_DEBUG_ASSERT( in && out && A );

    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    {
        PROFILE( "CSRLocalMatrixOperationsDevice::mult (local)" );
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::mult(
            row_starts_d, cols_loc_d, coeffs_d, nRows, in, out );
    }
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::multTranspose(
    const typename Policy::scalar_t *,
    std::shared_ptr<LocalMatrixData>,
    std::vector<typename Policy::scalar_t> &,
    std::vector<size_t> & )
{
    AMP_WARNING( "multTranspose not enabled for device." );
}


template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::scale(
    typename Policy::scalar_t alpha, std::shared_ptr<LocalMatrixData> A )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    const auto tnnz_d = A->numberOfNonZeros();

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::scale( tnnz_d, coeffs_d, alpha );
}
template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::matMultiply(
    std::shared_ptr<LocalMatrixData>,
    std::shared_ptr<LocalMatrixData>,
    std::shared_ptr<LocalMatrixData> )
{
    AMP_WARNING( "SpGEMM for CSRLocalMatrixOperationsDevice not implemented" );
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::axpy(
    typename Policy::scalar_t alpha,
    std::shared_ptr<LocalMatrixData> X,
    std::shared_ptr<LocalMatrixData> Y )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    const auto [row_starts_d_x, cols_d_x, cols_loc_d_x, coeffs_d_x] = X->getDataFields();
    auto [row_starts_d_y, cols_d_y, cols_loc_d_y, coeffs_d_y]       = Y->getDataFields();
    const auto tnnz                                                 = X->numberOfNonZeros();

    {
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::axpy(
            tnnz, alpha, coeffs_d_x, coeffs_d_y );
    }
}

template<typename Policy, class Allocator, class LocalMatrixData>
template<typename PolicyIn>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::copyCast(
    std::shared_ptr<CSRLocalMatrixData<PolicyIn, Allocator>> X, std::shared_ptr<LocalMatrixData> Y )
{
    // Check compatibility
    AMP_ASSERT( Y->getMemoryLocation() == X->getMemoryLocation() );
    AMP_ASSERT( Y->beginRow() == X->beginRow() );
    AMP_ASSERT( Y->endRow() == X->endRow() );
    AMP_ASSERT( Y->beginCol() == X->beginCol() );
    AMP_ASSERT( Y->endCol() == X->endCol() );

    AMP_ASSERT( Y->numberOfNonZeros() == X->numberOfNonZeros() );

    AMP_ASSERT( Y->numLocalRows() == X->numLocalRows() );
    AMP_ASSERT( Y->numUniqueColumns() == X->numUniqueColumns() );

    // ToDO: d_pParameters = x->d_pParameters;

    // Shallow copy data structure
    auto [X_row_starts, X_cols, X_cols_loc, X_coeffs] = X->getDataFields();
    auto [Y_row_starts, Y_cols, Y_cols_loc, Y_coeffs] = Y->getDataFields();

    // Copy column map only if off diag block
    if ( !X->isDiag() ) {
        auto X_col_map = X->getColumnMap();
        auto Y_col_map = Y->getColumnMap();
        Y_col_map      = X_col_map;
        AMP_ASSERT( Y_col_map );
    }

    Y_row_starts = X_row_starts;
    Y_cols       = X_cols;
    Y_cols_loc   = X_cols_loc;

    using scalar_t_in  = typename PolicyIn::scalar_t;
    using scalar_t_out = typename Policy::scalar_t;
    if constexpr ( std::is_same_v<scalar_t_in, scalar_t_out> ) {
        using gidx_t = typename Policy::gidx_t;
        using lidx_t = typename Policy::lidx_t;
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::copy(
            X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    } else {
        AMP::Utilities::copyCast<scalar_t_in, scalar_t_out>(
            X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    }
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::setScalar(
    typename Policy::scalar_t alpha, std::shared_ptr<LocalMatrixData> A )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    const auto tnnz_d = A->numberOfNonZeros();

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setScalar( tnnz_d, coeffs_d, alpha );
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::zero(
    std::shared_ptr<LocalMatrixData> A )
{
    using scalar_t = typename Policy::scalar_t;
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::setDiagonal(
    const typename Policy::scalar_t *in, std::shared_ptr<LocalMatrixData> A )
{
    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setDiagonal(
        row_starts_d, coeffs_d, nRows, in );
}


template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::extractDiagonal(
    std::shared_ptr<LocalMatrixData> A, typename Policy::scalar_t *buf )
{
    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::extractDiagonal(
        row_starts_d, coeffs_d, nRows, buf );
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::setIdentity(
    std::shared_ptr<LocalMatrixData> A )
{
    using lidx_t   = typename Policy::lidx_t;
    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    zero( A );

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setIdentity( row_starts_d, coeffs_d, nRows );
}

template<typename Policy, class Allocator, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Policy, Allocator, LocalMatrixData>::LinfNorm(
    std::shared_ptr<LocalMatrixData> A, typename Policy::scalar_t *rowSums )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::LinfNorm(
        nRows, coeffs_d, row_starts_d, rowSums );
}

} // namespace AMP::LinearAlgebra

#endif
