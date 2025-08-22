#ifndef included_CSRLocalMatrixOperationsDevice_HPP_
#define included_CSRLocalMatrixOperationsDevice_HPP_

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/device/CSRLocalMatrixOperationsDevice.h"
#include "AMP/matrices/operations/device/DeviceMatrixOperations.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"

#include <type_traits>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::mult(
    const typename Config::scalar_t *in,
    std::shared_ptr<LocalMatrixData> A,
    typename Config::scalar_t *out )
{
    PROFILE( "CSRLocalMatrixOperationsDevice::mult" );
    AMP_DEBUG_ASSERT( in && out && A );

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    {
        PROFILE( "CSRLocalMatrixOperationsDevice::mult (local)" );
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::mult(
            row_starts_d, cols_loc_d, coeffs_d, nRows, in, out );
    }
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::multTranspose(
    const typename Config::scalar_t *,
    std::shared_ptr<LocalMatrixData>,
    std::vector<typename Config::scalar_t> &,
    std::vector<size_t> & )
{
    AMP_WARNING( "multTranspose not enabled for device." );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::scale(
    typename Config::scalar_t alpha, std::shared_ptr<LocalMatrixData> A )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    const auto tnnz_d = A->numberOfNonZeros();

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::scale( tnnz_d, coeffs_d, alpha );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::matMatMult(
    std::shared_ptr<LocalMatrixData>,
    std::shared_ptr<LocalMatrixData>,
    std::shared_ptr<LocalMatrixData> )
{
    AMP_WARNING( "matMatMult for CSRLocalMatrixOperationsDevice not implemented" );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::axpy(
    typename Config::scalar_t alpha,
    std::shared_ptr<LocalMatrixData> X,
    std::shared_ptr<LocalMatrixData> Y )
{
    const auto [row_starts_d_x, cols_d_x, cols_loc_d_x, coeffs_d_x] = X->getDataFields();
    auto [row_starts_d_y, cols_d_y, cols_loc_d_y, coeffs_d_y]       = Y->getDataFields();
    const auto tnnz                                                 = X->numberOfNonZeros();

    {
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::axpy(
            tnnz, alpha, coeffs_d_x, coeffs_d_y );
    }
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::setScalar(
    typename Config::scalar_t alpha, std::shared_ptr<LocalMatrixData> A )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    const auto tnnz_d = A->numberOfNonZeros();

    AMP::Utilities::Algorithms<typename Config::scalar_t>::fill_n( coeffs_d, tnnz_d, alpha );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::zero(
    std::shared_ptr<LocalMatrixData> A )
{
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::setDiagonal(
    const typename Config::scalar_t *in, std::shared_ptr<LocalMatrixData> A )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setDiagonal(
        row_starts_d, coeffs_d, nRows, in );
}


template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::extractDiagonal(
    std::shared_ptr<LocalMatrixData> A, typename Config::scalar_t *buf )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::extractDiagonal(
        row_starts_d, coeffs_d, nRows, buf );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::setIdentity(
    std::shared_ptr<LocalMatrixData> A )
{
    zero( A );

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setIdentity( row_starts_d, coeffs_d, nRows );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::LinfNorm(
    std::shared_ptr<LocalMatrixData> A, typename Config::scalar_t *rowSums )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::LinfNorm(
        nRows, coeffs_d, row_starts_d, rowSums );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::copy(
    std::shared_ptr<const LocalMatrixData> X, std::shared_ptr<LocalMatrixData> Y )
{
    const auto [row_starts_d_x, cols_d_x, cols_loc_d_x, coeffs_d_x] =
        std::const_pointer_cast<LocalMatrixData>( X )->getDataFields();
    auto [row_starts_d_y, cols_d_y, cols_loc_d_y, coeffs_d_y] = Y->getDataFields();
    const auto tnnz                                           = X->numberOfNonZeros();

    {
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::copy( tnnz, coeffs_d_x, coeffs_d_y );
    }
}

template<typename Config, class LocalMatrixData>
template<typename ConfigIn>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::copyCast(
    std::shared_ptr<CSRLocalMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>>>
        X,
    std::shared_ptr<LocalMatrixData> Y )
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

    using scalar_t_in  = typename ConfigIn::scalar_t;
    using scalar_t_out = typename Config::scalar_t;
    if constexpr ( std::is_same_v<scalar_t_in, scalar_t_out> ) {
        using gidx_t = typename Config::gidx_t;
        using lidx_t = typename Config::lidx_t;
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::copy(
            X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    } else {
        AMP::Utilities::
            copyCast<scalar_t_in, scalar_t_out, AMP::Utilities::Backend::Hip_Cuda, allocator_type>(
                X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    }
}

} // namespace AMP::LinearAlgebra

#endif
