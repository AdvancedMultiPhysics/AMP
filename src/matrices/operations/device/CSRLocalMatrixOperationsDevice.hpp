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
            row_starts_d,
            cols_loc_d,
            coeffs_d,
            nRows,
            in,
            out,
            A->d_acceleration_context.getStream() );
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

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::scale(
        tnnz_d, coeffs_d, alpha, A->d_acceleration_context.getStream() );
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
            tnnz, alpha, coeffs_d_x, coeffs_d_y, Y->d_acceleration_context.getStream() );
    }
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::setScalar(
    typename Config::scalar_t alpha, std::shared_ptr<LocalMatrixData> A )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();

    const auto tnnz_d = A->numberOfNonZeros();

    AMP::Utilities::Algorithms::fill_n(
        coeffs_d, tnnz_d, alpha, Config::mem_loc, A->d_acceleration_context.getStream() );
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
        row_starts_d, coeffs_d, nRows, in, A->d_acceleration_context.getStream() );
}


template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::extractDiagonal(
    std::shared_ptr<LocalMatrixData> A, typename Config::scalar_t *buf )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::extractDiagonal(
        row_starts_d, coeffs_d, nRows, buf, A->d_acceleration_context.getStream() );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::setIdentity(
    std::shared_ptr<LocalMatrixData> A )
{
    zero( A );

    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::setIdentity(
        row_starts_d, coeffs_d, nRows, A->d_acceleration_context.getStream() );
}

template<typename Config, class LocalMatrixData>
void CSRLocalMatrixOperationsDevice<Config, LocalMatrixData>::LinfNorm(
    std::shared_ptr<LocalMatrixData> A, typename Config::scalar_t *rowSums )
{
    auto [row_starts_d, cols_d, cols_loc_d, coeffs_d] = A->getDataFields();
    const auto nRows                                  = static_cast<lidx_t>( A->numLocalRows() );

    DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::LinfNorm(
        nRows, coeffs_d, row_starts_d, rowSums, A->d_acceleration_context.getStream() );
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
        DeviceMatrixOperations<gidx_t, lidx_t, scalar_t>::copy(
            tnnz, coeffs_d_x, coeffs_d_y, Y->d_acceleration_context.getStream() );
    }
}

} // namespace AMP::LinearAlgebra

#endif
