#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/default/CSRLocalMatrixOperationsDefault.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>
#include <memory>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::mult( const typename Config::scalar_t *in,
                                                    std::shared_ptr<localmatrixdata_t> A,
                                                    typename Config::scalar_t *out )
{
    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );
    lidx_t *rs, *cols_loc;
    gidx_t *cols;
    scalar_t *coeffs;
    std::tie( rs, cols, cols_loc, coeffs ) = A->getDataFields();

    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            // Note: output is assumed to have useful values already
            out[row] += coeffs[c] * in[cols_loc[c]];
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::multTranspose(
    const typename Config::scalar_t *in,
    std::shared_ptr<localmatrixdata_t> A,
    std::vector<typename Config::scalar_t> &vvals,
    std::vector<size_t> &rcols )
{
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    A->getColumnMap( rcols );
    vvals.resize( rcols.size(), 0.0 );

    for ( lidx_t row = 0; row < nRows; ++row ) {
        const auto val = in[row];
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            vvals[cols_loc[c]] += coeffs[c] * val;
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::scale( typename Config::scalar_t alpha,
                                                     std::shared_ptr<localmatrixdata_t> A )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();

    const auto tnnz = A->numberOfNonZeros();

    std::transform( const_cast<scalar_t *>( coeffs ),
                    const_cast<scalar_t *>( coeffs ) + tnnz,
                    const_cast<scalar_t *>( coeffs ),
                    [=]( scalar_t val ) { return alpha * val; } );
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::scale( typename Config::scalar_t alpha,
                                                     const typename Config::scalar_t *D,
                                                     std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();

    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            coeffs[c] *= alpha * D[row];
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::scaleInv( typename Config::scalar_t alpha,
                                                        const typename Config::scalar_t *D,
                                                        std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();

    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            coeffs[c] *= alpha / D[row];
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::axpy( typename Config::scalar_t alpha,
                                                    std::shared_ptr<localmatrixdata_t> X,
                                                    std::shared_ptr<localmatrixdata_t> Y )
{
    const auto [rs_x, cols_x, cols_loc_x, coeffs_x] = X->getDataFields();
    auto [rs_y, cols_y, cols_loc_y, coeffs_y]       = Y->getDataFields();

    const auto nrows = static_cast<lidx_t>( X->numLocalRows() );

    for ( lidx_t row = 0; row < nrows; ++row ) {
        for ( lidx_t iy = rs_y[row]; iy < rs_y[row + 1]; ++iy ) {
            const auto yc = cols_loc_y[iy];
            for ( lidx_t ix = rs_x[row]; ix < rs_x[row + 1]; ++ix ) {
                if ( yc == cols_loc_x[ix] ) {
                    coeffs_y[iy] += alpha * coeffs_x[ix];
                    break;
                }
            }
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::setScalar( typename Config::scalar_t alpha,
                                                         std::shared_ptr<localmatrixdata_t> A )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    const auto tnnz                   = A->numberOfNonZeros();
    std::fill( coeffs, coeffs + tnnz, alpha );
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::zero( std::shared_ptr<localmatrixdata_t> A )
{
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::setDiagonal( const typename Config::scalar_t *in,
                                                           std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();

    for ( lidx_t row = 0; row < nRows; ++row ) {
        coeffs[rs[row]] = in[row];
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::setIdentity( std::shared_ptr<localmatrixdata_t> A )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );

    for ( lidx_t row = 0; row < nRows; ++row ) {
        coeffs[rs[row]] = 1.0;
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::extractDiagonal( std::shared_ptr<localmatrixdata_t> A,
                                                               typename Config::scalar_t *buf )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );

    for ( lidx_t row = 0; row < nRows; ++row ) {
        buf[row] = coeffs[rs[row]];
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::getRowSums( std::shared_ptr<localmatrixdata_t> A,
                                                          typename Config::scalar_t *buf )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );

    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            buf[row] += coeffs[c];
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::getRowSumsAbsolute(
    std::shared_ptr<localmatrixdata_t> A, typename Config::scalar_t *buf )
{
    auto [rs, cols, cols_loc, coeffs] = A->getDataFields();
    const auto nRows                  = static_cast<lidx_t>( A->numLocalRows() );

    for ( lidx_t row = 0; row < nRows; ++row ) {
        for ( lidx_t c = rs[row]; c < rs[row + 1]; ++c ) {
            buf[row] += std::abs( coeffs[c] );
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsDefault<Config>::copy( std::shared_ptr<const localmatrixdata_t> X,
                                                    std::shared_ptr<localmatrixdata_t> Y )
{
    AMP_DEBUG_ASSERT( Y->numberOfNonZeros() == X->numberOfNonZeros() );
    const auto tnnz = X->numberOfNonZeros();

    // Shallow copy data structure
    const auto [X_row_starts, X_cols, X_cols_loc, X_coeffs] =
        std::const_pointer_cast<localmatrixdata_t>( X )->getDataFields();
    auto [Y_row_starts, Y_cols, Y_cols_loc, Y_coeffs] = Y->getDataFields();
    std::copy( X_coeffs, X_coeffs + tnnz, Y_coeffs );
}


template<typename Config>
template<typename ConfigIn>
void CSRLocalMatrixOperationsDefault<Config>::copyCast(
    std::shared_ptr<CSRLocalMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>>>
        X,
    std::shared_ptr<localmatrixdata_t> Y )
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
        std::copy( X_coeffs, X_coeffs + X->numberOfNonZeros(), Y_coeffs );
    } else {
#ifdef AMP_USE_OPENMP
        constexpr auto DefaultBcknd = AMP::Utilities::Backend::OpenMP;
#else
        constexpr auto DefaultBcknd = AMP::Utilities::Backend::Serial;
#endif
        AMP::Utilities::copyCast<scalar_t_in, scalar_t_out, DefaultBcknd, allocator_type>(
            X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    }
}

} // namespace AMP::LinearAlgebra
