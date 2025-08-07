#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRLocalMatrixOperationsKokkos.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>

#include "ProfilerApp.h"

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Config, class ViewSpace>
auto wrapCSRDataKokkos( std::shared_ptr<const CSRLocalMatrixData<Config>> A )
{
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    const lidx_t nrows   = static_cast<lidx_t>( A->numLocalRows() );
    const lidx_t nnz_tot = A->numberOfNonZeros();
    auto [rowstarts, cols, cols_loc, coeffs] =
        std::const_pointer_cast<CSRLocalMatrixData<Config>>( A )->getDataFields();

    // coeffs not marked const so that setScalar and similar will work
    return std::make_tuple(
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, ViewSpace>( rowstarts, nrows + 1 ),
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, ViewSpace>( cols_loc, nnz_tot ),
        Kokkos::View<const scalar_t *, Kokkos::LayoutRight, ViewSpace>( coeffs, nnz_tot ) );
}

template<typename Config, class ViewSpace>
auto wrapCSRDataKokkos( std::shared_ptr<CSRLocalMatrixData<Config>> A )
{
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    const lidx_t nrows                       = static_cast<lidx_t>( A->numLocalRows() );
    const lidx_t nnz_tot                     = A->numberOfNonZeros();
    auto [rowstarts, cols, cols_loc, coeffs] = A->getDataFields();

    // coeffs not marked const so that setScalar and similar will work
    return std::make_tuple(
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, ViewSpace>( rowstarts, nrows + 1 ),
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, ViewSpace>( cols_loc, nnz_tot ),
        Kokkos::View<scalar_t *, Kokkos::LayoutRight, ViewSpace>( coeffs, nnz_tot ) );
}

namespace CSRMatOpsKokkosFunctor {

// This functor is based on the one inside KokkosKernels
// Modifications are made to handle our data structures,
// and more importantly our need to handle distributed matrices
template<typename Config,
         class ExecSpace,
         class RSView,
         class JAView,
         class AAView,
         class XView,
         class YView>
struct aAxpby {
    typedef typename Config::lidx_t lidx_t;
    typedef typename Config::scalar_t scalar_t;

    lidx_t num_rows;
    lidx_t num_rows_team;
    RSView rowstarts;
    JAView cols_loc;
    AAView coeffs;
    const scalar_t alpha;
    const scalar_t beta;
    XView inDataBlock;
    YView outDataBlock;

    aAxpby( lidx_t num_rows_,
            lidx_t num_rows_team_,
            RSView rowstarts_,
            JAView cols_loc_,
            AAView coeffs_,
            const scalar_t alpha_,
            const scalar_t beta_,
            XView inDataBlock_,
            YView outDataBlock_ )
        : num_rows( num_rows_ ),
          num_rows_team( num_rows_team_ ),
          rowstarts( rowstarts_ ),
          cols_loc( cols_loc_ ),
          coeffs( coeffs_ ),
          alpha( alpha_ ),
          beta( beta_ ),
          inDataBlock( inDataBlock_ ),
          outDataBlock( outDataBlock_ )
    {
    }

    // Calculate product for a single specific row
    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        if ( row >= num_rows ) {
            return;
        }
        scalar_t sum  = 0.0;
        const auto rs = rowstarts( row );
        const auto re = rowstarts( row + 1 );
        for ( lidx_t c = rs; c < re; ++c ) {
            const auto cl = cols_loc( c );
            sum += coeffs( c ) * inDataBlock( cl );
        }
        outDataBlock( row ) *= beta;
        outDataBlock( row ) += alpha * sum;
    }

    // process a block of rows hierarchically
    KOKKOS_INLINE_FUNCTION
    void operator()( const typename Kokkos::TeamPolicy<ExecSpace>::member_type &tm ) const
    {
        const auto lRank = tm.league_rank();
        const auto fRow  = lRank * num_rows_team;
        Kokkos::parallel_for( Kokkos::TeamThreadRange( tm, num_rows_team ),
                              [&]( const lidx_t tIdx ) {
                                  const auto row = fRow + tIdx;
                                  if ( row >= num_rows ) {
                                      return;
                                  }
                                  scalar_t sum  = 0.0;
                                  const auto rs = rowstarts( row );
                                  const auto re = rowstarts( row + 1 );
                                  Kokkos::parallel_reduce(
                                      Kokkos::ThreadVectorRange( tm, re - rs ),
                                      [&]( lidx_t &c, scalar_t &lsum ) {
                                          const auto cl = cols_loc( rs + c );
                                          lsum += coeffs( rs + c ) * inDataBlock( cl );
                                      },
                                      sum );
                                  outDataBlock( row ) *= beta;
                                  outDataBlock( row ) += alpha * sum;
                              } );
    }
};

template<typename Policy,
         class ExecSpace,
         class RSView,
         class JAView,
         class AAView,
         class XView,
         class YView>
struct MultTranspose {
    typedef typename Policy::lidx_t lidx_t;
    typedef typename Policy::scalar_t scalar_t;

    lidx_t num_rows;
    lidx_t num_rows_team;
    RSView rowstarts;
    JAView cols_loc;
    AAView coeffs;
    XView inDataBlock;
    YView outDataBlock;

    MultTranspose( lidx_t num_rows_,
                   lidx_t num_rows_team_,
                   RSView rowstarts_,
                   JAView cols_loc_,
                   AAView coeffs_,
                   XView inDataBlock_,
                   YView outDataBlock_ )
        : num_rows( num_rows_ ),
          num_rows_team( num_rows_team_ ),
          rowstarts( rowstarts_ ),
          cols_loc( cols_loc_ ),
          coeffs( coeffs_ ),
          inDataBlock( inDataBlock_ ),
          outDataBlock( outDataBlock_ )
    {
        // TODO: Add assertion that YView has atomic memory trait
    }

    // Calculate product for a single specific row
    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        if ( row >= num_rows ) {
            return;
        }
        const auto rs = rowstarts( row );
        const auto re = rowstarts( row + 1 );
        const auto xi = inDataBlock( row );
        for ( lidx_t c = rs; c < re; ++c ) {
            const auto cl = cols_loc( c );
            outDataBlock( cl ) += xi * coeffs( c );
        }
    }

    // process a block of rows hierarchically
    KOKKOS_INLINE_FUNCTION
    void operator()( const typename Kokkos::TeamPolicy<ExecSpace>::member_type &tm ) const
    {
        const auto lRank = tm.league_rank();
        const auto fRow  = lRank * num_rows_team;
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange( tm, num_rows_team ), [&]( const lidx_t tIdx ) {
                const auto row = fRow + tIdx;
                if ( row >= num_rows ) {
                    return;
                }
                const auto rs = rowstarts( row );
                const auto re = rowstarts( row + 1 );
                const auto xi = inDataBlock( row );
                Kokkos::parallel_for( Kokkos::ThreadVectorRange( tm, re - rs ), [&]( lidx_t &c ) {
                    const auto cl = cols_loc( rs + c );
                    outDataBlock( cl ) += xi * coeffs( rs + c );
                } );
            } );
    }
};

} // namespace CSRMatOpsKokkosFunctor

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::mult(
    const typename Config::scalar_t *in,
    const typename Config::scalar_t alpha,
    std::shared_ptr<localmatrixdata_t> A,
    const typename Config::scalar_t beta,
    typename Config::scalar_t *out )
{
    const auto nRows = A->numLocalRows();
    const auto nCols = A->numUniqueColumns();

    // Wrap in/out data into Kokkos Views
    Kokkos::View<const scalar_t *,
                 Kokkos::LayoutRight,
                 ViewSpace,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        inView( in, nCols );
    Kokkos::View<scalar_t *, Kokkos::LayoutRight, ViewSpace> outView( out, nRows );

    const auto [rowstarts, cols_loc, coeffs] = wrapCSRDataKokkos<Config, ViewSpace>( A );

    // rows per team and vector length influenced by KokkosKernels
    // should tune to architecture (AMD vs. NVidia) and "typical" problems
    const lidx_t team_rows     = 64;
    const lidx_t vector_length = 8;
    const lidx_t num_teams     = ( nRows + team_rows - 1 ) / team_rows;

    CSRMatOpsKokkosFunctor::aAxpby<Config,
                                   ExecSpace,
                                   decltype( rowstarts ),
                                   decltype( cols_loc ),
                                   decltype( coeffs ),
                                   decltype( inView ),
                                   decltype( outView )>
        ftor( nRows, team_rows, rowstarts, cols_loc, coeffs, alpha, beta, inView, outView );

    if constexpr ( std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace> ) {
        Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>> team_policy(
            d_exec_space, num_teams, Kokkos::AUTO, vector_length );
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::mult (local - team)", team_policy, ftor );
    } else {
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::mult (local - flat)",
                              Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
                              ftor );
    }
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::multTranspose(
    const typename Config::scalar_t *in,
    std::shared_ptr<localmatrixdata_t> A,
    typename Config::scalar_t *out )
{
    const auto [rowstarts, cols_loc, coeffs] = wrapCSRDataKokkos<Config, ViewSpace>( A );

    const auto nRows    = A->numLocalRows();
    const auto nCols    = A->numLocalColumns();
    const auto nColsUnq = A->numUniqueColumns();

    // Wrap in/out data into Kokkos Views
    Kokkos::View<const scalar_t *,
                 Kokkos::LayoutRight,
                 ViewSpace,
                 Kokkos::MemoryTraits<Kokkos::RandomAccess>>
        inView( in, nCols );

    // vvals only on host, use temporary and copy as needed after
    Kokkos::View<scalar_t *, Kokkos::LayoutRight, ViewSpace, Kokkos::MemoryTraits<Kokkos::Atomic>>
        outView( out, nColsUnq );

    // rows per team and vector length influenced by KokkosKernels
    // should tune to architecture (AMD vs. NVidia) and "typical" problems
    const lidx_t team_rows     = 64;
    const lidx_t vector_length = 8;
    const lidx_t num_teams     = ( nRows + team_rows - 1 ) / team_rows;

    CSRMatOpsKokkosFunctor::MultTranspose<Config,
                                          ExecSpace,
                                          decltype( rowstarts ),
                                          decltype( cols_loc ),
                                          decltype( coeffs ),
                                          decltype( inView ),
                                          decltype( outView )>
        ftor( nRows, team_rows, rowstarts, cols_loc, coeffs, inView, outView );

    if constexpr ( std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace> && false ) {
        Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>> team_policy(
            d_exec_space, num_teams, Kokkos::AUTO, vector_length );
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::multTranspose (local - team)", team_policy, ftor );
    } else {
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::multTranspose (local - flat)",
                              Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
                              ftor );
    }
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::scale(
    typename Config::scalar_t alpha, std::shared_ptr<localmatrixdata_t> A )
{
    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto coeffs     = std::get<2>( vTpl );

    const auto tnnz = A->numberOfNonZeros();

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::scale",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz ),
        KOKKOS_LAMBDA( lidx_t n ) { coeffs( n ) *= alpha; } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::matMatMult(
    std::shared_ptr<localmatrixdata_t>,
    std::shared_ptr<localmatrixdata_t>,
    std::shared_ptr<localmatrixdata_t> )
{
    AMP_WARNING( "matMatMult for CSRLocalMatrixOperationsKokkos not implemented" );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::axpy(
    typename Config::scalar_t alpha,
    std::shared_ptr<localmatrixdata_t> X,
    std::shared_ptr<localmatrixdata_t> Y )
{
    const auto vTplX = wrapCSRDataKokkos<Config, ViewSpace>( X );
    auto coeffsX     = std::get<2>( vTplX );

    const auto vTplY = wrapCSRDataKokkos<Config, ViewSpace>( Y );
    auto coeffsY     = std::get<2>( vTplY );

    const auto tnnz = X->numberOfNonZeros();

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::axpy",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, tnnz ),
        KOKKOS_LAMBDA( gidx_t n ) { coeffsY( n ) += alpha * coeffsX( n ); } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::setScalar(
    typename Config::scalar_t alpha, std::shared_ptr<localmatrixdata_t> A )
{
    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto coeffs     = std::get<2>( vTpl );

    Kokkos::deep_copy( d_exec_space, coeffs, alpha );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::zero(
    std::shared_ptr<localmatrixdata_t> A )
{
    setScalar( 0.0, A );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::setDiagonal(
    const typename Config::scalar_t *in, std::shared_ptr<localmatrixdata_t> A )
{
    if ( !A->isDiag() ) {
        AMP_WARNING( "Attempted to call CSRLocalMatrixOperationsKokkos::setDiagonal on "
                     "off-diagonal block. Ignoring." );
        return;
    }

    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    Kokkos::View<const scalar_t *, Kokkos::LayoutRight> vvals( in, nRows );

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::setDiagonal",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
        KOKKOS_LAMBDA( lidx_t row ) { coeffs( rowstarts( row ) ) = vvals( row ); } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::setIdentity(
    std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    if ( !A->isDiag() ) {
        return;
    }

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::setIdentity",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
        KOKKOS_LAMBDA( lidx_t row ) { coeffs( rowstarts( row ) ) = 1.0; } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::extractDiagonal(
    std::shared_ptr<localmatrixdata_t> A, typename Config::scalar_t *buf )
{
    if ( !A->isDiag() ) {
        AMP_WARNING( "Attempted to call CSRLocalMatrixOperationsKokkos::extractDiagonal on "
                     "off-diagonal block. Ignoring." );
        return;
    }

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    Kokkos::View<scalar_t *, Kokkos::LayoutRight> vvals( buf, nRows );

    std::cout << "in extractDiag, have: " << nRows << ", " << rowstarts.extent( 0 ) << ", "
              << coeffs.extent( 0 ) << ", " << vvals.extent( 0 ) << std::endl;

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::extractDiagonal",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
        KOKKOS_LAMBDA( lidx_t row ) { vvals( row ) = coeffs( rowstarts( row ) ); } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::LinfNorm(
    std::shared_ptr<localmatrixdata_t> A, typename Config::scalar_t *rowSums ) const
{
    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos<Config, ViewSpace>( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    Kokkos::View<scalar_t *, Kokkos::LayoutRight> sums( rowSums, nRows );

    Kokkos::parallel_for(
        "CSRMatrixOperationsKokkos::LinfNorm",
        Kokkos::RangePolicy<ExecSpace>( d_exec_space, 0, nRows ),
        KOKKOS_LAMBDA( lidx_t row ) {
            for ( lidx_t c = rowstarts( row ); c < rowstarts( row + 1 ); ++c ) {
                sums( row ) += Kokkos::fabs( coeffs( c ) );
            }
        } );
}

template<typename Config, class ExecSpace, class ViewSpace>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::copy(
    std::shared_ptr<const localmatrixdata_t> X, std::shared_ptr<localmatrixdata_t> Y )
{
    const auto vTplX = wrapCSRDataKokkos<Config, ViewSpace>( X );
    auto coeffsX     = std::get<2>( vTplX );

    const auto vTplY = wrapCSRDataKokkos<Config, ViewSpace>( Y );
    auto coeffsY     = std::get<2>( vTplY );

    Kokkos::deep_copy( coeffsY, coeffsX );
}

template<typename Config, class ExecSpace, class ViewSpace>
template<typename ConfigIn>
void CSRLocalMatrixOperationsKokkos<Config, ExecSpace, ViewSpace>::copyCast(
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
        const auto X_v = Kokkos::View<scalar_t_in *, Kokkos::LayoutRight, ViewSpace>(
            X_coeffs, X->numberOfNonZeros() );
        auto Y_v = Kokkos::View<scalar_t_out *, Kokkos::LayoutRight, ViewSpace>(
            Y_coeffs, Y->numberOfNonZeros() );

        Kokkos::deep_copy( Y_v, X_v );
    } else {
        AMP::Utilities::
            copyCast<scalar_t_in, scalar_t_out, AMP::Utilities::Backend::Kokkos, allocator_type>(
                X->numberOfNonZeros(), X_coeffs, Y_coeffs );
    }
}

} // namespace AMP::LinearAlgebra

#endif
