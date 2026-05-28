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

template<typename Config>
typename CSRLocalMatrixOperationsKokkos<Config>::csr_const_tuple_t
CSRLocalMatrixOperationsKokkos<Config>::wrapCSRDataKokkos(
    std::shared_ptr<const CSRLocalMatrixData<Config>> A )
{
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    const lidx_t nrows   = static_cast<lidx_t>( A->numLocalRows() );
    const lidx_t nnz_tot = A->numberOfNonZeros();
    auto [rowstarts, cols, cols_loc, coeffs] =
        std::const_pointer_cast<CSRLocalMatrixData<Config>>( A )->getDataFields();

    // coeffs not marked const so that setScalar and similar will work
    return std::make_tuple(
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>( rowstarts, nrows + 1 ),
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>( cols_loc, nnz_tot ),
        Kokkos::View<const scalar_t *, Kokkos::LayoutRight, csr_memspace_t>( coeffs, nnz_tot ) );
}

template<typename Config>
typename CSRLocalMatrixOperationsKokkos<Config>::csr_tuple_t
CSRLocalMatrixOperationsKokkos<Config>::wrapCSRDataKokkos(
    std::shared_ptr<CSRLocalMatrixData<Config>> A )
{
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    const lidx_t nrows                       = static_cast<lidx_t>( A->numLocalRows() );
    const lidx_t nnz_tot                     = A->numberOfNonZeros();
    auto [rowstarts, cols, cols_loc, coeffs] = A->getDataFields();

    // coeffs not marked const so that setScalar and similar will work
    return std::make_tuple(
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>( rowstarts, nrows + 1 ),
        Kokkos::View<const lidx_t *, Kokkos::LayoutRight, csr_memspace_t>( cols_loc, nnz_tot ),
        Kokkos::View<scalar_t *, Kokkos::LayoutRight, csr_memspace_t>( coeffs, nnz_tot ) );
}

namespace CSRMatOpsKokkosFunctor {

// This functor is based on the one inside KokkosKernels
// Modifications are made to handle our data structures,
// and more importantly our need to handle distributed matrices
template<class ExecSpace,
         typename lidx_t,
         typename scalar_t,
         class RSView,
         class JAView,
         class AAView,
         class XView,
         class YView>
struct aAxpby {
    const lidx_t num_rows;
    const lidx_t num_rows_team;
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

template<class ExecSpace,
         typename lidx_t,
         class RSView,
         class JAView,
         class AAView,
         class XView,
         class YView>
struct MultTranspose {
    const lidx_t num_rows;
    const lidx_t num_rows_team;
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

template<typename Config, class RSView, class AAView, class DView>
struct Scale {
    typedef typename Config::lidx_t lidx_t;
    typedef typename Config::scalar_t scalar_t;

    RSView rowstarts;
    AAView coeffs;
    DView diag;
    const scalar_t alpha;

    Scale( RSView rowstarts_, AAView coeffs_, DView diag_, const scalar_t alpha_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), diag( diag_ ), alpha( alpha_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        for ( lidx_t c = rowstarts( row ); c < rowstarts( row + 1 ); ++c ) {
            coeffs( c ) *= alpha * diag( row );
        }
    }
};

template<typename Config, class RSView, class AAView, class DView>
struct ScaleInv {
    typedef typename Config::lidx_t lidx_t;
    typedef typename Config::scalar_t scalar_t;

    RSView rowstarts;
    AAView coeffs;
    DView diag;
    const scalar_t alpha;

    ScaleInv( RSView rowstarts_, AAView coeffs_, DView diag_, const scalar_t alpha_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), diag( diag_ ), alpha( alpha_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        for ( lidx_t c = rowstarts( row ); c < rowstarts( row + 1 ); ++c ) {
            coeffs( c ) *= alpha / diag( row );
        }
    }
};

template<typename Config, class RSView, class AAView, class DView>
struct SetDiag {
    typedef typename Config::lidx_t lidx_t;

    RSView rowstarts;
    AAView coeffs;
    DView diag;

    SetDiag( RSView rowstarts_, AAView coeffs_, DView diag_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), diag( diag_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const { coeffs( rowstarts( row ) ) = diag( row ); }
};

template<typename Config, class RSView, class AAView, class DView>
struct ExtractDiag {
    typedef typename Config::lidx_t lidx_t;

    RSView rowstarts;
    AAView coeffs;
    DView diag;

    ExtractDiag( RSView rowstarts_, AAView coeffs_, DView diag_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), diag( diag_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const { diag( row ) = coeffs( rowstarts( row ) ); }
};

template<typename Config, class RSView, class AAView, class SView>
struct RowSums {
    typedef typename Config::lidx_t lidx_t;

    RSView rowstarts;
    AAView coeffs;
    SView sums;

    RowSums( RSView rowstarts_, AAView coeffs_, SView sums_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), sums( sums_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        for ( lidx_t c = rowstarts( row ); c < rowstarts( row + 1 ); ++c ) {
            sums( row ) += coeffs( c );
        }
    }
};

template<typename Config, class RSView, class AAView, class SView>
struct AbsRowSums {
    typedef typename Config::lidx_t lidx_t;

    RSView rowstarts;
    AAView coeffs;
    SView sums;

    AbsRowSums( RSView rowstarts_, AAView coeffs_, SView sums_ )
        : rowstarts( rowstarts_ ), coeffs( coeffs_ ), sums( sums_ )
    {
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        for ( lidx_t c = rowstarts( row ); c < rowstarts( row + 1 ); ++c ) {
            sums( row ) += Kokkos::fabs( coeffs( c ) );
        }
    }
};

template<typename Config, class SView>
struct RemoveZeros {
    typedef typename Config::lidx_t lidx_t;

    SView sums;

    RemoveZeros( SView sums_ ) : sums( sums_ ) {}

    KOKKOS_INLINE_FUNCTION
    void operator()( const lidx_t row ) const
    {
        sums( row ) = sums( row ) != 0.0 ? sums( row ) : 1.0;
    }
};

} // namespace CSRMatOpsKokkosFunctor

namespace impl {
template<typename ExecSpace,
         typename lidx_t,
         typename scalar_t,
         typename INView,
         typename OUTView,
         typename RSView,
         typename JAView,
         typename AAView>
void mult( ExecSpace exec_space,
           INView in,
           const scalar_t alpha,
           const lidx_t nRows,
           RSView rowstarts,
           JAView cols_loc,
           AAView coeffs,
           const scalar_t beta,
           OUTView out )
{

    // rows per team and vector length influenced by KokkosKernels
    // should tune to architecture (AMD vs. NVidia) and "typical" problems
    const lidx_t team_rows = 64;

    CSRMatOpsKokkosFunctor::
        aAxpby<ExecSpace, lidx_t, scalar_t, RSView, JAView, AAView, INView, OUTView>
            ftor( nRows, team_rows, rowstarts, cols_loc, coeffs, alpha, beta, in, out );

    if constexpr ( std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace> ) {
        const lidx_t num_teams     = ( nRows + team_rows - 1 ) / team_rows;
        const lidx_t vector_length = 8;
        Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>> team_policy(
            exec_space, num_teams, Kokkos::AUTO, vector_length );
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::mult (local - team)", team_policy, ftor );
    } else {
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::mult (local - flat)",
                              Kokkos::RangePolicy<ExecSpace>( exec_space, 0, nRows ),
                              ftor );
    }
}

} // namespace impl

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::mult( const typename Config::scalar_t *in,
                                                   const AMP::Utilities::MemoryType in_loc,
                                                   const typename Config::scalar_t alpha,
                                                   std::shared_ptr<localmatrixdata_t> A,
                                                   const typename Config::scalar_t beta,
                                                   typename Config::scalar_t *out,
                                                   const AMP::Utilities::MemoryType out_loc )
{
    const auto nRows = A->numLocalRows();
    const auto nCols = A->numUniqueColumns();
    AMP_DEBUG_ASSERT( nCols > 0 );

    // wrap matrix data in views
    const auto [rowstarts, cols_loc, coeffs] = wrapCSRDataKokkos( A );
    // Wrap in/out data into Kokkos Views
    auto in_view =
        WrapVector<const scalar_t, Kokkos::MemoryTraits<Kokkos::RandomAccess>>( in, nCols );
    auto out_view = WrapVector<scalar_t>( out, nRows );

    // pick an execution space based on memory spaces
    const bool device_exec =
        memoryLocationsDeviceAccessible( A->d_memory_location, in_loc, out_loc );

    if ( !device_exec ) {
        impl::mult(
            d_exec_host, in_view, alpha, nRows, rowstarts, cols_loc, coeffs, beta, out_view );
    } else {
        impl::mult(
            d_exec_device, in_view, alpha, nRows, rowstarts, cols_loc, coeffs, beta, out_view );
    }
}

namespace impl {
template<typename ExecSpace,
         typename lidx_t,
         typename INView,
         typename OUTView,
         typename RSView,
         typename JAView,
         typename AAView>
void multTranspose( ExecSpace exec_space,
                    INView in,
                    const lidx_t nRows,
                    RSView rowstarts,
                    JAView cols_loc,
                    AAView coeffs,
                    OUTView out )
{
    // rows per team and vector length influenced by KokkosKernels
    // should tune to architecture (AMD vs. NVidia) and "typical" problems
    const lidx_t team_rows     = 64;
    const lidx_t vector_length = 8;
    const lidx_t num_teams     = ( nRows + team_rows - 1 ) / team_rows;

    CSRMatOpsKokkosFunctor::
        MultTranspose<ExecSpace, lidx_t, RSView, JAView, AAView, INView, OUTView>
            ftor( nRows, team_rows, rowstarts, cols_loc, coeffs, in, out );

    if constexpr ( std::is_same_v<ExecSpace, Kokkos::DefaultExecutionSpace> && false ) {
        Kokkos::TeamPolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>> team_policy(
            exec_space, num_teams, Kokkos::AUTO, vector_length );
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::multTranspose (local - team)", team_policy, ftor );
    } else {
        Kokkos::parallel_for( "CSRMatrixOperationsKokkos::multTranspose (local - flat)",
                              Kokkos::RangePolicy<ExecSpace>( exec_space, 0, nRows ),
                              ftor );
    }
}
} // namespace impl

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::multTranspose(
    const typename Config::scalar_t *in,
    const AMP::Utilities::MemoryType in_loc,
    std::shared_ptr<localmatrixdata_t> A,
    typename Config::scalar_t *out,
    const AMP::Utilities::MemoryType out_loc )
{
    const auto nRows    = A->numLocalRows();
    const auto nCols    = A->numLocalColumns();
    const auto nColsUnq = A->numUniqueColumns();

    const auto [rowstarts, cols_loc, coeffs] = wrapCSRDataKokkos( A );

    // Wrap in/out data into Kokkos Views
    auto in_view =
        WrapVector<const scalar_t, Kokkos::MemoryTraits<Kokkos::RandomAccess>>( in, nCols );
    auto out_view = WrapVector<scalar_t, Kokkos::MemoryTraits<Kokkos::Atomic>>( out, nColsUnq );

    // pick an execution space based on memory spaces
    const bool device_exec =
        memoryLocationsDeviceAccessible( A->d_memory_location, in_loc, out_loc );

    if ( !device_exec ) {
        impl::multTranspose( d_exec_host, in_view, nRows, rowstarts, cols_loc, coeffs, out_view );
    } else {
        impl::multTranspose( d_exec_device, in_view, nRows, rowstarts, cols_loc, coeffs, out_view );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::scale( typename Config::scalar_t alpha,
                                                    std::shared_ptr<localmatrixdata_t> A )
{
    const auto vTpl = wrapCSRDataKokkos( A );
    auto coeffs     = std::get<2>( vTpl );

    const auto tnnz = A->numberOfNonZeros();

    if constexpr ( localmatrixdata_t::d_memory_location >= AMP::Utilities::MemoryType::managed ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scale",
            Kokkos::RangePolicy( d_exec_device, 0, tnnz ),
            KOKKOS_LAMBDA( lidx_t n ) { coeffs( n ) *= alpha; } );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scale",
            Kokkos::RangePolicy( d_exec_host, 0, tnnz ),
            KOKKOS_LAMBDA( lidx_t n ) { coeffs( n ) *= alpha; } );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::scale( typename Config::scalar_t alpha,
                                                    const typename Config::scalar_t *D,
                                                    const AMP::Utilities::MemoryType D_loc,
                                                    std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto D_view = WrapVector<const scalar_t>( D, nRows );

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, D_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scale",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                Scale<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view, alpha ) );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scale",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                Scale<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view, alpha ) );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::scaleInv( typename Config::scalar_t alpha,
                                                       const typename Config::scalar_t *D,
                                                       const AMP::Utilities::MemoryType D_loc,
                                                       std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto D_view = WrapVector<const scalar_t>( D, nRows );

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, D_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scaleInv",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                ScaleInv<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view, alpha ) );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::scaleInv",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                ScaleInv<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view, alpha ) );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::matMatMult( std::shared_ptr<localmatrixdata_t>,
                                                         std::shared_ptr<localmatrixdata_t>,
                                                         std::shared_ptr<localmatrixdata_t> )
{
    AMP_WARNING( "matMatMult for CSRLocalMatrixOperationsKokkos not implemented" );
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::axpy( typename Config::scalar_t alpha,
                                                   std::shared_ptr<localmatrixdata_t> X,
                                                   std::shared_ptr<localmatrixdata_t> Y )
{
    const auto vTplX = wrapCSRDataKokkos( X );
    auto rsX         = std::get<0>( vTplX );
    auto colslocX    = std::get<1>( vTplX );
    auto coeffsX     = std::get<2>( vTplX );

    const auto vTplY = wrapCSRDataKokkos( Y );
    auto rsY         = std::get<0>( vTplY );
    auto colslocY    = std::get<1>( vTplY );
    auto coeffsY     = std::get<2>( vTplY );

    const auto nRows = static_cast<lidx_t>( X->numLocalRows() );

    if constexpr ( localmatrixdata_t::d_memory_location >= AMP::Utilities::MemoryType::managed ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::axpy",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            KOKKOS_LAMBDA( lidx_t row ) {
                for ( lidx_t iy = rsY[row]; iy < rsY[row + 1]; ++iy ) {
                    const auto yc = colslocY[iy];
                    for ( lidx_t ix = rsX[row]; ix < rsX[row + 1]; ++ix ) {
                        if ( yc == colslocX[ix] ) {
                            coeffsY[iy] += alpha * coeffsX[ix];
                            break;
                        }
                    }
                }
            } );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::axpy",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            KOKKOS_LAMBDA( lidx_t row ) {
                for ( lidx_t iy = rsY[row]; iy < rsY[row + 1]; ++iy ) {
                    const auto yc = colslocY[iy];
                    for ( lidx_t ix = rsX[row]; ix < rsX[row + 1]; ++ix ) {
                        if ( yc == colslocX[ix] ) {
                            coeffsY[iy] += alpha * coeffsX[ix];
                            break;
                        }
                    }
                }
            } );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::setScalar( typename Config::scalar_t alpha,
                                                        std::shared_ptr<localmatrixdata_t> A )
{
    const auto vTpl = wrapCSRDataKokkos( A );
    auto coeffs     = std::get<2>( vTpl );
    Kokkos::deep_copy( coeffs, alpha );
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::zero( std::shared_ptr<localmatrixdata_t> A )
{
    setScalar( 0.0, A );
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::setDiagonal( const typename Config::scalar_t *D,
                                                          const AMP::Utilities::MemoryType D_loc,
                                                          std::shared_ptr<localmatrixdata_t> A )
{
    if ( !A->isDiag() ) {
        AMP_WARNING( "Attempted to call CSRLocalMatrixOperationsKokkos::setDiagonal on "
                     "off-diagonal block. Ignoring." );
        return;
    }

    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto D_view = WrapVector<const scalar_t>( D, nRows );

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, D_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::setDiagonal",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                SetDiag<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view ) );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::setDiagonal",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                SetDiag<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view ) );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::setIdentity( std::shared_ptr<localmatrixdata_t> A )
{
    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    if ( !A->isDiag() ) {
        return;
    }

    if constexpr ( localmatrixdata_t::d_memory_location >= AMP::Utilities::MemoryType::managed ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::setIdentity",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            KOKKOS_LAMBDA( lidx_t row ) { coeffs( rowstarts( row ) ) = 1.0; } );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::setIdentity",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            KOKKOS_LAMBDA( lidx_t row ) { coeffs( rowstarts( row ) ) = 1.0; } );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::extractDiagonal(
    std::shared_ptr<localmatrixdata_t> A,
    typename Config::scalar_t *D,
    const AMP::Utilities::MemoryType D_loc )
{
    if ( !A->isDiag() ) {
        AMP_WARNING( "Attempted to call CSRLocalMatrixOperationsKokkos::extractDiagonal on "
                     "off-diagonal block. Ignoring." );
        return;
    }

    const auto nRows = static_cast<lidx_t>( A->numLocalRows() );

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto D_view = WrapVector<scalar_t>( D, nRows );

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, D_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::extractDiagonal",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                ExtractDiag<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view ) );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::extractDiagonal",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                ExtractDiag<Config, decltype( rowstarts ), decltype( coeffs ), decltype( D_view )>(
                    rowstarts, coeffs, D_view ) );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::getRowSums( std::shared_ptr<localmatrixdata_t> A,
                                                         typename Config::scalar_t *buf,
                                                         const AMP::Utilities::MemoryType buf_loc,
                                                         const bool zero_first ) const
{
    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto buf_view = WrapVector<scalar_t>( buf, nRows );

    if ( zero_first ) {
        Kokkos::deep_copy( buf_view, 0.0 );
    }

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, buf_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::getRowSums",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                RowSums<Config, decltype( rowstarts ), decltype( coeffs ), decltype( buf_view )>(
                    rowstarts, coeffs, buf_view ) );
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::getRowSums",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                RowSums<Config, decltype( rowstarts ), decltype( coeffs ), decltype( buf_view )>(
                    rowstarts, coeffs, buf_view ) );
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::getRowSumsAbsolute(
    std::shared_ptr<localmatrixdata_t> A,
    typename Config::scalar_t *buf,
    const AMP::Utilities::MemoryType buf_loc,
    const bool zero_first,
    const bool remove_zeros ) const
{
    const auto nRows = A->numLocalRows();

    const auto vTpl = wrapCSRDataKokkos( A );
    auto rowstarts  = std::get<0>( vTpl );
    auto coeffs     = std::get<2>( vTpl );

    // Wrap D into Kokkos View
    auto buf_view = WrapVector<scalar_t>( buf, nRows );

    if ( zero_first ) {
        Kokkos::deep_copy( buf_view, 0.0 );
    }

    const bool device_exec = memoryLocationsDeviceAccessible( A->d_memory_location, buf_loc );

    if ( device_exec ) {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::getRowSumsAbsolute",
            Kokkos::RangePolicy( d_exec_device, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                AbsRowSums<Config, decltype( rowstarts ), decltype( coeffs ), decltype( buf_view )>(
                    rowstarts, coeffs, buf_view ) );
        if ( remove_zeros ) {
            Kokkos::parallel_for(
                "CSRMatrixOperationsKokkos::getRowSumsAbsolute(remove zeros)",
                Kokkos::RangePolicy( d_exec_device, 0, nRows ),
                CSRMatOpsKokkosFunctor::RemoveZeros<Config, decltype( buf_view )>( buf_view ) );
        }
    } else {
        Kokkos::parallel_for(
            "CSRMatrixOperationsKokkos::getRowSumsAbsolute",
            Kokkos::RangePolicy( d_exec_host, 0, nRows ),
            CSRMatOpsKokkosFunctor::
                AbsRowSums<Config, decltype( rowstarts ), decltype( coeffs ), decltype( buf_view )>(
                    rowstarts, coeffs, buf_view ) );
        if ( remove_zeros ) {
            Kokkos::parallel_for(
                "CSRMatrixOperationsKokkos::getRowSumsAbsolute(remove zeros)",
                Kokkos::RangePolicy( d_exec_host, 0, nRows ),
                CSRMatOpsKokkosFunctor::RemoveZeros<Config, decltype( buf_view )>( buf_view ) );
        }
    }
}

template<typename Config>
void CSRLocalMatrixOperationsKokkos<Config>::copy( std::shared_ptr<const localmatrixdata_t> X,
                                                   std::shared_ptr<localmatrixdata_t> Y )
{
    const auto vTplX = wrapCSRDataKokkos( X );
    auto coeffsX     = std::get<2>( vTplX );

    const auto vTplY = wrapCSRDataKokkos( Y );
    auto coeffsY     = std::get<2>( vTplY );

    Kokkos::deep_copy( coeffsY, coeffsX );
}

template<typename Config>
template<typename ConfigIn>
void CSRLocalMatrixOperationsKokkos<Config>::copyCast(
    std::shared_ptr<CSRLocalMatrixData<ConfigIn>> X, std::shared_ptr<localmatrixdata_t> Y )
{
    // Check compatibility
    AMP_ASSERT( Y->beginRow() == X->beginRow() );
    AMP_ASSERT( Y->endRow() == X->endRow() );
    AMP_ASSERT( Y->beginCol() == X->beginCol() );
    AMP_ASSERT( Y->endCol() == X->endCol() );

    AMP_ASSERT( Y->numberOfNonZeros() == X->numberOfNonZeros() );

    AMP_ASSERT( Y->numLocalRows() == X->numLocalRows() );
    AMP_ASSERT( Y->numUniqueColumns() == X->numUniqueColumns() );

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
        const auto X_v = Kokkos::View<scalar_t_in *, Kokkos::LayoutRight, csr_memspace_t>(
            X_coeffs, X->numberOfNonZeros() );
        auto Y_v = Kokkos::View<scalar_t_out *, Kokkos::LayoutRight, csr_memspace_t>(
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
