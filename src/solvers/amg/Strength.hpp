#ifndef included_AMP_AMG_STRENGTH_hpp
#define included_AMP_AMG_STRENGTH_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Strength.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <numeric>
#include <type_traits>

#ifdef AMP_USE_DEVICE
    #define AMP_FUNCTION __host__ __device__
    #define AMP_LAMBDA [=] __host__ __device__
    // not side-effect safe, pass single scalars only
    #define AMP_FABS( v ) ( -( v ) > ( v ) ? -( v ) : ( v ) )
#else
    #define AMP_FUNCTION
    #define AMP_LAMBDA [=]
    #define AMP_FABS( v ) std::fabs( v )
#endif

namespace AMP::Solver::AMG {

// classical measure compares against either the most negative entry
// or largest entry in absolute value, connection is strong if either
// -a_ij >= -thresh * min( a_i: )
// or
// |a_ij| >= thresh * max( |a_i:| )
template<norm norm_type>
struct classical_strength {
    template<class T>
    static constexpr std::remove_cv_t<T> strongest( span<T> s )
    {
        if ( s.size() == 0 || s.begin() == nullptr ) {
            return 0;
        }
        if constexpr ( norm_type == norm::min ) {
            auto el = std::min_element( s.begin(), s.end() );
            return el == s.end() ? std::numeric_limits<T>::lowest() : -*el;
        } else {
            auto el = std::max_element( s.begin(), s.end(), []( const T &a, const T &b ) {
                return std::abs( a ) < std::abs( b );
            } );
            return el == s.end() ? std::numeric_limits<T>::lowest() : std::abs( *el );
        }
    }

    template<class T>
    AMP_FUNCTION static constexpr std::remove_cv_t<T> strongest( T *s, int len )
    {
        if ( len == 0 ) {
            return 0;
        }
        if constexpr ( norm_type == norm::min ) {
            std::remove_cv_t<T> el = std::numeric_limits<T>::max();
            for ( int n = 0; n < len; ++n ) {
                el = el < s[n] ? el : s[n];
            }
            return -el;
        } else {
            std::remove_cv_t<T> el{ 0 };
            for ( int n = 0; n < len; ++n ) {
                el = el > AMP_FABS( s[n] ) ? el : AMP_FABS( s[n] );
            }
            return el;
        }
    }

    template<class T>
    AMP_FUNCTION static bool is_strong( T strongest_connection, float threshold, T, T, T val )
    {
        if ( strongest_connection == 0 )
            return false;
        if constexpr ( norm_type == norm::min ) {
            return -val >= threshold * strongest_connection;
        } else {
            const auto aval = AMP_FABS( val );
            return aval >= threshold * strongest_connection;
        }
    }
};

// symmetric aggregation style strength compares against the root
// of the product of diagonal entries, connection is strong if either
// -a_ij >= thresh * sqrt( |a_ii| * |a_jj| )
// or
// |a_ij| >= thresh * sqrt( |a_ii| * |a_jj| )
// this is symmetric since both a_ij and a_ji see same rhs in the
// comparisons
template<norm norm_type>
struct symagg_strength {
    template<class T>
    static constexpr std::remove_cv_t<T> strongest( span<T> )
    {
        return 0;
    }

    template<class T>
    AMP_FUNCTION static constexpr std::remove_cv_t<T> strongest( T *, int )
    {
        return 0;
    }

    template<class T>
    AMP_FUNCTION static bool is_strong( T, float threshold, T Aii, T Ajj, T val )
    {
        // square whole test expression to avoid needing device sqrt
        const auto Fii = AMP_FABS( Aii );
        const auto Fjj = AMP_FABS( Ajj );

        const bool abs_strong = ( val * val >= threshold * threshold * Fii * Fjj );
        if constexpr ( norm_type == norm::min ) {
            return abs_strong && val < 0.0;
        } else {
            return abs_strong;
        }
    }
};

template<class Mat>
AMG::Strength<Mat>::Strength( csr_view<Mat> A )
{
    auto init = [=]( auto src, auto &dst ) {
        auto [rowptr, colind, values] = src;
        dst.rowptr                    = rowptr;
        dst.colind                    = colind;
        dst.mat_values                = values;
        dst.values.resize( colind.size() );
    };
    init( A.diag(), d_diag );
    init( A.offd(), d_offd );
}

#ifdef AMP_USE_DEVICE

template<class StrengthPolicy, typename lidx_t, typename scalar_t, typename mask_t>
__global__ void compute_soc_device( StrengthPolicy,
                                    const lidx_t *row_ptr_diag,
                                    const lidx_t *cols_loc_diag,
                                    const scalar_t *vals_diag,
                                    const lidx_t *row_ptr_offd,
                                    const scalar_t *vals_offd,
                                    const lidx_t num_rows,
                                    const float threshold,
                                    mask_t *strength_diag )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
          i += blockDim.x * gridDim.x ) {
        const auto rs_diag = row_ptr_diag[i], re_diag = row_ptr_diag[i + 1];
        const auto row_len_diag = re_diag - rs_diag;
        if ( row_len_diag == 0 ) {
            continue;
        }
        const auto rs_offd = row_ptr_offd[i], re_offd = row_ptr_offd[i + 1];
        const auto row_len_offd = re_offd - rs_offd;

        // diagonal block rowsum, diagonal value
        const auto Aii = vals_diag[rs_diag];

        // strongest connection per block row and overall
        const auto strongest_diag = row_len_diag > 1 ?
                                        StrengthPolicy::template strongest<const scalar_t>(
                                            &vals_diag[rs_diag + 1], row_len_diag - 1 ) :
                                        0.0;
        const auto strongest_offd = row_len_offd > 0 ?
                                        StrengthPolicy::template strongest<const scalar_t>(
                                            &vals_offd[rs_offd], row_len_offd ) :
                                        0.0;
        const auto strongest_connection =
            strongest_diag > strongest_offd ? strongest_diag : strongest_offd;

        for ( lidx_t n = rs_diag; n < re_diag; ++n ) {
            const auto rs_j = row_ptr_diag[cols_loc_diag[n]];
            const auto Ajj  = vals_diag[rs_j];
            const bool str  = StrengthPolicy::template is_strong<const scalar_t>(
                strongest_connection, threshold, Aii, Ajj, vals_diag[n] );
            strength_diag[n] = str ? 1 : 0;
        }
    }
}

#endif

template<class StrengthPolicy, class Mat>
Strength<Mat> compute_soc( csr_view<Mat> A, float threshold )
{
    using lidx_t   = typename csr_view<Mat>::lidx_t;
    using scalar_t = typename csr_view<Mat>::scalar_t;
    using mask_t   = typename csr_view<Mat>::mask_t;

    Strength S( A );

    if constexpr ( std::is_same_v<typename csr_view<Mat>::allocator_type,
                                  AMP::HostAllocator<void>> ) {

        auto get_colsloc = [=]( auto csr_ptrs, lidx_t r ) {
            auto rowptr = std::get<0>( csr_ptrs );
            auto cols   = std::get<1>( csr_ptrs );
            return cols.subspan( rowptr[r], rowptr[r + 1] - rowptr[r] );
        };

        auto get_vals = [=]( auto csr_ptrs, lidx_t r ) {
            auto rowptr = std::get<0>( csr_ptrs );
            auto values = std::get<2>( csr_ptrs );
            return values.subspan( rowptr[r], rowptr[r + 1] - rowptr[r] );
        };

        auto get_diag_val = [=]( lidx_t r ) {
            auto rowptr = std::get<0>( A.diag() );
            auto values = std::get<2>( A.diag() );
            return values[rowptr[r]];
        };

        for ( size_t r = 0; r < A.numLocalRows(); ++r ) {
            auto Ad_cols = get_colsloc( A.diag(), r );
            auto Ad_vals = get_vals( A.diag(), r );
            if ( Ad_vals.size() == 0 )
                continue;

            const scalar_t Aii = get_diag_val( r );
            AMP_ASSERT( Aii > 0 );

            auto strongest_connection =
                StrengthPolicy::template strongest<const scalar_t>( Ad_vals.subspan( 1 ) );
            if ( A.has_offd() )
                strongest_connection = std::max(
                    strongest_connection,
                    StrengthPolicy::template strongest<const scalar_t>( get_vals( A.offd(), r ) ) );

            auto fill_strength = [&]( auto vals, auto cols, auto strength ) {
                for ( std::size_t i = 0; i < vals.size(); ++i ) {
                    const scalar_t Ajj = get_diag_val( cols[i] );
                    const bool str     = StrengthPolicy::template is_strong<const scalar_t>(
                        strongest_connection, threshold, Aii, Ajj, vals[i] );
                    strength[i] = str ? 1 : 0;
                }
            };

            fill_strength( Ad_vals, Ad_cols, S.diag_row( r ) );
        }
    } else {
#ifdef AMP_USE_DEVICE
        const auto num_rows      = static_cast<lidx_t>( A.numLocalRows() );
        const auto row_ptr_diag  = std::get<0>( A.diag() ).data();
        const auto cols_loc_diag = std::get<1>( A.diag() ).data();
        const auto vals_diag     = std::get<2>( A.diag() ).data();
        const auto row_ptr_offd  = std::get<0>( A.offd() ).data();
        const auto vals_offd     = std::get<2>( A.offd() ).data();
        mask_t *mask_data        = S.diag_mask_data();

        AMP_DEBUG_ASSERT( AMP::Utilities::getMemoryType( row_ptr_diag ) >
                          AMP::Utilities::MemoryType::host );
        AMP_DEBUG_ASSERT( AMP::Utilities::getMemoryType( vals_diag ) >
                          AMP::Utilities::MemoryType::host );

        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        compute_soc_device<<<GridDim, BlockDim>>>( StrengthPolicy{},
                                                   row_ptr_diag,
                                                   cols_loc_diag,
                                                   vals_diag,
                                                   row_ptr_offd,
                                                   vals_offd,
                                                   num_rows,
                                                   threshold,
                                                   mask_data );
        getLastDeviceError( "compute_soc" );
#else
        AMP_ERROR( "compute_soc: Undefined memory location" );
#endif
    }

    return S;
}

} // namespace AMP::Solver::AMG

#undef AMP_FUNCTION
#undef AMP_LAMBDA
#undef AMP_FABS

#endif
