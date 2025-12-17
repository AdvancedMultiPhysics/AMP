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
// -a_ij >= thresh * min( a_i: )
// or
// |a_ij| >= thresh * max( |a_i:| )
template<norm norm_type>
struct classical_strength {
    template<class T>
    static constexpr std::remove_cv_t<T> DSum( span<T> )
    {
        return 0;
    }

    template<class T>
    AMP_FUNCTION static constexpr std::remove_cv_t<T> DSum( T *, int )
    {
        return 0;
    }

    template<class T>
    static constexpr std::remove_cv_t<T>
    strongest( span<T> s, [[maybe_unused]] T D = 0.0, [[maybe_unused]] T Aii = 0.0 )
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
    AMP_FUNCTION static constexpr std::remove_cv_t<T>
    strongest( T *s, int len, [[maybe_unused]] T D = 0.0, [[maybe_unused]] T Aii = 0.0 )
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
    AMP_FUNCTION static auto is_strong( T strongest_connection,
                                        float threshold,
                                        [[maybe_unused]] T D   = 0.0,
                                        [[maybe_unused]] T Aii = 0.0 )
    {
        return AMP_LAMBDA( T val )
        {
            if ( strongest_connection == 0 )
                return false;
            if constexpr ( norm_type == norm::min ) {
                return -val >= threshold * strongest_connection;
            } else {
                const auto aval = AMP_FABS( val );
                return aval >= threshold * strongest_connection;
            }
        };
    }
};

// Evolution measure applies one step of relaxation to delta function
// z_: = delta_i - Dinv * A_i:
// where Dinv is one over absolute row sum (Jacobi L1 relaxation) and
// symmetry of A is assumed.
// Strength then sets
// M_ii = 0
// M_ij = | 1 - z_i / z_j | = | 1 + (D - A_ii)/A_ij |
// and the threshold sets strong connections as
// m = min_k (M_ik : k != i)
// S_ij = M_ij if M_ij < (1/thresh) * m, otherwise weak
struct evolution_strength {
    template<class T>
    static T DSum( span<T> s )
    {
        if ( s.size() == 0 || s.begin() == nullptr ) {
            return 0;
        }
        return std::transform_reduce(
            s.begin(), s.end(), 0.0, std::plus{}, []( auto v ) { return std::fabs( v ); } );
    }

    template<class T>
    AMP_FUNCTION static std::remove_cv_t<T> DSum( T *s, int len )
    {

        std::remove_cv_t<T> sum{ 0 };
        for ( int n = 0; n < len; ++n ) {
            sum += AMP_FABS( s[n] );
        }
        return sum;
    }

    template<class T>
    AMP_FUNCTION static constexpr T Sij( T D, T Aii, T Aij )
    {
        const auto v  = 1.0 + ( D - Aii ) / Aij;
        const auto av = AMP_FABS( v );
        return Aij != 0.0 ? av : std::numeric_limits<T>::max();
    }

    template<class T>
    static T strongest( span<T> s, T D, T Aii )
    {
        if ( s.size() == 0 || s.begin() == nullptr ) {
            return 0;
        }
        // get smallest measure ignoring stored A_ij == 0 entries
        const auto m = std::transform_reduce(
            s.begin(),
            s.end(),
            std::numeric_limits<T>::max(),
            []( auto l, auto r ) { return std::fmin( l, r ); },
            [D, Aii]( T val ) { return Sij( D, Aii, val ); } );
        return m;
    }

    template<class T>
    AMP_FUNCTION static constexpr std::remove_cv_t<T>
    strongest( T *s, int len, [[maybe_unused]] T D = 0.0, [[maybe_unused]] T Aii = 0.0 )
    {
        if ( len == 0 ) {
            return 0;
        }
        std::remove_cv_t<T> el = std::numeric_limits<T>::max();
        for ( int n = 0; n < len; ++n ) {
            const auto sij = Sij( D, Aii, s[n] );
            el             = el < sij ? el : sij;
        }
        return el;
    }

    template<class T>
    AMP_FUNCTION static auto is_strong( T strongest_connection, float threshold, T D, T Aii )
    {
        return AMP_LAMBDA( T val )
        {
            if ( strongest_connection == 0 )
                return false;
            return Sij( D, Aii, val ) < strongest_connection / threshold;
        };
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
        const auto D   = StrengthPolicy::DSum( &vals_diag[rs_diag], row_len_diag );
        const auto Aii = vals_diag[rs_diag];

        // strongest connection per block row and overall
        const auto strongest_diag = row_len_diag > 1 ?
                                        StrengthPolicy::template strongest<const scalar_t>(
                                            &vals_diag[rs_diag + 1], row_len_diag - 1, D, Aii ) :
                                        0.0;
        const auto strongest_offd = row_len_offd > 1 ?
                                        StrengthPolicy::template strongest<const scalar_t>(
                                            &vals_offd[rs_offd + 1], row_len_offd - 1, D, Aii ) :
                                        0.0;
        const auto strongest_connection =
            strongest_diag > strongest_offd ? strongest_diag : strongest_offd;

        // strength test lambda
        auto is_strong = StrengthPolicy::template is_strong<const scalar_t>(
            strongest_connection, threshold, D, Aii );

        for ( lidx_t n = rs_diag; n < re_diag; ++n ) {
            strength_diag[n] = is_strong( vals_diag[n] ) ? 1.0 : 0.0;
        }
    }
}

#endif

template<class StrengthPolicy, class Mat>
Strength<Mat> compute_soc( csr_view<Mat> A, float threshold )
{
    using lidx_t   = typename csr_view<Mat>::lidx_t;
    using scalar_t = typename csr_view<Mat>::scalar_t;

    Strength S( A );

    if constexpr ( std::is_same_v<typename csr_view<Mat>::allocator_type,
                                  AMP::HostAllocator<void>> ) {

        auto get_row = [=]( auto csr_ptrs ) {
            auto rowptr = std::get<0>( csr_ptrs );
            auto values = std::get<2>( csr_ptrs );
            return
                [=]( lidx_t r ) { return values.subspan( rowptr[r], rowptr[r + 1] - rowptr[r] ); };
        };

        auto diag_rows = get_row( A.diag() );
        auto offd_rows = get_row( A.offd() );

        for ( size_t r = 0; r < A.numLocalRows(); ++r ) {
            auto diag_values = diag_rows( r );
            if ( diag_values.size() == 0 )
                continue;

            const scalar_t D   = StrengthPolicy::DSum( diag_values );
            const scalar_t Aii = diag_values[0];

            auto strongest_connection = StrengthPolicy::template strongest<const scalar_t>(
                diag_values.subspan( 1 ), D, Aii );
            if ( A.has_offd() )
                strongest_connection = std::max(
                    strongest_connection,
                    StrengthPolicy::template strongest<const scalar_t>( offd_rows( r ), D, Aii ) );

            auto is_strong = StrengthPolicy::template is_strong<const scalar_t>(
                strongest_connection, threshold, D, Aii );
            auto fill_strength = [&]( auto vals, auto strength ) {
                for ( std::size_t i = 0; i < vals.size(); ++i ) {
                    strength[i] = is_strong( vals[i] ) ? 1 : 0;
                }
            };

            fill_strength( diag_values, S.diag_row( r ) );
            // fill_strength(offd_values, S.offd_row(r));
        }
    } else {
#ifdef AMP_USE_DEVICE
        using mask_t = typename csr_view<Mat>::mask_t;

        const auto num_rows     = static_cast<lidx_t>( A.numLocalRows() );
        const auto row_ptr_diag = std::get<0>( A.diag() ).data();
        const auto vals_diag    = std::get<2>( A.diag() ).data();
        const auto row_ptr_offd = std::get<0>( A.offd() ).data();
        const auto vals_offd    = std::get<2>( A.offd() ).data();
        mask_t *mask_data       = S.diag_mask_data();

        AMP_DEBUG_ASSERT( AMP::Utilities::getMemoryType( row_ptr_diag ) >
                          AMP::Utilities::MemoryType::host );
        AMP_DEBUG_ASSERT( AMP::Utilities::getMemoryType( vals_diag ) >
                          AMP::Utilities::MemoryType::host );

        dim3 BlockDim;
        dim3 GridDim;
        setKernelDims( num_rows, BlockDim, GridDim );
        compute_soc_device<<<GridDim, BlockDim>>>( StrengthPolicy{},
                                                   row_ptr_diag,
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
