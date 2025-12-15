#ifndef included_AMP_AMG_STRENGTH_hpp
#define included_AMP_AMG_STRENGTH_hpp

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Strength.h"

#include <algorithm>
#include <numeric>

namespace AMP::Solver::AMG {

// classical measure compares against either the most negative entry
// or largest entry in absolute value, connection is strong if either
// -a_ij >= thresh * min( a_i: )
// or
// |a_ij| >= thresh * max( |a_i:| )
template<norm norm_type>
struct classical_strength {
    template<class T>
    static constexpr T DSum( span<T> )
    {
        return 0;
    }

    template<class T>
    static constexpr T
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
    static auto is_strong( T strongest_connection,
                           float threshold,
                           [[maybe_unused]] T D   = 0.0,
                           [[maybe_unused]] T Aii = 0.0 )
    {
        return [=]( T val ) {
            if ( strongest_connection == 0 )
                return false;
            if constexpr ( norm_type == norm::min ) {
                return -val >= threshold * strongest_connection;
            } else {
                return std::abs( val ) >= threshold * strongest_connection;
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
    static constexpr T Sij( T D, T Aii, T Aij )
    {
        return Aij != 0.0 ? std::fabs( 1.0 + ( D - Aii ) / Aij ) : std::numeric_limits<T>::max();
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
    static auto is_strong( T strongest_connection, float threshold, T D, T Aii )
    {
        return [=]( T val ) {
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


template<class StrengthPolicy, class Mat>
Strength<Mat> compute_soc( csr_view<Mat> A, float threshold )
{
    using lidx_t   = typename csr_view<Mat>::lidx_t;
    using scalar_t = typename csr_view<Mat>::scalar_t;
    auto get_row   = [=]( auto csr_ptrs ) {
        auto rowptr = std::get<0>( csr_ptrs );
        auto values = std::get<2>( csr_ptrs );
        return [=]( lidx_t r ) { return values.subspan( rowptr[r], rowptr[r + 1] - rowptr[r] ); };
    };

    auto diag_rows = get_row( A.diag() );
    auto offd_rows = get_row( A.offd() );

    Strength S( A );

    for ( size_t r = 0; r < A.numLocalRows(); ++r ) {
        auto diag_values = diag_rows( r );
        if ( diag_values.size() == 0 )
            continue;

        const scalar_t D   = StrengthPolicy::DSum( diag_values );
        const scalar_t Aii = diag_values[0];

        auto strongest_connection =
            StrengthPolicy::template strongest<const scalar_t>( diag_values.subspan( 1 ), D, Aii );
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

    return S;
}


} // namespace AMP::Solver::AMG

#endif
