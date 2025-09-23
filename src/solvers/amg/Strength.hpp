#ifndef included_AMP_AMG_STRENGTH_hpp
#define included_AMP_AMG_STRENGTH_hpp

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/Strength.h"

namespace AMP::Solver::AMG {

template<norm norm_type>
struct classical_strength {
    template<class T>
    static constexpr T strongest( span<T> s )
    {
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
    static auto is_strong( T strongest_connection, float threshold )
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
    using lidx_t = typename csr_view<Mat>::lidx_t;
    auto get_row = [=]( auto csr_ptrs ) {
        auto rowptr = std::get<0>( csr_ptrs );
        auto values = std::get<2>( csr_ptrs );
        return [=]( lidx_t r ) { return values.subspan( rowptr[r], rowptr[r + 1] - rowptr[r] ); };
    };

    auto diag_rows = get_row( A.diag() );
    auto offd_rows = get_row( A.offd() );

    Strength S( A );

    for ( size_t r = 0; r < A.numLocalRows(); ++r ) {
        auto diag_values          = diag_rows( r );
        auto strongest_connection = StrengthPolicy::strongest( diag_values.subspan( 1 ) );
        if ( A.has_offd() )
            strongest_connection =
                std::max( strongest_connection, StrengthPolicy::strongest( offd_rows( r ) ) );
        auto is_strong     = StrengthPolicy::is_strong( strongest_connection, threshold );
        auto fill_strength = [&]( auto vals, auto strength ) {
            for ( std::size_t i = 0; i < vals.size(); ++i )
                strength[i] = is_strong( vals[i] ) ? 1 : 0;
        };
        fill_strength( diag_values, S.diag_row( r ) );
        // fill_strength(offd_values, S.offd_row(r));
    }

    return S;
}


} // namespace AMP::Solver::AMG

#endif
