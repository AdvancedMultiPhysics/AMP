// rpoly based off Jenkins–Traub algorithm (TOMS 493)
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace AMP {


/******************************************************************
 * Sort the roots                                                  *
 ******************************************************************/
static void sort( int n, std::complex<double> *roots )
{
    auto compare = []( const std::complex<double> &a, const std::complex<double> &b ) {
        if ( a.real() == b.real() )
            return a.imag() < b.imag();
        return a.real() < b.real();
    };
    std::sort( roots, roots + n, compare );
}


/******************************************************************
 * Quadratic solver                                                *
 ******************************************************************/
static void quad( double a, double b, double c, std::complex<double> *roots )
{
    double b2   = 0.5 * b;
    double disc = b2 * b2 - a * c;
    if ( disc >= 0 ) {
        double d  = std::sqrt( disc );
        double r1 = ( -b2 + std::copysign( d, -b2 ) ) / a;
        double r2 = ( c / r1 ) / a;
        roots[0]  = std::complex<double>( r1, 0 );
        roots[1]  = std::complex<double>( r2, 0 );
    } else {
        double real = -b2 / a;
        double imag = std::sqrt( -disc ) / a;
        roots[0]    = std::complex<double>( real, imag );
        roots[1]    = std::complex<double>( real, -imag );
    }
    if ( roots[1].real() < roots[0].real() )
        std::swap( roots[0], roots[1] );
    else if ( roots[1].imag() < roots[0].imag() && roots[1].real() == roots[0].real() )
        std::swap( roots[0], roots[1] );
}


/******************************************************************
 * Polynomial / Derivative evaluation                              *
 ******************************************************************/
template<class T>
static inline T eval( int n, const double *p, T x )
{
    T result = p[n];
    for ( int i = n - 1; i >= 0; --i )
        result = result * x + p[i];
    return result;
}
template<class T>
static inline T evalDeriv( int n, const double *p, T x )
{
    T result = 0.0;
    for ( int i = n - 1; i >= 0; --i )
        result = result * x + ( i + 1 ) * p[i];
    return result;
}


/******************************************************************
 * Newton iteration                                                *
 ******************************************************************/
template<class T>
static inline T refineRoot( int n, const double *p, T x0 )
{
    constexpr int max_iter = 20;
    constexpr double tol   = 1e-13;

    auto x = x0;
    for ( int i = 0; i < max_iter; ++i ) {
        auto f  = eval( n, p, x );
        auto df = evalDeriv( n, p, x );
        if ( std::abs( df ) < tol )
            break;
        auto dx = f / df;
        x -= dx;
        if ( std::abs( dx ) < tol )
            break;
    }
    return x;
}


/******************************************************************
 * Enforce real + conjugate pairing                                *
 ******************************************************************/
static void enforce_conjugate_pairs( int n,
                                     const double *coeffs,
                                     std::complex<double> *roots,
                                     double tol = 1e-10 )
{
    std::vector<std::complex<double>> result;
    result.reserve( n );

    // Case 1: real roots
    size_t N = 0;
    for ( int i = 0; i < n; ++i ) {
        auto r = roots[i];
        if ( std::abs( r.imag() / r.real() ) < tol ) {
            double r2 = refineRoot( n, coeffs, r.real() );
            result.emplace_back( r2, 0.0 );
            continue;
        } else {
            roots[N++] = roots[i];
        }
    }

    // Case 2: find conjugate partners
    std::vector<bool> used( n, false );
    sort( N, roots );
    for ( int i = 0; i < n; ++i ) {
        if ( used[i] )
            continue;

        auto r = roots[i];

        int best_j      = -1;
        double best_err = 1e100;

        for ( int j = i + 1; j < n; ++j ) {
            if ( used[j] )
                continue;

            auto candidate = roots[j];

            // want candidate ~= conjugate of r
            auto diff  = candidate - std::conj( r );
            double err = std::abs( diff );

            if ( err < best_err ) {
                best_err = err;
                best_j   = static_cast<int>( j );
            }
        }

        if ( best_j >= 0 && best_err < 1e-5 ) {
            // average to enforce exact conjugates
            auto r1 = roots[i];
            auto r2 = roots[best_j];

            double real_part = 0.5 * ( r1.real() + r2.real() );
            double imag_part = 0.5 * ( std::abs( r1.imag() ) + std::abs( r2.imag() ) );

            result.emplace_back( real_part, imag_part );
            result.emplace_back( real_part, -imag_part );

            used[i]      = true;
            used[best_j] = true;
        } else {
            // fallback: treat as real if nearly real
            if ( std::abs( r.imag() / r.real() ) < 1e-4 ) {
                result.emplace_back( r.real(), 0.0 );
                used[i] = true;
            } else {
                // All failed, keep complex for now
                result.emplace_back( r );
                used[i] = true;
            }
        }
    }

    for ( int i = 0; i < n; i++ )
        roots[i] = result[i];
}


/******************************************************************
 * rpoly interface                                                 *
 *    n - number of roots (x^n)                                    *
 *    coeffs - coefficients (n+1)                                  *
 *             a0 + a1*x + a2*x^2 + ... + an*x^n                   *
 *    roots - roots (n)                                            *
 ******************************************************************/
void rpoly( int n, const double *coeffs, std::complex<double> *roots )
{
    if ( n == 0 )
        throw std::invalid_argument( "Polynomial must not be empty" );

    if ( coeffs[n] == 0 )
        throw std::invalid_argument( "Leading coefficient cannot be zero" );

    if ( n > 1000 )
        throw std::invalid_argument( "rpoly is not stable for large number of roots" );

    if ( coeffs[0] == 0 ) {
        // Handle roots == 0 directly
        int i = 0;
        while ( coeffs[i] == 0 ) {
            roots[i] = 0;
            i++;
        }
        rpoly( n - i, &coeffs[i], &roots[i] );
        sort( n, roots );
        return;
    }

    // Handle trivial cases
    if ( n == 1 ) {
        roots[0] = std::complex<double>( -coeffs[0] / coeffs[1], 0 );
        return;
    } else if ( n == 2 ) {
        quad( coeffs[2], coeffs[1], coeffs[0], roots );
        return;
    }

    // Initial guesses distributed on circle
    constexpr double pi = 3.1415926535897932;
    for ( int i = 0; i < n; ++i ) {
        double theta = 2 * pi * ( 0.5 + i ) / n;
        roots[i]     = std::complex<double>( std::cos( theta ), std::sin( theta ) );
    }

#if 0
    // Aberth iteration
    constexpr int max_iter = 500;
    constexpr double tol   = 1e-12;
    for ( int iter = 0; iter < max_iter; ++iter ) {
        bool converged = true;
        for ( int i = 0; i < n; ++i ) {
            auto zi = roots[i];
            auto f = eval(n,coeffs, zi);
            auto fp = evalDeriv(n,coeffs, zi);
            if ( std::abs(f) < tol )
                continue;
            // Aberth correction term
            std::complex<double> sum = 0.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    auto diff = zi - roots[j];
                    if (std::abs(diff) > 1e-14)
                        sum += 1.0 / diff;
                }
            }
            auto denom = fp - f * sum;
            if (std::abs(denom) < tol)
                continue;
            auto delta = f / denom;
            roots[i] -= delta;
            if ( std::abs(delta) > tol )
                converged = false;
        }
        if (converged)
            break;
    }
#else
    // Durand–Kerner iteration
    constexpr int max_iter = 200;
    constexpr double tol   = 1e-12;
    for ( int iter = 0; iter < max_iter; ++iter ) {
        bool converged = true;
        for ( int i = 0; i < n; ++i ) {
            std::complex<double> denom = 1.0;
            for ( int j = 0; j < n; ++j )
                if ( i != j )
                    denom *= ( roots[i] - roots[j] );
            auto delta = eval( n, coeffs, roots[i] ) / denom;
            roots[i] -= delta;
            if ( std::abs( delta ) > tol )
                converged = false;
        }
        if ( converged )
            break;
    }
#endif

    // Refine roots
    for ( int i = 0; i < n; i++ )
        roots[i] = refineRoot( n, coeffs, roots[i] );

    // Clean / pair the roots
    enforce_conjugate_pairs( n, coeffs, roots );

    sort( n, roots );
}


} // namespace AMP
