// rpoly based off Jenkins–Traub algorithm (TOMS 493)
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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
 * The real versions use the Compensated Horner Scheme to increase *
 * precision.                                                      *
 ******************************************************************/
static inline std::complex<double> eval( int n, const double *a, std::complex<double> x )
{
    std::complex<double> result = a[n];
    for ( int i = n - 1; i >= 0; --i )
        result = result * x + a[i];
    return result;
}
/*static inline double eval( int n, const double *a, double x )
{
    double r = a[n];
    double c = 0.0; // Correction term accumulator
    for (int i = n - 1; i >= 0; --i) {
        double p = std::fma(r, x, a[i]); // Main Horner step with FMA
        // Two-Product error: tracks error from (r * x)
        // Using FMA: error_prod = (r * x) - fl(r * x)
        double prod = r * x;
        double error_prod = std::fma(r, x, -prod);
        // Two-Sum error: tracks error from (prod + a[i])
        double error_sum = (prod - (p - a[i])) + (a[i] - (p - prod));
        // Update correction term: c = c * x + (error_prod + error_sum)
        c = std::fma(c, x, error_prod + error_sum);
        r = p;
    }
    return r + c; // Final corrected result
}*/
static inline double eval( int n, const double *a, long double x )
{
    long double result = a[n];
    for ( int i = n - 1; i >= 0; --i )
        result = std::fmal( result, x, a[i] );
    return result;
}
template<class T>
static inline T evalDeriv( int n, const double *a, T x )
{
    T result = 0.0;
    for ( int i = n; i > 0; --i )
        result = ( result + i * a[i] ) * x;
    result += a[1];
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
 * Find the root of the polynomial in the given interval           *
 ******************************************************************/
#define sign_equal( a, b ) ( ( a >= 0 ) == ( b >= 0 ) )
static inline double findRoot( int n, const double *p, double lb, double ub )
{
    if ( ub < lb )
        std::swap( lb, ub );
    double f_lb = eval( n, p, lb );
    double f_ub = eval( n, p, ub );
    if ( f_lb == 0.0 )
        return lb;
    if ( f_ub == 0.0 )
        return ub;
    if ( sign_equal( f_lb, f_ub ) )
        return 0;
    // Use a combination of Newton's method and the bisection method
    double x     = 0.5 * ( lb + ub );
    auto f       = eval( n, p, x );
    double range = ub - lb;
    int it       = 0;
    double tol   = 1e-14 * fabs( x );
    while ( range > tol ) {
        // Adjust the boundaries to maintain the bisection method
        if ( sign_equal( f, f_lb ) ) {
            lb   = x;
            f_lb = f;
        } else {
            ub   = x;
            f_ub = f;
        }
        range = ub - lb;
        if ( range < 0.0 )
            return lb;
        // Choose the new guess (use Newton's method and the adjust)
        auto df = evalDeriv( n, p, x );
        x -= f / df;
        x += ( it % 2 == 0 ? -0.01 : 0.01 ) * range;
        x = std::max( x, lb + 0.15 * range );
        x = std::min( x, ub - 0.15 * range );
        f = eval( n, p, x );
        ++it;
    }
    return 0.5 * ( lb + ub );
}


/******************************************************************
 * Remove a (real) root: P = p / ( r - x )                         *
 ******************************************************************/
static inline int removeRoot( int n, double *p, double r )
{
    long double pr = p[n--];
    long double s  = 1.0L / pr;
    for ( int i = n; i >= 0; i-- ) {
        long double p0 = p[i];
        p[i]           = s * pr;
        pr             = std::fmal( pr, r, p0 );
    }
    p[n] = 1;
    return n;
}


/******************************************************************
 * Perform Durand–Kerner iteration                                 *
 ******************************************************************/
static inline double iterate( int n, const double *a, std::complex<double> *roots )
{
    constexpr int max_iter = 200;
    constexpr double tol   = 1e-12;
    for ( int iter = 0; iter < max_iter; ++iter ) {
        bool converged = true;
        for ( int i = 0; i < n; ++i ) {
            std::complex<double> denom = 1.0;
            for ( int j = 0; j < n; ++j )
                if ( i != j )
                    denom *= ( roots[i] - roots[j] );
            auto delta = eval( n, a, roots[i] ) / denom;
            roots[i] -= delta;
            if ( std::abs( delta ) > tol )
                converged = false;
            // Check for real roots
            if ( ( ( iter & 0x7 ) == 0 ) && fabs( roots[i].imag() / roots[i].real() ) < 1e-4 ) {
                double r = findRoot( n, a, 0.99 * roots[i].real(), 1.01 * roots[i].real() );
                if ( r != 0 )
                    return r;
            }
        }
        if ( converged )
            break;
    }
    for ( int i = 0; i < n; ++i ) {
        if ( fabs( roots[i].imag() / roots[i].real() ) < 1e-4 ) {
            double r  = roots[i].real();
            double r2 = findRoot( n, a, 0.99 * r, 1.01 * r );
            if ( r2 != 0 )
                return r2;
            r2 = findRoot( n, a, 0.9 * r, 1.1 * r );
            if ( r != 0 )
                return r2;
            r2 = findRoot( n, a, 0.5 * r, 2 * r );
            if ( r2 != 0 )
                return r2;
        }
    }
    return 0;
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
    AMP_INSIST( n > 0, "Polynomial must not be empty" );
    AMP_INSIST( coeffs[n] != 0, "Leading coefficient cannot be zero" );

    // Handle roots == 0 directly
    if ( coeffs[0] == 0 ) {
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

    // Copy coefficients (scaling by a_n)
    constexpr int n_max = 120;
    AMP_INSIST( n < n_max, "rpoly is not stable for large number of roots" );
    double a[n_max];
    for ( int i = 0; i < n; i++ )
        a[i] = coeffs[i] / coeffs[n];
    a[n] = 1;

    // Initial guess
    constexpr double pi = 3.1415926535897932;
    for ( int i = 0; i < n; ++i ) {
        double t = 2 * pi * ( 0.5 + i ) / n;
        roots[i] = std::complex<double>( std::cos( t ), std::sin( t ) );
    }

    // Find the roots
    int N  = 0;
    int n2 = n;
    while ( true ) {
        double r = iterate( n2, a, &roots[N] );
        if ( r == 0 )
            break;
        roots[N++] = r;
        n2         = removeRoot( n2, a, r );
    }

    // Refine roots
    for ( int i = 0; i < N; i++ ) {
        double r = roots[i].real();
        r        = findRoot( n, coeffs, 0.999 * r, 1.001 * r );
        if ( r != 0 )
            roots[i] = r;
    }
    for ( int i = N; i < n; i++ )
        roots[i] = refineRoot( n2, a, roots[i] );

    // Clean / pair the roots
    enforce_conjugate_pairs( n2, a, &roots[N] );

    // Sort the roots
    sort( n, roots );
}


} // namespace AMP
