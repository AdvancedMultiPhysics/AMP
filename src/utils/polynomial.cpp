#include "AMP/utils/polynomial.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Utilities.h"

#if defined( AMP_USE_LAPACK ) && defined( AMP_USE_LAPACK_WRAPPERS )
    #define USE_LAPACK
    #include "LapackWrappers.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>


namespace AMP {


#define sign_equal( a, b ) ( ( a >= 0 ) == ( b >= 0 ) )


/******************************************************************
 * Constructors                                                    *
 ******************************************************************/
Polynomial::Polynomial() { d_p.resize( 1, 0.0 ); }
Polynomial::Polynomial( double a0 ) { d_p.resize( 1, a0 ); }
Polynomial::Polynomial( int n, const double *a )
{
    d_p.resize( n + 1 );
    for ( int i = 0; i <= n; i++ )
        d_p[i] = a[i];
    // Decrease the order of the polynomial if the last coefficient is zero
    while ( d_p.size() > 1 && d_p.back() == 0.0 )
        d_p.resize( d_p.size() - 1 );
}
Polynomial::Polynomial( std::vector<double> a ) : d_p( std::move( a ) )
{
    if ( d_p.empty() )
        d_p.resize( 1, 0 );
    // Decrease the order of the polynomial if the last coefficient is zero
    while ( d_p.size() > 1 && d_p.back() == 0.0 )
        d_p.resize( d_p.size() - 1 );
}


/******************************************************************
 * Create polynomial coefficients from roots using a product-tree  *
 ******************************************************************/
using TYPE = double;
static int poly_mul( int na, const TYPE *a, int nb, const TYPE *b, TYPE *c )
{
    int nc = na + nb - 1;
    for ( int i = 0; i < nc; i++ )
        c[i] = 0;
    for ( int i = 0; i < na; ++i ) {
        for ( int j = 0; j < nb; ++j ) {
            c[i + j] = std::fma( a[i], b[j], c[i + j] );
        }
    }
    return nc;
}
Polynomial Polynomial::createFromRoots( int n, const double *roots )
{
    if ( n == 0 )
        return {};
    if ( n == 1 )
        return Polynomial( { -roots[0], 1.0 } );

    // Allocate temporary memory for processing
    auto polys = new TYPE[5 * n + 1];
    AMP_ASSERT( polys );

    // Copy + sort by magnitude (important for stability)
    auto r = &polys[3 * n];
    for ( int i = 0; i < n; i++ )
        r[i] = roots[i];
    std::sort( r, r + n, []( TYPE a, TYPE b ) { return std::abs( a ) < std::abs( b ); } );

    // Start with linear factors (x - r)
    int N  = 0;
    int np = 3;
    for ( int i = 0; i < n; i += 2 ) {
        auto p = &polys[3 * N];
        if ( i + 1 < n ) {
            p[0] = r[i] * r[i + 1];
            p[1] = -r[i] - r[i + 1];
            p[2] = 1.0L;
        } else {
            p[0] = -r[i];
            p[1] = 1.0L;
            p[2] = 0.0L;
        }
        N++;
    }

    // Build the tree
    auto c = &polys[3 * n];
    while ( N > 1 ) {
        int nc = 2 * np - 1;
        int N2 = 0;
        for ( int i = 0; i < N; i += 2 ) {
            auto a = &polys[i * np];
            auto b = &polys[( i + 1 ) * np];
            if ( i + 1 < N ) {
                poly_mul( np, a, np, b, c );
            } else {
                for ( int i = 0; i < np; i++ )
                    c[i] = a[i];
                for ( int i = np; i < nc; i++ )
                    c[i] = 0;
            }
            a = &polys[N2 * nc];
            for ( int i = 0; i < nc; i++ )
                a[i] = c[i];
            N2++;
        }
        np = nc;
        N  = N2;
    }

    // Convert back to double
    std::vector<double> result( n + 1 );
    for ( size_t i = 0; i < result.size(); ++i )
        result[i] = static_cast<double>( polys[i] );
    delete[] polys;

    return Polynomial( std::move( result ) );
}
Polynomial Polynomial::createFromRoots( const std::vector<double> &roots )
{
    return createFromRoots( roots.size(), roots.data() );
}


/******************************************************************
 * Basic functions                                                 *
 ******************************************************************/
void Polynomial::setCoeff( int N, double a )
{
    // Increase the storage space if necessary
    if ( N >= (int) d_p.size() )
        d_p.resize( N + 1, 0 );
    // Set the value
    d_p[N] = a;
    // Decrease the order of the polynomial if the last coefficient is 0
    while ( d_p.size() > 1 && d_p.back() == 0.0 )
        d_p.resize( d_p.size() - 1 );
}
std::string Polynomial::getPolynomial() const
{
    std::stringstream ss;
    ss << d_p[0];
    for ( size_t i = 1; i < d_p.size(); i++ ) {
        if ( d_p[i] >= 0.0 )
            ss << " + " << d_p[i] << "*x^" << i;
        else
            ss << " - " << -d_p[i] << "*x^" << i;
    }
    return ss.str();
}
void Polynomial::print() const { AMP::pout << "p = " << getPolynomial().c_str() << std::endl; }


/******************************************************************
 * Polynomial addition, subtraction, and multiplcation             *
 ******************************************************************/
Polynomial operator+( const Polynomial &a, const Polynomial &b )
{
    std::vector<double> p( std::max( a.d_p.size(), b.d_p.size() ), 0 );
    for ( size_t i = 0; i < a.d_p.size(); i++ )
        p[i] += a.d_p[i];
    for ( size_t i = 0; i < b.d_p.size(); i++ )
        p[i] += b.d_p[i];
    return Polynomial( std::move( p ) );
}
Polynomial operator-( const Polynomial &a, const Polynomial &b )
{
    std::vector<double> p( std::max( a.d_p.size(), b.d_p.size() ), 0 );
    for ( size_t i = 0; i < a.d_p.size(); i++ )
        p[i] += a.d_p[i];
    for ( size_t i = 0; i < b.d_p.size(); i++ )
        p[i] -= b.d_p[i];
    return Polynomial( std::move( p ) );
}
Polynomial operator*( const Polynomial &a, const Polynomial &b )
{
    std::vector<double> c( a.order() + b.order() + 1, 0.0 );
    for ( int i = 0; i <= a.order(); ++i ) {
        for ( int j = 0; j <= b.order(); ++j ) {
            c[i + j] = std::fma( a[i], b[j], c[i + j] );
        }
    }
    return Polynomial( std::move( c ) );
}
std::tuple<Polynomial, Polynomial> operator/( const Polynomial &a, const Polynomial &b )
{
    if ( b.d_p.size() > a.d_p.size() )
        return std::make_tuple( Polynomial(), a );
    int bn = b.order();
    int n  = a.d_p.size() - b.d_p.size();
    std::vector<double> pq( a.d_p.size(), 0 );
    std::vector<double> pr( a.d_p );
    for ( int i = n; i >= 0; i-- ) {
        pq[i] = pr[i + bn] / b.d_p[bn];
        for ( int j = 0; j <= bn; j++ )
            pr[i + j] -= pq[i] * b.d_p[j];
    }
    Polynomial Q( std::move( pq ) );
    Polynomial R( std::move( pr ) );
    return std::make_tuple( Q, R );
}


/******************************************************************
 * Compute the derivative of the polynomial                        *
 ******************************************************************/
Polynomial Polynomial::derivative() const
{
    std::vector<double> d( d_p.size() - 1 );
    for ( size_t i = 0; i < d.size(); i++ )
        d[i] = ( i + 1 ) * d_p[i + 1];
    return Polynomial( std::move( d ) );
}


/******************************************************************
 * Evalulate the polynomial at x using Horner's scheme             *
 ******************************************************************/
static std::array<double, 2> eval2( const std::vector<double> &p, double x ) noexcept
{
    long double x2 = x;
    long double f  = p.back();
    for ( int i = p.size() - 2; i >= 0; i-- )
        f = static_cast<long double>( p[i] ) + f * x2;
    long double df = 0;
    for ( int i = p.size() - 2; i >= 0; i-- )
        df = static_cast<long double>( ( i + 1 ) * p[i + 1] ) + df * x2;
    return { (double) f, (double) df };
}
double Polynomial::eval( double x ) const noexcept
{
    long double f  = d_p.back();
    long double x2 = x;
    for ( int i = d_p.size() - 2; i >= 0; i-- )
        f = static_cast<long double>( d_p[i] ) + f * x2;
    return f;
}
std::complex<double> Polynomial::eval( const std::complex<double> &x ) const noexcept
{
    std::complex<long double> f = d_p.back();
    std::complex<long double> x2( x.real(), x.imag() );
    for ( int i = d_p.size() - 2; i >= 0; i-- )
        f = static_cast<long double>( d_p[i] ) + f * x2;
    return std::complex<double>( f.real(), f.imag() );
}


/******************************************************************
 * Evalulate the derivative of the polynomial at x                 *
 ******************************************************************/
double Polynomial::evalDerivative( double x ) const
{
    double tmp = 0.0;
    for ( int i = d_p.size() - 1; i > 1; i-- )
        tmp = ( tmp + i * d_p[i] ) * x;
    tmp += d_p[1];
    return tmp;
}
std::complex<double> Polynomial::evalDerivative( const std::complex<double> &x ) const
{
    std::complex<double> tmp = 0.0;
    for ( int i = d_p.size() - 1; i > 1; i-- )
        tmp = ( tmp + i * d_p[i] ) * x;
    tmp += d_p[1];
    return tmp;
}


/******************************************************************
 * Find the root of the polynomial in the given interval           *
 ******************************************************************/
double Polynomial::rootInterval( double lb, double ub, double tol ) const
{
    // Check that the bounds are sorted
    if ( ub < lb ) {
        double tmp = lb;
        lb         = ub;
        ub         = tmp;
    }
    // Check that the bounds are distinct
    if ( lb == ub )
        return lb;
    [[maybe_unused]] double f_lb = eval( lb );
    [[maybe_unused]] double f_ub = eval( ub );
    if ( f_lb == 0.0 )
        return lb;
    if ( f_ub == 0.0 )
        return ub;
    if ( sign_equal( f_lb, f_ub ) )
        throw std::runtime_error( "Bounds must have different signs" );
    // Use a combination of Newton's method and the bisection method
    double x     = 0.5 * ( lb + ub );
    auto f_df    = eval2( d_p, x );
    double range = ub - lb;
    int it       = 0;
    while ( range > tol ) {
        // Adjust the boundaries to maintain the bisection method
        if ( sign_equal( f_df[0], f_lb ) ) {
            lb   = x;
            f_lb = f_df[0];
        } else {
            ub   = x;
            f_ub = f_df[0];
        }
        range = ub - lb;
        if ( range < 0.0 )
            return lb;
        // Choose the new guess (use Newton's method and the adjust)
        x -= f_df[0] / f_df[1];
        x += ( it % 2 == 0 ? -0.01 : 0.01 ) * range;
        x    = std::max( x, lb + 0.15 * range );
        x    = std::min( x, ub - 0.15 * range );
        f_df = eval2( d_p, x );
        ++it;
    }
    return 0.5 * ( lb + ub );
}


/******************************************************************
 * Find all the root of the polynomial                             *
 ******************************************************************/
void rpoly( int, const double *, std::complex<double> * );
std::vector<std::complex<double>> Polynomial::roots() const
{
    if ( d_p.size() == 1 )
        return {};
    // Find the roots based on the rpoly (Jenkins-Traub) algorithm
    int n = d_p.size() - 1;
    std::vector<std::complex<double>> roots( n, 0 );
    rpoly( n, d_p.data(), roots.data() );
    return roots;
}


/******************************************************************
 * Fit a function                                                  *
 ******************************************************************/
Polynomial
Polynomial::fit( int n, std::function<double( double )> fun, double lb, double ub, int N )
{
    // Create the points to evaluate
    std::vector<double> x( N ), y( N );
    double dx = ( ub - lb ) / ( N - 1 );
    for ( int i = 0; i < N; i++ ) {
        x[i] = lb + i * dx;
        y[i] = fun( x[i] );
    }
    // Create the polynomial
    return fit( n, x, y );
}

#ifdef USE_LAPACK
Polynomial Polynomial::fit( int n, const std::vector<double> &x, const std::vector<double> &y0 )
{
    // Create the Vandermonde matrix
    AMP::Array<double> X( x.size(), n + 1 );
    for ( size_t i = 0; i < x.size(); i++ ) {
        X( i, 0 ) = 1.0;
        for ( int j = 1; j <= n; j++ )
            X( i, j ) = X( i, j - 1 ) * x[i];
    }
    // Create y
    AMP::Array<double> y( y0.size() );
    y.copy( y0.data() );
    // Solve a = ( X^T X )^-1 X^T y
    auto Xt   = X.reverseDim();
    auto M    = Xt * X;
    auto b    = Xt * y;
    auto IPIV = new int[n + 1];
    int error;
    Lapack<double>::gesv( n + 1, 1, M.data(), n + 1, IPIV, b.data(), n + 1, error );
    AMP_ASSERT( error == 0 );
    delete[] IPIV;
    // Return the polynomial
    return Polynomial( n, b.data() );
}
#else
Polynomial Polynomial::fit( int, const std::vector<double> &, const std::vector<double> & )
{
    AMP_ERROR( "fitting a polynomial requires Lapack" );
}
#endif
double Polynomial::error( std::function<double( double )> fun, double lb, double ub ) const
{
    size_t N   = 2 * d_p.size() + 7;
    double err = 0.0;
    double dx  = ( ub - lb ) / ( N - 1 );
    for ( size_t i = 0; i < N; i++ ) {
        double x = lb + i * dx;
        double f = fun( x );
        double y = eval( x );
        err      = std::max( err, fabs( y - f ) );
    }
    return err;
}


} // namespace AMP
