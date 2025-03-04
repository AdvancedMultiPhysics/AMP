#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/arrayHelpers.h"

#include <algorithm>
#include <cmath>


namespace AMP::Geometry::GeometryHelpers {


/****************************************************************
 * Map the logical coordinates to a circle                       *
 ****************************************************************/
static inline std::array<double, 2> map_c2p( int method, double xc, double yc )
{
    // map xc > 0 and |yc| < xc = d to (xp,yp) in r=1 using the mapping by:
    //    Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids
    //       and Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //       SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    if ( fabs( yc ) > fabs( xc ) ) {
        auto [yp, xp] = map_c2p( method, yc, xc );
        return { xp, yp };
    }
    // map xc > 0 and |yc| < xc = d
    constexpr double invsqrt2 = 0.7071067811865475244;
    AMP_DEBUG_ASSERT( xc >= 0 && xc >= fabs( yc ) );
    if ( fabs( xc ) < 1e-12 )
        return { 0.0, 0.0 };
    if ( xc > 1.0 ) {
        double scale = xc;
        yc /= scale;
        double yp = invsqrt2 * yc;
        double xp = std::sqrt( 1 - yp * yp );
        return { scale * xp, scale * yp };
    }
    double xp = 0;
    double yp = 0;
    if ( method == 1 ) {
        yp = invsqrt2 * yc;
        xp = std::sqrt( xc * xc - yp * yp );
    } else if ( method == 2 ) {
        double center = invsqrt2 * xc - std::sqrt( 1.0 - 0.5 * xc * xc );
        yp            = invsqrt2 * yc;
        xp            = center + std::sqrt( 1.0 - yp * yp );
    } else if ( method == 3 ) {
        double D      = invsqrt2 * xc * ( 2 - xc );
        double center = D - std::sqrt( 1.0 - D * D );
        yp            = invsqrt2 * ( 2 - xc ) * yc;
        xp            = center + std::sqrt( 1.0 - yp * yp );
    } else {
        AMP_ERROR( "Invalid method" );
    }
    return { xp, yp };
}
static inline std::array<double, 2> map_p2c( int method, double xp, double yp )
{
    // Perform the inverse mapping as map_c2p
    if ( fabs( yp ) > fabs( xp ) ) {
        auto [yc, xc] = map_p2c( method, yp, xp );
        return { xc, yc };
    }
    // map |xp| > |yp| to logical
    constexpr double sqrt2    = 1.4142135623730950488;
    constexpr double invsqrt2 = 0.7071067811865475244;
    if ( fabs( xp ) < 1e-12 )
        return { 0.0, 0.0 };
    double R = std::sqrt( xp * xp + yp * yp );
    if ( R > 1.0 )
        return { R, sqrt2 * yp };
    double xc = 0;
    double yc = 0;
    if ( method == 1 ) {
        yc = yp * sqrt2;
        xc = std::sqrt( xp * xp + yp * yp );
    } else if ( method == 2 ) {
        yc     = yp * sqrt2;
        auto z = xp - std::sqrt( 1 - yp * yp );
        xc     = invsqrt2 * ( z + std::sqrt( 2 - z * z ) );
    } else if ( method == 3 ) {
        auto z = xp - std::sqrt( 1 - yp * yp );
        auto D = 0.5 * ( z + std::sqrt( 2 - z * z ) );
        xc     = 1.0 - std::sqrt( std::max( 1 - D * sqrt2, 0.0 ) );
        xc     = xc - ( sqrt2 * D - xc * ( 2 - xc ) ) / std::max( xc - 1, 1e-3 );
        yc     = yp * sqrt2 / ( 2 - xc );
    } else {
        AMP_ERROR( "Invalid method" );
    }
    return { xc, yc };
}
std::array<double, 2> map_logical_circle( double r, int method, double x, double y )
{
    // This maps from a logically rectangular 2D mesh to a circular mesh using the mapping by:
    //    Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids
    //       and Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //       SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    const double xc = 2 * x - 1; // Change domain to [-1,1]
    const double yc = 2 * y - 1; // Change domain to [-1,1]
    auto [xp, yp]   = map_c2p( method, fabs( xc ), fabs( yc ) );
    if ( xc < 0.0 )
        xp = -xp;
    if ( yc < 0.0 )
        yp = -yp;
    return { r * xp, r * yp };
}
std::array<double, 2> map_circle_logical( double r, int method, double x, double y )
{
    // Get the points in the unit circle
    double xp = x / r;
    double yp = y / r;
    // Perform the inverse mapping to [-1,1]
    auto [xc, yc] = map_p2c( method, fabs( xp ), fabs( yp ) );
    if ( xp < 0.0 )
        xc = -xc;
    if ( yp < 0.0 )
        yc = -yc;
    // Change domain to [0,1]
    return { 0.5 * ( xc + 1 ), 0.5 * ( yc + 1 ) };
}


/****************************************************************
 * Map between logical and physical coordinates in a polygon     *
 ****************************************************************/
static double inline get_m( int N )
{
    if ( N < 8 ) {
        const double m[] = { 0.577350269189626, 1.0,
                             1.376381920471173, 1.732050807568877,
                             2.076521396572336, 2.414213562373095 };
        return m[N - 3];
    }
    constexpr double pi = 3.14159265358979323;
    return tan( pi * ( N - 2 ) / ( 2 * N ) );
}
std::array<double, 2> map_poly_logical( int N, double R, double x, double y )
{
    constexpr double pi = 3.14159265358979323;
    // Map the coordinates from a polygon to a circle
    double a = atan( y / x );
    if ( x > 0 ) {
        a += pi;
    } else if ( x == 0 ) {
        a = pi / 2;
        if ( y > 0 )
            a = 1.5 * pi;
    }
    a            = fmod( a + 2.5 * pi, 2 * pi / N );
    a            = fabs( a - pi / N );
    double m     = get_m( N );
    double tan_a = tan( a );
    double r     = m * std::sqrt( 1.0 + tan_a * tan_a ) / ( m + tan_a );
    double x2    = x / r;
    double y2    = y / r;
    // Map the coordinates to a circle
    return map_circle_logical( R, 2, x2, y2 );
}
std::array<double, 2> map_logical_poly( int N, double R, double x, double y )
{
    constexpr double pi = 3.14159265358979323;
    // Map the coordinates to a circle
    auto [x2, y2] = map_logical_circle( R, 2, x, y );
    // Map the coordinates from a circle to the polygon
    double a = atan( y2 / x2 );
    if ( x2 > 0 ) {
        a += pi;
    } else if ( x2 == 0 ) {
        a = pi / 2;
        if ( y2 > 0 )
            a = 1.5 * pi;
    }
    a            = fmod( a + 2.5 * pi, 2 * pi / N );
    a            = fabs( a - pi / N );
    double m     = get_m( N );
    double tan_a = tan( a );
    double r     = m * std::sqrt( 1.0 + tan_a * tan_a ) / ( m + tan_a );
    x2 *= r;
    y2 *= r;
    return { x2, y2 };
}
std::vector<Point2D> get_poly_vertices( int N, double R )
{
    // Get the starting angle
    constexpr double pi = 3.14159265358979323;
    double theta        = 0.5 * pi - pi / N;
    double d_theta      = 2.0 * pi / N;
    // Create the vertices
    std::vector<Point2D> vertices( N );
    for ( int i = 0; i < N; i++ ) {
        vertices[i] = { R * cos( theta ), R * sin( theta ) };
        theta -= d_theta;
    }
    return vertices;
}


/****************************************************************
 * Helper function to map x,y,z logical coordinates in [0,1]     *
 * to x,y,z coordinates in a sphere with radius r                *
 ****************************************************************/
Point3D map_logical_sphere( double r, double x, double y, double z )
{
    constexpr double sqrt3 = 1.732050807568877; // std::sqrt(3)
    // This maps from a a logically rectangular 3D mesh to a sphere mesh using the mapping by:
    //    Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids
    //       and Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //       SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    double xc = 2 * x - 1; // Change domain to [-1,1]
    double yc = 2 * y - 1; // Change domain to [-1,1]
    double zc = 2 * z - 1; // Change domain to [-1,1]
    double d  = std::max( { fabs( xc ), fabs( yc ), fabs( zc ) } );
    double r2 = std::sqrt( xc * xc + yc * yc + zc * zc );
    r2        = std::max( r2, 1e-10 );
    double d2 = d * d;
    double c  = r * ( d * d2 / r2 + ( 1.0 - d2 ) / sqrt3 );
    double x2 = c * xc;
    double y2 = c * yc;
    double z2 = c * zc;
    return { x2, y2, z2 };
}
Point3D map_sphere_logical( double r, double x2, double y2, double z2 )
{
    constexpr double sqrt1_3 = 0.577350269189626; // 1/sqrt(3)
    // This maps from a physical coordinates to logical coordinates using the mapping by:
    //    Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids
    //       and Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //       SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    double d1  = std::max( { fabs( x2 ), fabs( y2 ), fabs( z2 ) } );
    double r21 = std::sqrt( x2 * x2 + y2 * y2 + z2 * z2 );
    r21        = std::max( r21, 1e-10 );
    // Solve c = a + b/c^2
    double a  = r * sqrt1_3;
    double a2 = a * a;
    double b  = r * d1 * d1 * ( d1 / r21 - sqrt1_3 );
    double c  = 0;
    if ( fabs( b ) < 1e-8 * a ) {
        c = a;
        c = a + b / ( c * c );
    } else {
        c = 1.5 * std::sqrt( 12 * a2 * a * b + 81 * b * b ) + a2 * a + 13.5 * b;
        c = cbrt( c );
        c = 0.33333333333333333 * ( c + a2 / c + a );
    }
    // Compute (x,y,z)
    double xc = x2 / c;
    double yc = y2 / c;
    double zc = z2 / c;
    double x  = 0.5 + 0.5 * xc;
    double y  = 0.5 + 0.5 * yc;
    double z  = 0.5 + 0.5 * zc;
    return { x, y, z };
}


/****************************************************************
 * Helper function to map x,y logical coordinates in [0,1]       *
 *    to x,y,z coordinates on the surface of a sphere            *
 * Dona Calhoun, Christiane Helzel, Randall LeVeque,             *
 *    "Logically Rectangular Grids and Finite Volume Methods for *
 *    PDEs in Circular and Spherical Domains",                   *
 *    SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)            *
 ****************************************************************/
Point3D map_logical_sphere_surface( int method, double R, double x, double y )
{
    if ( method == 1 ) {
        // Maps to a spherical surface, but boundary conditions and neighbors are wrong
        double x2 = 2 * x - 1;  // Change domain to [-1,1]
        double x3 = fabs( x2 ); // We need to make x go from 1:0:1
        // Map x,y to the unit circle
        auto point = map_logical_circle( 1.0, 3, x3, y );
        double xp  = point[0];
        double yp  = point[1];
        double zp  = std::sqrt( fabs( 1.0 - ( xp * xp + yp * yp ) ) );
        if ( zp < 1e-7 )
            zp = 0;
        else if ( x2 < 0 )
            zp = -zp; // negate z in lower hemisphere
        xp *= R;
        yp *= R;
        zp *= R;
        return { xp, yp, zp };
    } else if ( method == 2 ) {
        // Maps to half of spherical surface, but I don't know what else to do
        auto [xp, yp] = map_logical_circle( 1.0, 3, x, y );
        double zp     = std::sqrt( 1 - ( xp * xp + yp * yp ) );
        if ( zp != zp )
            zp = 0;
        return { R * xp, R * yp, R * zp };
    } else if ( method == 3 ) {
        // Spherical coordinates, not my favorite but at least it maps
        double pi    = 3.14159265358979323;
        double theta = pi * x;
        double phi   = 2 * pi * y;
        double xp    = R * sin( theta ) * cos( phi );
        double yp    = R * sin( theta ) * sin( phi );
        double zp    = R * cos( theta );
        return { xp, yp, zp };
    } else {
        AMP_ERROR( "Unknown method" );
    }
    return {};
}
constexpr static inline int sgn( double val ) { return ( 0 < val ) - ( val < 0 ); }
static_assert( sgn( -2.0 ) == -1 );
static_assert( sgn( 0.0 ) == 0 );
static_assert( sgn( 2.0 ) == 1 );
std::array<double, 2>
map_sphere_surface_logical( int method, double R, double x, double y, double z )
{
    if ( method == 1 ) {
        double xp  = x / R;
        double yp  = y / R;
        auto point = map_circle_logical( 1.0, 3, xp, yp );
        double x2  = z < 0 ? -point[0] : point[0];
        return { 0.5 + 0.5 * x2, point[1] };
    } else if ( method == 2 ) {
        double R2 = std::max( std::sqrt( x * x + y * y + z * z ), 1e-10 );
        double xp = x / R2;
        double yp = y / R2;
        return map_circle_logical( 1.0, 3, xp, yp );
    } else if ( method == 3 ) {
        // Spherical coordinates, not my favorite but at least it maps
        double pi    = 3.14159265358979323;
        double theta = acos( z / R );
        double phi   = sgn( y ) * acos( x / std::sqrt( x * x + y * y ) );
        return { theta / pi, phi / ( 2 * pi ) };
    } else {
        AMP_ERROR( "Unknown method" );
    }
    return {};
}


/****************************************************************
 * Helper function to map x,y,z logical coordinates in [0,1]     *
 * to x,y,z coordinates in a shell with r1 <= r <= r2            *
 ****************************************************************/
Point3D map_logical_shell( double r1, double r2, double x, double y, double z )
{
    // This maps from a a logically rectangular 3D mesh to a shell mesh using the mapping by:
    // Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids and
    //    Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //    SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    double Rz = r1 + z * ( r2 - r1 ); // radius based on z[0,1]
    return map_logical_sphere_surface( 1, Rz, x, y );
}
Point3D map_shell_logical( double r1, double r2, double x0, double y0, double z0 )
{
    // This maps from a a logically rectangular 3D mesh to a shell mesh using the mapping by:
    // Dona Calhoun, Christiane Helzel, Randall LeVeque, "Logically Rectangular Grids and
    //    Finite Volume Methods for PDEs in Circular and Spherical Domains",
    //    SIAM Review, Vol. 50, No. 4, pp. 723-752 (2008)
    double R = std::sqrt( x0 * x0 + y0 * y0 + z0 * z0 );
    auto xy  = map_sphere_surface_logical( 1, R, x0, y0, z0 );
    double z = ( R - r1 ) / ( r2 - r1 );
    return { std::get<0>( xy ), std::get<1>( xy ), z };
}


/****************************************************************
 * Compute the distance between the points                       *
 ****************************************************************/
static inline double dist2( const Point3D &x, const Point3D &y )
{
    return ( x[0] - y[0] ) * ( x[0] - y[0] ) + ( x[1] - y[1] ) * ( x[1] - y[1] ) +
           ( x[2] - y[2] ) * ( x[2] - y[2] );
}
double distance( const Point3D &x, const Point3D &y ) { return std::sqrt( dist2( x, y ) ); }


/****************************************************************
 * Compute the distance to the line segment                      *
 ****************************************************************/
double
distanceToLine( const Point2D &pos, const Point2D &ang, const Point2D &p1, const Point2D &p2 )
{
    Point2D v1 = pos - p1;
    Point2D v2 = p2 - p1;
    Point2D v3 = { -ang[1], ang[0] };
    double d23 = dot( v2, v3 );
    if ( fabs( d23 ) < 1e-12 )
        return std::numeric_limits<double>::infinity();
    double t1 = cross( v2, v1 ) / d23;
    double t2 = dot( v1, v3 ) / d23;
    if ( t1 >= -1e-12 && t2 >= -1e-10 && t2 <= 1.0 + 1e-10 )
        return t1;
    return std::numeric_limits<double>::infinity();
}
double
distanceToLine( const Point3D &pos, const Point3D &ang, const Point3D &p1, const Point3D &p2 )
{
    // Find the intersection in 2D
    double tmp[3] = { fabs( ang[0] ), fabs( ang[1] ), fabs( ang[2] ) };
    int i         = 0;
    Point2D pos2, ang2, p3, p4;
    if ( tmp[2] <= std::min( tmp[0], tmp[1] ) ) {
        i    = 2;
        pos2 = { pos[0], pos[1] };
        ang2 = { ang[0], ang[1] };
        p3   = { p1[0], p1[1] };
        p4   = { p2[0], p2[1] };
    } else if ( tmp[1] <= tmp[0] ) {
        i    = 1;
        pos2 = { pos[0], pos[2] };
        ang2 = { ang[0], ang[2] };
        p3   = { p1[0], p1[2] };
        p4   = { p2[0], p2[2] };
    } else {
        i    = 0;
        pos2 = { pos[1], pos[2] };
        ang2 = { ang[1], ang[2] };
        p3   = { p1[1], p1[2] };
        p4   = { p2[1], p2[2] };
    }
    Point2D v1 = pos2 - p3;
    Point2D v2 = p4 - p3;
    Point2D v3 = { -ang2[1], ang2[0] };
    double d23 = dot( v2, v3 );
    if ( fabs( d23 ) < 1e-12 )
        return std::numeric_limits<double>::infinity();
    double t1 = cross( v2, v1 ) / d23;
    double t2 = dot( v1, v3 ) / d23;
    if ( t1 < 0.0 || t2 < -1e-10 || t2 > 1.0 + 1e-10 )
        return std::numeric_limits<double>::infinity();
    // Check if the intersection point matches for the final coordinate
    double a = pos[i] + t1 * ang[i];
    double b = p1[i] + t2 * ( p2[i] - p1[i] );
    if ( fabs( a - b ) < 1e-10 )
        return t1;
    return std::numeric_limits<double>::infinity();
}


/****************************************************************
 * Compute the distance to the surface of a plane                *
 ****************************************************************/
double
distanceToPlane( const Point3D &n, const Point3D &p0, const Point3D &pos, const Point3D &ang )
{
    double d = dot( n, ang );
    if ( fabs( d ) > 1e-6 ) {
        auto v = p0 - pos;
        auto t = dot( v, n ) / d;
        if ( t >= 0 )
            return t;
    }
    return std::numeric_limits<double>::infinity();
}


/****************************************************************
 * Compute the distance to a box                                 *
 ****************************************************************/
template<std::size_t NDIM>
static constexpr bool inside( const std::array<double, NDIM> &p,
                              const std::array<double, NDIM> &lb,
                              const std::array<double, NDIM> &ub )
{
    // Check if the intersection of each surface is within the bounds of the box
    if constexpr ( NDIM == 1 ) {
        return ( p[0] >= lb[0] - 1e-12 ) && ( p[0] <= ub[0] + 1e-12 );
    } else if constexpr ( NDIM == 2 ) {
        return ( p[0] >= lb[0] - 1e-12 ) && ( p[0] <= ub[0] + 1e-12 ) &&
               ( p[1] >= lb[1] - 1e-12 ) && ( p[1] <= ub[1] + 1e-12 );
    } else if constexpr ( NDIM == 3 ) {
        return ( p[0] >= lb[0] - 1e-12 ) && ( p[0] <= ub[0] + 1e-12 ) &&
               ( p[1] >= lb[1] - 1e-12 ) && ( p[1] <= ub[1] + 1e-12 ) &&
               ( p[2] >= lb[2] - 1e-12 ) && ( p[2] <= ub[2] + 1e-12 );
    } else {
        bool in = true;
        for ( size_t d = 0; d < NDIM; d++ )
            in = in && ( p[d] >= lb[d] - 1e-12 ) && ( p[d] <= ub[d] + 1e-12 );
        return in;
    }
}
template<std::size_t NDIM>
double distanceToBox( const std::array<double, NDIM> &pos,
                      const std::array<double, NDIM> &ang,
                      const std::array<double, NDIM> &lb,
                      const std::array<double, NDIM> &ub )
{
    // Compute the distance to each surface and check if it is closer
    double d = std::numeric_limits<double>::infinity();
    for ( size_t i = 0; i < NDIM; i++ ) {
        double d1 = ( lb[i] - pos[i] ) / ang[i];
        double d2 = ( ub[i] - pos[i] ) / ang[i];
        if ( d1 >= 0 ) {
            auto p = pos + d1 * ang;
            if ( inside( p, lb, ub ) )
                d = std::min( d, d1 );
        }
        if ( d2 >= 0 ) {
            auto p = pos + d2 * ang;
            if ( inside( p, lb, ub ) )
                d = std::min( d, d2 );
        }
    }
    // Return the distance
    if ( d == std::numeric_limits<double>::infinity() )
        return d;
    if ( inside( pos, lb, ub ) )
        d = -d;
    return d;
}
template double distanceToBox<1>( const std::array<double, 1> &,
                                  const std::array<double, 1> &,
                                  const std::array<double, 1> &,
                                  const std::array<double, 1> & );
template double distanceToBox<2>( const std::array<double, 2> &,
                                  const std::array<double, 2> &,
                                  const std::array<double, 2> &,
                                  const std::array<double, 2> & );
template double distanceToBox<3>( const std::array<double, 3> &,
                                  const std::array<double, 3> &,
                                  const std::array<double, 3> &,
                                  const std::array<double, 3> & );
template double distanceToBox<4>( const std::array<double, 4> &,
                                  const std::array<double, 4> &,
                                  const std::array<double, 4> &,
                                  const std::array<double, 4> & );
template double distanceToBox<5>( const std::array<double, 5> &,
                                  const std::array<double, 5> &,
                                  const std::array<double, 5> &,
                                  const std::array<double, 5> & );


/****************************************************************
 * Compute the distance to a circle                              *
 ****************************************************************/
double distanceToCircle( double r, const Point2D &pos, const Point2D &ang )
{
    double r2   = pos[0] * pos[0] + pos[1] * pos[1];
    bool inside = r2 < r * r;
    double dx   = ang[0];
    double dy   = ang[1];
    double dr   = std::sqrt( dx * dx + dy * dy );
    double D    = pos[0] * ( pos[1] + ang[1] ) - ( pos[0] + ang[0] ) * pos[1];
    double t    = r * r * dr * dr - D * D;
    if ( t < 0 )
        return std::numeric_limits<double>::infinity();
    t = std::sqrt( t );
    double x1, x2, d1, d2;
    if ( ang[0] != 0.0 ) {
        double s = dy < 0 ? -1 : 1;
        x1       = ( D * dy + s * dx * t ) / ( dr * dr );
        x2       = ( D * dy - s * dx * t ) / ( dr * dr );
        d1       = ( x1 - pos[0] ) / ang[0];
        d2       = ( x2 - pos[0] ) / ang[0];
    } else {
        double s = dx < 0 ? -1 : 1;
        x1       = ( D * dx + s * dy * t ) / ( dr * dr );
        x2       = ( D * dx - s * dy * t ) / ( dr * dr );
        d1       = ( x1 - pos[1] ) / ang[1];
        d2       = ( x2 - pos[1] ) / ang[1];
    }
    if ( d1 < 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( d2 < 0 )
        d2 = std::numeric_limits<double>::infinity();
    double d = std::min( d1, d2 );
    return ( inside ? -1 : 1 ) * d;
}


/****************************************************************
 * Compute the distance to the surface of a cylinder             *
 ****************************************************************/
double distanceToCylinder( double r, double h, const Point3D &pos, const Point3D &ang )
{
    // Check if the point is inside the cylinder
    double r2   = pos[0] * pos[0] + pos[1] * pos[1];
    bool inside = std::abs( pos[2] ) <= 0.5 * h && r2 <= r * r;
    // First check the distance to the faces
    double d1 = ( 0.5 * h - pos[2] ) / ang[2];
    double d2 = ( -0.5 * h - pos[2] ) / ang[2];
    if ( d1 <= 0 || ang[2] == 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( d2 <= 0 || ang[2] == 0 )
        d2 = std::numeric_limits<double>::infinity();
    auto pos2 = pos + std::min( d1, d2 ) * ang;
    r2        = pos2[0] * pos2[0] + pos2[1] * pos2[1];
    if ( r2 > r * r ) {
        d1 = std::numeric_limits<double>::infinity();
        d2 = std::numeric_limits<double>::infinity();
    }
    // Compute the intersection of a line with the circle of the cylinder
    double d3 = std::abs( distanceToCircle( r, { pos[0], pos[1] }, { ang[0], ang[1] } ) );
    // Check that the z-point is within the cylinder
    double z = pos[2] + d3 * ang[2];
    if ( fabs( z ) > 0.5 * h )
        d3 = std::numeric_limits<double>::infinity();
    // Return the closest surface
    double d = std::min( { d1, d2, d3 } );
    if ( d < 1e100 )
        return ( inside ? -1 : 1 ) * d;
    return std::numeric_limits<double>::infinity();
}


/****************************************************************
 * Compute the distance to the surface of a cylinder             *
 ****************************************************************/
double
distanceToTube( double r_min, double r_max, double h, const Point3D &pos, const Point3D &ang )
{
    double r_min2 = r_min * r_min;
    double r_max2 = r_max * r_max;
    // Check if the point is inside the tube
    double r2   = pos[0] * pos[0] + pos[1] * pos[1];
    bool inside = std::abs( pos[2] ) <= 0.5 * h && r2 >= r_min2 && r2 <= r_max2;
    // Check the distance to the faces
    double d1 = ( 0.5 * h - pos[2] ) / ang[2];
    double d2 = ( -0.5 * h - pos[2] ) / ang[2];
    if ( d1 <= 0 || ang[2] == 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( d2 <= 0 || ang[2] == 0 )
        d2 = std::numeric_limits<double>::infinity();
    auto pos2 = pos + std::min( d1, d2 ) * ang;
    r2        = pos2[0] * pos2[0] + pos2[1] * pos2[1];
    if ( r2 < r_min2 || r2 > r_max2 ) {
        d1 = std::numeric_limits<double>::infinity();
        d2 = std::numeric_limits<double>::infinity();
    }
    // Check the intersection of a line with the circles of the tube
    auto checkCircle = [pos, ang, h]( double r ) {
        if ( ang[0] == 0 && ang[1] == 0 )
            return std::numeric_limits<double>::infinity();
        double d = std::abs( distanceToCircle( r, { pos[0], pos[1] }, { ang[0], ang[1] } ) );
        double z = pos[2] + d * ang[2];
        // We did not intersect with the surface, check for a second intersection
        if ( fabs( z ) > 0.5 * h ) {
            d += 1e-8;
            auto pos2 = pos + d * ang;
            double d2 = std::abs( distanceToCircle( r, { pos2[0], pos2[1] }, { ang[0], ang[1] } ) );
            d += d2;
            z = pos[2] + d * ang[2];
            if ( fabs( z ) > 0.5 * h )
                d = std::numeric_limits<double>::infinity();
        }
        return d;
    };
    double d3 = checkCircle( r_min );
    double d4 = checkCircle( r_max );
    // Return the closest surface
    double d = std::min( { d1, d2, d3, d4 } );
    if ( d < 1e100 )
        return ( inside ? -1 : 1 ) * d;
    return std::numeric_limits<double>::infinity();
}


/****************************************************************
 * Compute the distance to the surface of a sphere               *
 *    http://paulbourke.net/geometry/circlesphere                *
 ****************************************************************/
double distanceToSphere( double r, const Point3D &pos, const Point3D &ang )
{
    double a = ang[0] * ang[0] + ang[1] * ang[1] + ang[2] * ang[2];
    double b = 2 * ( ang[0] * pos[0] + ang[1] * pos[1] + ang[2] * pos[2] );
    double c = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2] - r * r;
    double t = b * b - 4 * a * c;
    if ( t < 0 )
        return std::numeric_limits<double>::infinity();
    t         = std::sqrt( t );
    double d1 = ( -b + t ) / ( 2 * a );
    double d2 = ( -b - t ) / ( 2 * a );
    if ( d1 < 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( d2 < 0 )
        d2 = std::numeric_limits<double>::infinity();
    double d    = std::min( d1, d2 );
    double r2   = pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2];
    bool inside = r2 < r * r;
    return ( inside ? -1 : 1 ) * d;
}


/****************************************************************
 * Compute the distance to the surface of a cone                *
 * http://lousodrome.net/blog/light/2017/01/03/intersection-of-a-ray-and-a-cone
 ****************************************************************/
double distanceToCone( const Point3D &V, double theta, const Point3D &pos, const Point3D &ang )
{
    double DV  = dot( ang, V );
    double CV  = dot( pos, V );
    double ct  = cos( theta );
    double ct2 = ct * ct;
    double a   = DV * DV - ct2;
    double b   = 2 * ( DV * CV - dot( ang, pos ) * ct2 );
    double c   = CV * CV - dot( pos, pos ) * ct2;
    double t   = b * b - 4 * a * c;
    if ( t < 0 )
        return std::numeric_limits<double>::infinity();
    t         = std::sqrt( t );
    double d1 = ( -b + t ) / ( 2 * a );
    double d2 = ( -b - t ) / ( 2 * a );
    if ( d1 < 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( d2 < 0 )
        d2 = std::numeric_limits<double>::infinity();
    if ( dot( pos + d1 * ang, V ) < 0 )
        d1 = std::numeric_limits<double>::infinity();
    if ( dot( pos + d2 * ang, V ) < 0 )
        d2 = std::numeric_limits<double>::infinity();
    return std::min( d1, d2 );
}
double
distanceToCircularFrustum( double rb, double rt, double h, const Point3D &pos, const Point3D &ang )
{
    AMP_INSIST( rb > rt && rt, "Invalid values for radii" );
    // Compute the apex of the underlying cone
    double h2    = h * rb / ( rb - rt );
    double theta = atan( rb / h2 );
    Point3D C    = { 0, 0, h2 };
    // Compute the intersection with the infinite cone
    Point3D V = { 0, 0, -1 };
    double d  = std::abs( distanceToCone( V, theta, pos - C, ang ) );
    auto pc   = pos + d * ang;
    if ( pc[2] < 0 || pc[2] > h )
        d = std::numeric_limits<double>::infinity();
    // Compute the intersection with the planes slicing the cone
    double d1 = -pos[2] / ang[2];
    double d2 = ( h - pos[2] ) / ang[2];
    auto p1   = pos + d1 * ang;
    auto p2   = pos + d2 * ang;
    double r1 = p1[0] * p1[0] + p1[1] * p1[1];
    double r2 = p2[0] * p2[0] + p2[1] * p2[1];
    if ( ( d1 < 0 ) || ( r1 > rb * rb ) )
        d1 = std::numeric_limits<double>::infinity();
    if ( ( d2 < 0 ) || ( r2 > rt * rt ) )
        d2 = std::numeric_limits<double>::infinity();
    // Keep the closest intersection
    d = std::min( { d, d1, d2 } );
    // Check if the point is inside the volume
    if ( d < 1e200 ) {
        double r2   = pos[0] * pos[0] + pos[1] * pos[1];
        double R    = rb * ( 1.0 - pos[2] ) + rt * pos[2];
        bool inside = pos[2] >= 0 && pos[2] <= h && r2 <= R * R;
        if ( inside )
            d = -d;
    }
    return d;
}


/****************************************************************
 * Get the Barycentric coordinates (u, v, w) for point p with   *
 *   respect to triangle (a, b, c)                              *
 ***************************************************************/
template<>
std::array<double, 3> barycentric<3, 3>( const std::array<std::array<double, 3>, 3> &x,
                                         const std::array<double, 3> &p )
{
    auto v0  = x[1] - x[0];
    auto v1  = x[2] - x[0];
    auto v2  = p - x[0];
    auto d00 = dot( v0, v0 );
    auto d01 = dot( v0, v1 );
    auto d11 = dot( v1, v1 );
    auto d20 = dot( v2, v0 );
    auto d21 = dot( v2, v1 );
    auto inv = 1.0 / ( d00 * d11 - d01 * d01 );
    auto v   = inv * ( d11 * d20 - d01 * d21 );
    auto w   = inv * ( d00 * d21 - d01 * d20 );
    auto u   = 1.0 - v - w;
    return { u, v, w };
}
template<>
std::array<double, 3> barycentric<3, 2>( const std::array<std::array<double, 2>, 3> &x,
                                         const std::array<double, 2> &xi )
{
    return DelaunayHelpers::computeBarycentric<2, double>( x.data(), xi );
}
template<>
std::array<double, 4> barycentric<4, 3>( const std::array<std::array<double, 3>, 4> &x,
                                         const std::array<double, 3> &xi )
{
    return DelaunayHelpers::computeBarycentric<3, double>( x.data(), xi );
}


/****************************************************************
 * Compute the distance to the surface of a triangle/tetrahedron *
 ****************************************************************/
static bool containsPoint( const std::array<Point2D, 3> &x, const Point2D &pos, double TOL )
{
    auto L = barycentric<3, 2>( x, pos );
    return ( L[0] >= -TOL ) && ( L[1] >= -TOL ) && ( L[2] >= -TOL );
}
static bool containsPoint( const std::array<Point3D, 3> &x, const Point3D &pos, double TOL )
{
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
    auto p0 = 1.0 / 3.0 * ( x[0] + x[1] + x[2] );
    auto L0 = barycentric<3, 3>( x, p0 );
    AMP_ASSERT( L0[0] > 0 && L0[1] > 0 && L0[2] > 0 );
#endif
    auto L = barycentric<3, 3>( x, pos );
    return ( L[0] >= -TOL ) && ( L[1] >= -TOL ) && ( L[2] >= -TOL );
}
static bool containsPoint( const std::array<Point3D, 4> &x, const Point3D &pos, double TOL )
{
    auto L = barycentric<4, 3>( x, pos );
    return ( L[0] >= -TOL ) && ( L[1] >= -TOL ) && ( L[2] >= -TOL ) && ( L[3] >= -TOL );
}
double distanceToTriangle( const std::array<Point2D, 3> &x, const Point2D &pos, const Point2D &dir )
{
    double d1 = distanceToLine( pos, dir, x[0], x[1] );
    double d2 = distanceToLine( pos, dir, x[1], x[2] );
    double d3 = distanceToLine( pos, dir, x[2], x[0] );
    double d  = std::min( { d1, d2, d3 } );
    if ( containsPoint( x, pos, 1e-8 ) ) {
        AMP_ASSERT( d < 1e200 );
        d = -d;
    }
    return d;
}
double distanceToTriangle( const std::array<Point3D, 3> &x, const Point3D &pos, const Point3D &dir )
{
    // Get the normal and a point on the plane containing the triangle
    auto n = AMP::Geometry::GeometryHelpers::normal( x[0], x[1], x[2] );
    // Check if the ray and plane are parallel
    double w = dot( n, dir );
    if ( fabs( w ) < 1e-8 ) {
        // The ray is parallel to the plane
        // Create the matrix to map between 2D and 3D coordinates
        Point3D v1  = normalize( x[1] - x[0] );
        Point3D v2  = cross( n, v1 );
        Point3D v3  = n;
        double M[9] = { v1[0], v1[1], v1[2], v2[0], v2[1], v2[2], v3[0], v3[1], v3[2] };
        double M_inv[9];
        DelaunayHelpers::inverse( 3, M, M_inv );
        // Check if the point is within the plane
        double z  = M_inv[2] * pos[0] + M_inv[5] * pos[1] + M_inv[8] * pos[2];
        double z0 = M_inv[2] * x[0][0] + M_inv[5] * x[0][1] + M_inv[8] * x[0][2];
        if ( fabs( z - z0 ) > 1e-8 )
            return std::numeric_limits<double>::infinity(); // ray is outside plane
        // Convert the triangle and ray coordinates to 2D and re-test
        auto convert = [&M_inv]( const Point3D &x ) {
            Point2D y = { M_inv[0] * x[0] + M_inv[3] * x[1] + M_inv[6] * x[2],
                          M_inv[1] * x[0] + M_inv[4] * x[1] + M_inv[7] * x[2] };
            return y;
        };
        std::array<Point2D, 3> x2 = { convert( x[0] ), convert( x[1] ), convert( x[2] ) };
        Point2D pos2              = convert( pos );
        Point2D dir2              = normalize( convert( pos + dir ) - pos2 );
        double d                  = distanceToTriangle( x2, pos2, dir2 );
        return d;
    } else {
        // The ray and plane are not parallel
        // Calculate the distance to the plane
        auto v = x[0] - pos;
        auto d = dot( v, n ) / w;
        if ( d < 0 )
            return std::numeric_limits<double>::infinity();
        // Check if the point of intersection is within the triangle
        auto p = pos + d * dir;
        if ( containsPoint( x, p, 1e-8 ) )
            return d;
        else
            return std::numeric_limits<double>::infinity();
    }
}
double
distanceToTetrahedron( const std::array<Point3D, 4> &x, const Point3D &pos, const Point3D &dir )
{
    double d1 = fabs( distanceToTriangle( { x[0], x[1], x[2] }, pos, dir ) );
    double d2 = fabs( distanceToTriangle( { x[1], x[2], x[3] }, pos, dir ) );
    double d3 = fabs( distanceToTriangle( { x[2], x[3], x[0] }, pos, dir ) );
    double d4 = fabs( distanceToTriangle( { x[3], x[0], x[1] }, pos, dir ) );
    double d  = std::min( { d1, d2, d3, d4 } );
    if ( d > 1e100 )
        return d;
    if ( containsPoint( x, pos, 1e-8 ) )
        d = -d;
    return d;
}


/****************************************************************
 * Compute the distance to a quadrilateral in 2D/3D              *
 ****************************************************************/
Point3D normalToQuadrilateral( const std::array<Point3D, 4> &x )
{
    // Get the centroid for the quadralateral and split into 4 triangles
    Point3D p = { 0.25 * ( x[0][0] + x[1][0] + x[2][0] + x[3][0] ),
                  0.25 * ( x[0][1] + x[1][1] + x[2][1] + x[3][1] ),
                  0.25 * ( x[0][2] + x[1][2] + x[2][2] + x[3][2] ) };
    auto n1   = normal( x[0], x[1], p );
    auto n2   = normal( x[1], x[2], p );
    auto n3   = normal( x[2], x[3], p );
    auto n4   = normal( x[3], x[0], p );
    return normalize( 0.25 * ( n1 + n2 + n3 + n4 ) );
}
double
distanceToQuadrilateral( const std::array<Point2D, 4> &x, const Point2D &pos, const Point2D &ang )
{
    // Get the centroid for the quadralateral and split into 4 triangles
    Point2D p = { 0.25 * ( x[0][0] + x[1][0] + x[2][0] + x[3][0] ),
                  0.25 * ( x[0][1] + x[1][1] + x[2][1] + x[3][1] ) };
    double d1 = distanceToTriangle( { x[0], x[1], p }, pos, ang );
    double d2 = distanceToTriangle( { x[1], x[2], p }, pos, ang );
    double d3 = distanceToTriangle( { x[2], x[3], p }, pos, ang );
    double d4 = distanceToTriangle( { x[3], x[0], p }, pos, ang );
    double d  = std::min( { d1, d2, d3, d4 } );
    if ( d <= 0 ) {
        // We are inside a triangle, check each edge
        d1 = std::abs( distanceToLine( pos, ang, x[0], x[1] ) );
        d2 = std::abs( distanceToLine( pos, ang, x[1], x[2] ) );
        d3 = std::abs( distanceToLine( pos, ang, x[2], x[3] ) );
        d4 = std::abs( distanceToLine( pos, ang, x[3], x[0] ) );
        return -std::min( { d1, d2, d3, d4 } );
    }
    return d;
}
double
distanceToQuadrilateral( const std::array<Point3D, 4> &x, const Point3D &pos, const Point3D &ang )
{
    // Get the centroid for the quadralateral and split into 4 triangles
    Point3D p = { 0.25 * ( x[0][0] + x[1][0] + x[2][0] + x[3][0] ),
                  0.25 * ( x[0][1] + x[1][1] + x[2][1] + x[3][1] ),
                  0.25 * ( x[0][2] + x[1][2] + x[2][2] + x[3][2] ) };
    double d1 = distanceToTriangle( { x[0], x[1], p }, pos, ang );
    double d2 = distanceToTriangle( { x[1], x[2], p }, pos, ang );
    double d3 = distanceToTriangle( { x[2], x[3], p }, pos, ang );
    double d4 = distanceToTriangle( { x[3], x[0], p }, pos, ang );
    double d  = std::min( { d1, d2, d3, d4 } );
    if ( d <= 0 ) {
        // We are inside a triangle, check each edge
        d1 = std::abs( distanceToLine( pos, ang, x[0], x[1] ) );
        d2 = std::abs( distanceToLine( pos, ang, x[1], x[2] ) );
        d3 = std::abs( distanceToLine( pos, ang, x[2], x[3] ) );
        d4 = std::abs( distanceToLine( pos, ang, x[3], x[0] ) );
        return -std::min( { d1, d2, d3, d4 } );
    }
    return d;
}


/****************************************************************
 * Compute the normal to a line/plane                            *
 ****************************************************************/
Point2D normal( const Point2D &a, const Point2D &b )
{
    Point2D n = { a[1] - b[1], b[0] - a[0] };
    return normalize( n );
}
Point3D normal( const Point3D &a, const Point3D &b, const Point3D &c )
{
    return normalize( cross( a - b, a - c ) );
}


/****************************************************************
 * Compute the nearest point to a line segment                   *
 ****************************************************************/
Point2D nearest( const Point2D &A, const Point2D &B, const Point2D &P )
{
    auto v = B - A;
    auto u = A - P;
    auto t = -dot( v, u ) / dot( v, v );
    t      = std::min( std::max( t, 0.0 ), 1.0 );
    return ( 1.0 - t ) * A + t * B;
}
Point3D nearest( const Point3D &A, const Point3D &B, const Point3D &P )
{
    auto v = B - A;
    auto u = A - P;
    auto t = -dot( v, u ) / dot( v, v );
    t      = std::min( std::max( t, 0.0 ), 1.0 );
    return ( 1.0 - t ) * A + t * B;
}


/****************************************************************
 * Compute the nearest point to a triangle                       *
 ****************************************************************/
Point3D nearest( const std::array<Point3D, 3> &v, const Point3D &p0 )
{
    constexpr double TOL = 1e-8;
    // Get the normal and a point on the plane containing the triangle
    auto n = AMP::Geometry::GeometryHelpers::normal( v[0], v[1], v[2] );
    // Find the closest point on the plane
    double d = dot( n, p0 - v[0] );
    auto p   = p0 - d * n;
    // Compute the barycentric coordinates
    auto L = barycentric<3, 3>( v, p );
    if ( L[0] > -TOL && L[1] > -TOL && L[2] > -TOL ) {
        // Point is inside triangle
        return p;
    } else if ( L[1] <= 0 && L[2] <= 0 ) {
        // Point is closest to the first vertex
        return v[0];
    } else if ( L[0] <= 0 && L[2] <= 0 ) {
        // Point is closest to the second vertex
        return v[1];
    } else if ( L[0] <= 0 && L[1] <= 0 ) {
        // Point is closest to the third vertex
        return v[2];
    } else if ( L[0] <= 0 ) {
        // Point is closest to line between second and third vertices
        return nearest( v[1], v[2], p0 );
    } else if ( L[1] <= 0 ) {
        // Point is closest to line between first and third vertices
        return nearest( v[0], v[2], p0 );
    } else {
        // Point is closest to line between first and second vertices
        return nearest( v[0], v[1], p0 );
    }
}


/****************************************************************
 * Compute the nearest point to a triangle                       *
 ****************************************************************/
Point3D nearest( const std::array<Point3D, 4> &v, const Point3D &p0 )
{
    // Compute the barycentric coordinates
    auto L = barycentric<4, 3>( v, p0 );
    // Restrict the coordinates
    L[0]       = std::max( std::min( L[0], 1.0 ), 0.0 );
    L[1]       = std::max( std::min( L[1], 1.0 ), 0.0 );
    L[2]       = std::max( std::min( L[2], 1.0 ), 0.0 );
    L[3]       = std::max( std::min( L[3], 1.0 ), 0.0 );
    double tmp = 1.0 / ( L[0] + L[1] + L[2] + L[3] );
    AMP_ASSERT( fabs( tmp - 1.0 ) < 1e-8 );
    // Return the nearest point
    return L[0] * v[0] + L[1] * v[1] + L[2] * v[2] + L[3] * v[3];
}


/****************************************************************
 * Recursively divide the triangle                               *
 ****************************************************************/
std::vector<AMP::Mesh::Point> subdivide( const std::array<AMP::Mesh::Point, 3> &v, double res )
{
    double d1  = ( v[0] - v[1] ).norm();
    double d2  = ( v[0] - v[2] ).norm();
    double d3  = ( v[1] - v[2] ).norm();
    double max = std::max( { d1, d2, d3 } );
    if ( max < res * res )
        return std::vector<AMP::Mesh::Point>();
    std::vector<AMP::Mesh::Point> s1, s2;
    if ( d1 == max ) {
        auto p = 0.5 * ( v[0] + v[1] );
        s1     = subdivide( { v[0], p, v[2] }, res );
        s2     = subdivide( { p, v[1], v[2] }, res );
        s1.push_back( p );
    } else if ( d2 == max ) {
        auto p = 0.5 * ( v[0] + v[2] );
        s1     = subdivide( { v[0], v[1], p }, res );
        s2     = subdivide( { p, v[1], v[2] }, res );
        s1.push_back( p );
    } else {
        auto p = 0.5 * ( v[1] + v[2] );
        s1     = subdivide( { v[0], v[1], p }, res );
        s2     = subdivide( { v[0], p, v[2] }, res );
        s1.push_back( p );
    }
    const double tol = 0.5 * res * res;
    for ( const auto &p : s2 ) {
        bool found = false;
        for ( const auto &p0 : s1 )
            found = found || ( p0 - p ).norm() < tol;
        if ( !found )
            s1.push_back( p );
    }
    return s1;
}


/****************************************************************
 * Sample points                                                 *
 ****************************************************************/
std::vector<Point3D> sampleLine( const std::array<Point3D, 2> &v, double d0, bool interior )
{
    // Sample a line
    double d = distance( v[0], v[1] );
    int N    = ceil( d / d0 );
    auto dx  = ( 1.0 / N ) * ( v[1] - v[0] );
    if ( interior ) {
        std::vector<std::array<double, 3>> p( N );
        for ( int i = 0; i < N; i++ )
            p[i] = v[0] + ( 0.5 + i ) * dx;
        return p;
    } else {
        std::vector<std::array<double, 3>> p( N + 1 );
        p[0] = v[0];
        for ( int i = 1; i < N; i++ )
            p[i] = v[0] + i * dx;
        p[N] = v[1];
        return p;
    }
}
static void
sampleTri( const std::array<Point3D, 3> &v, double d0, bool interior, std::vector<Point3D> &points )
{
    // Sample a triangle
    // Get the distance between each vertex and check if they are all within d0
    double d01 = distance( v[0], v[1] );
    double d12 = distance( v[1], v[2] );
    double d03 = distance( v[2], v[0] );
    if ( interior ) {
        // Get the centroid for the triangle and check if all points are within that distance
        constexpr double inv3 = 1.0 / 3.0;
        Point3D p             = inv3 * ( v[0] + v[1] + v[2] );
        double d1             = distance( p, v[0] );
        double d2             = distance( p, v[1] );
        double d3             = distance( p, v[2] );
        if ( std::max( { d1, d2, d3 } ) < d0 ) {
            points.push_back( p );
            return;
        }
    } else {
        // Check if all points are within d0 of a vertex
        if ( std::max( { d01, d12, d03 } ) < d0 ) {
            points.push_back( v[0] );
            points.push_back( v[1] );
            points.push_back( v[2] );
            return;
        }
    }
    // Split the triangle along the longest edge
    if ( d01 >= std::max( d12, d03 ) ) {
        sampleTri( { v[0], 0.5 * ( v[0] + v[1] ), v[2] }, d0, interior, points );
        sampleTri( { 0.5 * ( v[0] + v[1] ), v[1], v[2] }, d0, interior, points );
    } else if ( d12 >= d03 ) {
        sampleTri( { v[0], v[1], 0.5 * ( v[1] + v[2] ) }, d0, interior, points );
        sampleTri( { v[0], 0.5 * ( v[1] + v[2] ), v[2] }, d0, interior, points );
    } else {
        sampleTri( { v[0], v[1], 0.5 * ( v[0] + v[2] ) }, d0, interior, points );
        sampleTri( { v[1], v[2], 0.5 * ( v[0] + v[2] ) }, d0, interior, points );
    }
}
std::vector<Point3D> sampleTri( const std::array<Point3D, 3> &v, double d0, bool interior )
{
    std::vector<Point3D> p;
    sampleTri( v, d0, interior, p );
    if ( !interior )
        AMP::Utilities::unique( p );
    return p;
}
std::vector<Point3D> sampleQuad( const std::array<Point3D, 4> &v, double d0, bool interior )
{
    // Sample a quadrilateral
    // Note: we want to preserve the surface defined is distanceToQuadrilateral
    // Get the centroid for the quadrilateral
    Point3D p0 = { 0.25 * ( v[0][0] + v[1][0] + v[2][0] + v[3][0] ),
                   0.25 * ( v[0][1] + v[1][1] + v[2][1] + v[3][1] ),
                   0.25 * ( v[0][2] + v[1][2] + v[2][2] + v[3][2] ) };
    if ( interior ) {
        // Get the distance to each vertex and check if they are all within d0
        double d1 = distance( p0, v[0] );
        double d2 = distance( p0, v[1] );
        double d3 = distance( p0, v[2] );
        double d4 = distance( p0, v[3] );
        if ( std::max( { d1, d2, d3, d4 } ) < d0 )
            return { p0 };
    } else {
        // Get the distance between each vertex and check if they are all within d0
        double d1 = distance( v[0], v[1] );
        double d2 = distance( v[1], v[2] );
        double d3 = distance( v[2], v[3] );
        double d4 = distance( v[3], v[0] );
        if ( std::max( { d1, d2, d3, d4 } ) < d0 )
            return { v[0], v[1], v[2], v[3] };
    }
    // Split the quadrilateral into 4 triangles and sample the triangles
    std::vector<Point3D> p;
    sampleTri( { v[0], v[1], p0 }, d0, interior, p );
    sampleTri( { v[1], v[2], p0 }, d0, interior, p );
    sampleTri( { v[2], v[3], p0 }, d0, interior, p );
    sampleTri( { v[3], v[0], p0 }, d0, interior, p );
    // Remove duplicate points
    if ( !interior )
        AMP::Utilities::unique( p );
    return p;
}
std::vector<Point3D> sampleTet( const std::array<Point3D, 4> &, double, bool )
{
    AMP_ERROR( "sampleTet: Not finished" );
    return {};
}


} // namespace AMP::Geometry::GeometryHelpers
