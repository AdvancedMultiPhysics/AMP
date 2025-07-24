#include "AMP/geometry/shapes/CircleFrustum.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>
#include <vector>


namespace AMP::Geometry {


Point operator-( const Point &x, const std::array<double, 3> &y )
{
    return { x.x() - y[0], x.y() - y[1], x.z() - y[2] };
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
CircleFrustum::CircleFrustum( std::shared_ptr<const AMP::Database> db )
    : LogicalGeometry( 3, 3, { 2, 2, 2, 2, 0, 1 } )
{
    double r1 = db->getScalar<double>( "BaseRadius" );
    double r2 = db->getScalar<double>( "TopRadius" );
    double h  = db->getScalar<double>( "Height" );
    auto dir  = db->getString( "Dir" );
    int dir2  = 0;
    if ( dir == "-x" )
        dir2 = 0;
    else if ( dir == "+x" )
        dir2 = 1;
    else if ( dir == "-y" )
        dir2 = 2;
    else if ( dir == "+y" )
        dir2 = 3;
    else if ( dir == "-z" )
        dir2 = 4;
    else if ( dir == "+z" )
        dir2 = 5;
    else
        AMP_ERROR( "Invalid value for Dir" );
    initialize( dir2, { r1, r2 }, h );
}
CircleFrustum::CircleFrustum( const std::array<double, 2> &r, int dir, double height )
    : LogicalGeometry( 3, 3, { 2, 2, 2, 2, 0, 1 } )
{
    initialize( dir, r, height );
}
void CircleFrustum::initialize( int dir, const std::array<double, 2> &r, double h )
{
    d_dir    = dir;
    d_r      = r;
    d_h      = h;
    d_offset = { 0, 0, 0 };
    AMP_INSIST( d_r[0] > d_r[1] && d_r[1] > 0, "Invalid value for r" );
    AMP_INSIST( d_dir < 6, "Invalid value for dir" );
}


/********************************************************
 * Convert coordinates to/from reference frame           *
 ********************************************************/
Point CircleFrustum::convertToReference( const Point &p0 ) const
{
    // Get the physical point in a non-rotated frame
    Point p;
    if ( d_dir == 0 ) {
        p = { p0.y(), p0.z(), -p0.x() };
    } else if ( d_dir == 1 ) {
        p = { p0.y(), p0.z(), p0.x() };
    } else if ( d_dir == 2 ) {
        p = { p0.x(), p0.z(), -p0.y() };
    } else if ( d_dir == 3 ) {
        p = { p0.x(), p0.z(), p0.y() };
    } else if ( d_dir == 4 ) {
        p = { p0.x(), p0.y(), -p0.z() };
    } else {
        p = { p0.x(), p0.y(), p0.z() };
    }
    return p;
}
Point CircleFrustum::convertFromReference( const Point &p0 ) const
{
    // Rotate the coordinates
    Point p;
    if ( d_dir == 0 ) {
        p = { -p0.z(), p0.x(), p0.y() };
    } else if ( d_dir == 1 ) {
        p = { -p0.z(), p0.x(), p0.y() };
    } else if ( d_dir == 2 ) {
        p = { p0.x(), -p0.z(), p0.y() };
    } else if ( d_dir == 3 ) {
        p = { p0.x(), p0.z(), p0.y() };
    } else if ( d_dir == 4 ) {
        p = { p0.x(), p0.y(), -p0.z() };
    } else {
        p = { p0.x(), p0.y(), p0.z() };
    }
    return p;
}


/********************************************************
 * Compute the nearest point on the surface              *
 * Note: this is currently an approximation              *
 ********************************************************/
Point CircleFrustum::nearest( const Point &pos ) const
{
#if 1
    auto L = logical( pos );
    L.x()  = std::max( L.x(), 0.0 );
    L.x()  = std::min( L.x(), 1.0 );
    L.y()  = std::max( L.y(), 0.0 );
    L.y()  = std::min( L.y(), 1.0 );
    L.z()  = std::max( L.z(), 0.0 );
    L.z()  = std::min( L.z(), 1.0 );
#else
    // Get the logical coordinates
    auto L   = logical( pos );
    auto tmp = physical( L );
    if ( L.z() < 0.0 || L.z() > 1.0 ) {
        // Get the radius and z-position
        double z  = d_h * L.z();
        double rz = d_r[0] * ( 1.0 - z ) + d_r[1] * z;
        auto pl   = GeometryHelpers::map_logical_circle( rz, 2, L.x(), L.y() );
        double r  = std::sqrt( pl[0] * pl[0] + pl[1] * pl[1] );
        if ( L.z() < 0.0 && r < d_r[0] ) {
            // Closest point is on the bottom surface
            L.z() = 0.0;
        } else if ( L.z() > 1.0 && r < d_r[1] ) {
            // Closest point is on the top surface
            L.z() = 1.0;
        } else {
            // Closest point is on the outer surface
            // Determine the closest point on the line of the radius
            std::array<double, 2> A = { 0.5 * d_r[0], 0.0 };
            std::array<double, 2> B = { 0.5 * d_r[1], d_h };
            auto p                  = GeometryHelpers::nearest( A, B, { r, z } );
            L.z()                   = p[1] / d_h;
        }
    }
    L.x() = std::max( L.x(), 0.0 );
    L.x() = std::min( L.x(), 1.0 );
    L.y() = std::max( L.y(), 0.0 );
    L.y() = std::min( L.y(), 1.0 );
#endif
    return physical( L );
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double CircleFrustum::distance( const Point &pos, const Point &ang ) const
{
    // Remove the offset
    auto p0 = pos - d_offset;
    // Compute the intersection with the frustum
    using GeometryHelpers::distanceToCircularFrustum;
    using GeometryHelpers::Point3D;
    auto p = convertToReference( p0 );
    auto a = convertToReference( ang );
    return distanceToCircularFrustum( d_r[0], d_r[1], d_h, p, a );
}


/********************************************************
 * Check if the ray is inside the geometry               *
 ********************************************************/
bool CircleFrustum::inside( const Point &pos ) const
{
    constexpr double t1 = -1e-12;
    constexpr double t2 = 1.0 + 1e-12;
    // Get and check the logical coordinates
    auto p = logical( pos );
    return p.x() >= t1 && p.x() <= t2 && p.y() >= t1 && p.y() <= t2 && p.z() >= t1 && p.z() <= t2;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
int CircleFrustum::surface( const Point &pos ) const
{
    auto p   = logical( pos );
    double t = std::min(
        { std::abs( p.x() ), std::abs( 1 - p.x() ), std::abs( p.y() ), std::abs( 1 - p.y() ) } );
    if ( std::abs( p.z() ) < t ) {
        // We are at the - face
        return 0;
    } else if ( std::abs( 1 - p.z() ) < t ) {
        // We are at the + face
        return 1;
    } else {
        // We are at the cone face
        return 2;
    }
}
Point CircleFrustum::surfaceNorm( const Point &pos ) const
{
    int s    = surface( pos );
    Point v  = { 0, 0, 0 };
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2];
    if ( s == 0 ) {
        v[d_dir / 2] = d_dir % 2 == 0 ? 1 : -1;
    } else if ( s == 1 ) {
        v[d_dir / 2] = d_dir % 2 == 0 ? -1 : 1;
    } else {
        double h2    = d_h * d_r[0] / ( d_r[0] - d_r[1] );
        double theta = atan( d_r[0] / h2 );
        double sin_t = sin( theta );
        double cos_t = cos( theta );
        if ( d_dir == 0 ) {
            double r = std::sqrt( y * y + z * z );
            v        = { -sin_t, cos_t * y / r, cos_t * z / r };
        } else if ( d_dir == 1 ) {
            double r = std::sqrt( y * y + z * z );
            v        = { sin_t, cos_t * y / r, cos_t * z / r };
        } else if ( d_dir == 2 ) {
            double r = std::sqrt( x * x + z * z );
            v        = { cos_t * x / r, -sin_t, cos_t * z / r };
        } else if ( d_dir == 3 ) {
            double r = std::sqrt( x * x + z * z );
            v        = { cos_t * x / r, sin_t, cos_t * z / r };
        } else if ( d_dir == 4 ) {
            double r = std::sqrt( x * x + y * y );
            v        = { cos_t * x / r, cos_t * y / r, -sin_t };
        } else {
            double r = std::sqrt( x * x + y * y );
            v        = { cos_t * x / r, cos_t * y / r, sin_t };
        }
    }
    return v;
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point CircleFrustum::physical( const Point &pos ) const
{
    // Swap directions to preserve positive volume
    auto p0   = pos;
    bool swap = d_dir == 0 || d_dir == 1 || d_dir == 3 || d_dir == 4;
    if ( swap )
        p0.z() = 1.0 - p0.z();
    // Get the height and current radius
    double r = d_r[0] * ( 1.0 - p0.z() ) + d_r[1] * p0.z();
    double z = d_h * p0.z();
    // Map the x/y coordinates to a circle
    auto pl = GeometryHelpers::map_logical_circle( r, 2, p0.x(), p0.y() );
    Point p = { pl[0], pl[1], z };
    // Rotate the coordinates
    p = convertFromReference( p );
    // Add the offset
    p.x() += d_offset[0];
    p.y() += d_offset[1];
    p.z() += d_offset[2];
    return p;
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point CircleFrustum::logical( const Point &pos ) const
{
    // Remove the offset
    Point p0 = pos;
    p0.x() -= d_offset[0];
    p0.y() -= d_offset[1];
    p0.z() -= d_offset[2];
    // Get the physical point in a non-rotated frame
    p0 = convertToReference( p0 );
    // Get the logical height and current radius
    double z = p0.z() / d_h;
    double r = d_r[0] * ( 1.0 - z ) + d_r[1] * z;
    // Map the x/y coordinates from a circle
    auto pl = GeometryHelpers::map_circle_logical( r, 2, p0.x(), p0.y() );
    Point p = { pl[0], pl[1], z };
    // Swap directions to preserve positive volume
    bool swap = d_dir == 0 || d_dir == 1 || d_dir == 3 || d_dir == 4;
    if ( swap )
        p.z() = 1.0 - p.z();
    return p;
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point CircleFrustum::centroid() const
{
    double R11 = d_r[0] * d_r[0];
    double R12 = d_r[0] * d_r[1];
    double R22 = d_r[1] * d_r[1];
    double z   = 0.25 * d_h * ( R11 + 2 * R12 + 3 * R22 ) / ( R11 + R12 + R22 );
    auto dir2  = d_dir / 2;
    Point p    = { 0, 0, 0 };
    p[dir2]    = ( d_dir % 2 == 0 ? -1 : 1 ) * z;
    p.x() += d_offset[0];
    p.y() += d_offset[1];
    p.z() += d_offset[2];
    return p;
}
std::pair<Point, Point> CircleFrustum::box() const
{
    auto dir2 = d_dir / 2;
    Point lb  = { -d_r[0], -d_r[0], -d_r[0] };
    Point ub  = { d_r[0], d_r[0], d_r[0] };
    lb[dir2]  = d_dir % 2 == 0 ? -d_h : 0;
    ub[dir2]  = d_dir % 2 == 0 ? 0 : d_h;
    lb.x() += d_offset[0];
    lb.y() += d_offset[1];
    lb.z() += d_offset[2];
    ub.x() += d_offset[0];
    ub.y() += d_offset[1];
    ub.z() += d_offset[2];
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double CircleFrustum::volume() const
{
    constexpr double pi = 3.141592653589793;
    double R            = 0.5 * ( d_r[0] + d_r[1] );
    return d_h * pi * R * R;
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize CircleFrustum::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.size() == 2, "Size must be an array of length 2" );
    return { 2 * x[0], 2 * x[0], x[1] };
}
ArraySize CircleFrustum::getLogicalGridSize( const std::vector<double> &res ) const
{
    AMP_INSIST( res.size() == 3u, "Resolution must be an array of length 3" );
    double R = std::max( d_r[0], d_r[1] );
    return { (size_t) ( R / res[0] ), (size_t) ( R / res[1] ), (size_t) ( d_h / res[2] ) };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void CircleFrustum::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
    d_offset[2] += x[2];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> CircleFrustum::clone() const
{
    return std::make_unique<CircleFrustum>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool CircleFrustum::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const CircleFrustum *>( &rhs );
    if ( !geom )
        return false;
    return d_dir == geom->d_dir && d_r == geom->d_r && d_h == geom->d_h &&
           d_offset == geom->d_offset;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void CircleFrustum::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "dir", d_dir );
    AMP::IO::writeHDF5( fid, "h", d_h );
    AMP::IO::writeHDF5( fid, "r", d_r );
}
CircleFrustum::CircleFrustum( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "dir", d_dir );
    AMP::IO::readHDF5( fid, "h", d_h );
    AMP::IO::readHDF5( fid, "r", d_r );
}


} // namespace AMP::Geometry
