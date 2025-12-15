#include "AMP/geometry/shapes/CircleSurface.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
CircleSurface::CircleSurface( std::shared_ptr<const AMP::Database> db )
    : LogicalGeometry( 2, 1, { -1, -1, -3, -3, -3, -3 } )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    auto range  = db->getVector<double>( "Range" );
    AMP_INSIST( range.size() == 1u, "Range must be an array of length 1" );
    d_R = range[0];
}
CircleSurface::CircleSurface( double R )
    : LogicalGeometry( 2, 1, { -1, -1, -3, -3, -3, -3 } ), d_R( R )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
Point CircleSurface::nearest( const Point &pos ) const
{
    // Get the current point in the reference frame of the CircleSurface
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    // Calculate the nearest point
    double r = std::sqrt( x * x + y * y );
    if ( r == 0 ) {
        return { d_R + d_offset[0], 0 };
    } else {
        x *= d_R / r;
        y *= d_R / r;
        return { x + d_offset[0], y + d_offset[1] };
    }
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double CircleSurface::distance( const Point &pos, const Point &ang ) const
{
    // Get the current point in the reference frame of the circle
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    // Compute the distance to the circle
    double d = GeometryHelpers::distanceToCircle( d_R, { x, y }, ang );
    return std::abs( d );
}


/********************************************************
 * Check if the ray is inside the geometry               *
 ********************************************************/
bool CircleSurface::inside( const Point &pos ) const
{
    double x = pos[0] - d_offset[0];
    double y = pos[1] - d_offset[1];
    double r = std::sqrt( x * x + y * y );
    return std::abs( r - d_R ) < 1e-6 * d_R;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
Point CircleSurface::surfaceNorm( const Point &pos ) const
{
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double n = std::sqrt( x * x + y * y );
    return { x / n, y / n, 0 };
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point CircleSurface::physical( const Point &pos ) const
{
    constexpr double pi = 3.141592653589793;
    double phi          = 2 * pi * ( pos.x() + d_offset[0] );
    return { d_R * cos( phi ), d_R * sin( phi ) };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point CircleSurface::logical( const Point &pos ) const
{
    constexpr double pi = 3.141592653589793;
    double x            = pos.x() - d_offset[0];
    double y            = pos.y() - d_offset[1];
    double phi          = 0;
    if ( x == 0 && y == 0 ) {
        phi = 0;
    } else if ( x == 0 ) {
        phi = y >= 0 ? ( 0.5 * pi ) : ( -0.5 * pi );
    } else if ( x > 0 ) {
        phi = atan( y / x );
    } else if ( y >= 0 ) {
        phi = atan( y / x ) + pi;
    } else {
        phi = atan( y / x ) - pi;
    }
    return { phi / ( 2 * pi ) };
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point CircleSurface::centroid() const { return { d_offset[0], d_offset[1] }; }
std::pair<Point, Point> CircleSurface::box() const
{
    Point lb = { d_offset[0] - d_R, d_offset[1] - d_R };
    Point ub = { d_offset[0] + d_R, d_offset[1] + d_R };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double CircleSurface::volume() const
{
    constexpr double pi = 3.141592653589793;
    return 2 * pi * d_R;
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize CircleSurface::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.ndim() == 1u, "Size must be an array of length 1" );
    return { x[0] };
}
ArraySize CircleSurface::getLogicalGridSize( const std::vector<double> &res ) const
{
    constexpr double pi = 3.141592653589793;
    if ( res.size() == 1 ) {
        return { (size_t) ( 2 * pi * d_R / res[0] ) };
    } else if ( res.size() == 2 ) {
        return { (size_t) ( 2 * pi * d_R / std::min( res[0], res[1] ) ) };
    } else {
        AMP_ERROR( "Resolution must be an array of length 2" );
    }
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void CircleSurface::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> CircleSurface::clone() const
{
    return std::make_unique<CircleSurface>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool CircleSurface::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const CircleSurface *>( &rhs );
    if ( !geom )
        return false;
    return d_R == geom->d_R && d_offset == geom->d_offset;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void CircleSurface::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "R", d_R );
}
CircleSurface::CircleSurface( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "R", d_R );
}


} // namespace AMP::Geometry
