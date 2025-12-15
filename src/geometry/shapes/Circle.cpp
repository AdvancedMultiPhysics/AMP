#include "AMP/geometry/shapes/Circle.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
Circle::Circle( std::shared_ptr<const AMP::Database> db )
    : LogicalGeometry( 2, 2, { 1, 1, 1, 1, -3, -3 } )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    auto range  = db->getVector<double>( "Range" );
    AMP_INSIST( range.size() == 1u, "Range must be an array of length 1" );
    d_R = range[0];
}
Circle::Circle( double R ) : LogicalGeometry( 2, 2, { 1, 1, 1, 1, -3, -3 } ), d_R( R )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
Point Circle::nearest( const Point &pos ) const
{
    // Get the current point in the reference frame of the circle
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    // Calculate the nearest point
    double r = std::sqrt( x * x + y * y );
    if ( r <= d_R ) {
        return pos;
    } else {
        x *= d_R / r;
        y *= d_R / r;
        return { x + d_offset[0], y + d_offset[1] };
    }
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double Circle::distance( const Point &pos, const Point &ang ) const
{
    // Get the current point in the reference frame of the circle
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    // Compute the distance to the circle
    double d = GeometryHelpers::distanceToCircle( d_R, { x, y }, ang );
    return d;
}


/********************************************************
 * Check if the ray is inside the geometry               *
 ********************************************************/
bool Circle::inside( const Point &pos ) const
{
    double x   = pos[0] - d_offset[0];
    double y   = pos[1] - d_offset[1];
    double R21 = x * x + y * y;
    double R22 = d_R * d_R;
    return R21 <= ( 1.0 + 1e-12 ) * R22;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
Point Circle::surfaceNorm( const Point &pos ) const
{
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double n = std::sqrt( x * x + y * y );
    return { x / n, y / n, 0 };
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point Circle::physical( const Point &pos ) const
{
    auto tmp = GeometryHelpers::map_logical_circle( d_R, 2, pos.x(), pos.y() );
    double x = tmp[0] + d_offset[0];
    double y = tmp[1] + d_offset[1];
    return { x, y };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point Circle::logical( const Point &pos ) const
{
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    auto tmp = GeometryHelpers::map_circle_logical( d_R, 2, x, y );
    return Point( tmp[0], tmp[1] );
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point Circle::centroid() const { return { d_offset[0], d_offset[1] }; }
std::pair<Point, Point> Circle::box() const
{
    Point lb = { d_offset[0] - d_R, d_offset[1] - d_R };
    Point ub = { d_offset[0] + d_R, d_offset[1] + d_R };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double Circle::volume() const
{
    constexpr double pi = 3.141592653589793;
    return pi * d_R * d_R;
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize Circle::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.ndim() == 1u, "Size must be an array of length 1" );
    return { 2 * x[0], 2 * x[0] };
}
ArraySize Circle::getLogicalGridSize( const std::vector<double> &res ) const
{
    AMP_INSIST( res.size() == 2u, "Resolution must be an array of length 2" );
    return { (size_t) ( d_R / res[0] ), (size_t) ( d_R / res[1] ) };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void Circle::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> Circle::clone() const
{
    return std::make_unique<Circle>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool Circle::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const Circle *>( &rhs );
    if ( !geom )
        return false;
    return d_R == geom->d_R && d_offset == geom->d_offset;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void Circle::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "R", d_R );
}
Circle::Circle( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "R", d_R );
}


} // namespace AMP::Geometry
