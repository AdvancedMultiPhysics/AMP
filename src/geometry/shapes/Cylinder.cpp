#include "AMP/geometry/shapes/Cylinder.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>

namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
Cylinder::Cylinder() : LogicalGeometry( 3, 3, { 4, 4, 4, 4, 2, 1 } )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    d_offset[2] = 0;
}
Cylinder::Cylinder( std::shared_ptr<const AMP::Database> db ) : Cylinder()
{
    d_method   = db->getWithDefault<int>( "Method", 2 );
    auto range = db->getVector<double>( "Range" );
    if ( range.size() == 3u ) {
        d_r     = range[0];
        d_z_min = range[1];
        d_z_max = range[2];
    } else if ( range.size() == 5u ) {
        d_r          = range[0];
        d_z_min      = range[1];
        d_z_max      = range[2];
        d_chamfer[0] = range[3];
        d_chamfer[1] = range[4];
    } else {
        AMP_INSIST( range.size() == 3u, "Range must be an array of length 3" );
    }
}
Cylinder::Cylinder( double r, double z_min, double z_max ) : Cylinder()
{
    d_r      = r;
    d_z_min  = z_min;
    d_z_max  = z_max;
    d_method = 2;
}


/********************************************************
 * Get cylinder radius at current height                 *
 ********************************************************/
double Cylinder::getR( double z ) const
{
    double z2 = std::min( fabs( z - d_z_min ), fabs( z - d_z_max ) );
    if ( z2 < d_chamfer[0] )
        return d_r - d_chamfer[1] * ( 1 - z2 / d_chamfer[0] );
    return d_r;
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
Point Cylinder::nearest( const Point &pos ) const
{
    // Get the current point in the reference frame of the circle
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2];
    // Calculate the nearest point
    z        = std::min( z, d_z_max );
    z        = std::max( z, d_z_min );
    double r = std::sqrt( x * x + y * y );
    double R = getR( z );
    if ( r > R ) {
        x *= R / r;
        y *= R / r;
    }
    return { x + d_offset[0], y + d_offset[1], z + d_offset[2] };
}


/********************************************************
 * Compute the distance to the object                    *
 * http://mathworld.wolfram.com/Circle-LineIntersection.html
 ********************************************************/
double Cylinder::distance( const Point &pos, const Point &ang ) const
{
    // Get the current point in the reference frame of the cylinder
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2] - 0.5 * ( d_z_min + d_z_max );
    // Compute the distance to the cylinder
    double h = d_z_max - d_z_min - 2 * d_chamfer[0];
    double d = GeometryHelpers::distanceToCylinder( d_r, h, { x, y, z }, ang );
    if ( d_chamfer[0] > 0 ) {
        double ax = ang[0];
        double ay = ang[1];
        double az = ang[2];
        double z1 = ( d_z_min + d_chamfer[0] ) - z;
        double z2 = z - ( d_z_max - d_chamfer[0] );
        double d1 = GeometryHelpers::distanceToCircularFrustum(
            d_r, d_r - d_chamfer[1], d_chamfer[0], { x, y, z1 }, { ax, ay, -az } );
        double d2 = GeometryHelpers::distanceToCircularFrustum(
            d_r, d_r - d_chamfer[1], d_chamfer[0], { x, y, z2 }, { ax, ay, az } );
        d = std::min( { d, d1, d2 } );
    }
    return d;
}


/********************************************************
 * Check if the point is inside the geometry             *
 ********************************************************/
bool Cylinder::inside( const Point &pos ) const
{
    double x  = pos.x() - d_offset[0];
    double y  = pos.y() - d_offset[1];
    double z  = pos.z() - d_offset[2];
    double R  = getR( z );
    double t1 = 1e-12 * R * R;
    double t2 = 1e-12 * std::max( fabs( d_z_min ), fabs( d_z_max ) );
    double r2 = x * x + y * y;
    bool in_r = r2 <= R * R + t1;
    bool in_z = z >= d_z_min - t2 && z <= d_z_max + t2;
    return in_r && in_z;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
int Cylinder::surface( const Point &pos ) const
{
    double x  = pos.x() - d_offset[0];
    double y  = pos.y() - d_offset[1];
    double z  = pos.z() - d_offset[2];
    double r  = std::sqrt( x * x + y * y );
    double R  = getR( z );
    double d1 = std::abs( r - R );
    double d2 = std::abs( z - d_z_min );
    double d3 = std::abs( z - d_z_max );
    if ( d1 <= std::min( d2, d3 ) )
        return 0; // Cylinder
    if ( d2 <= std::min( d1, d3 ) )
        return 1; // -z surface
    if ( d3 <= std::min( d1, d2 ) )
        return 2; // +z surface
    AMP_ERROR( "Internal error" );
    return -1;
}
Point Cylinder::surfaceNorm( const Point &pos ) const
{
    int s = surface( pos );
    if ( s == 1 ) {
        // -z surface
        return { 0, 0, -1 };
    } else if ( s == 2 ) {
        // +z surface
        return { 0, 0, 1 };
    } else {
        // r
        double x = pos.x() - d_offset[0];
        double y = pos.y() - d_offset[1];
        double n = std::sqrt( x * x + y * y );
        return { x / n, y / n, 0 };
    }
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point Cylinder::physical( const Point &pos ) const
{
    double z0 = d_z_min + pos[2] * ( d_z_max - d_z_min );
    double R  = getR( z0 );
    auto tmp  = GeometryHelpers::map_logical_circle( R, d_method, pos[0], pos[1] );
    double x  = tmp[0] + d_offset[0];
    double y  = tmp[1] + d_offset[1];
    double z  = z0 + d_offset[2];
    return { x, y, z };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point Cylinder::logical( const Point &pos ) const
{
    double R = getR( pos[2] - d_offset[2] );
    auto tmp = GeometryHelpers::map_circle_logical(
        R, d_method, pos[0] - d_offset[0], pos[1] - d_offset[1] );
    double z = ( pos[2] - d_z_min - d_offset[2] ) / ( d_z_max - d_z_min );
    return Point( tmp[0], tmp[1], z );
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point Cylinder::centroid() const
{
    return { d_offset[0], d_offset[1], d_offset[2] + 0.5 * ( d_z_max + d_z_min ) };
}
std::pair<Point, Point> Cylinder::box() const
{
    Point lb = { d_offset[0] - d_r, d_offset[1] - d_r, d_offset[2] + d_z_min };
    Point ub = { d_offset[0] + d_r, d_offset[1] + d_r, d_offset[2] + d_z_max };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double Cylinder::volume() const
{
    constexpr double pi = 3.141592653589793;
    double V            = ( d_z_max - d_z_min ) * pi * d_r * d_r;
    V += d_chamfer[0] * ( d_r + d_r - d_chamfer[1] );
    return V;
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize Cylinder::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.size() == 2u, "Size must be an array of length 2" );
    return { 2 * x[1], 2 * x[1], x[0] };
}
ArraySize Cylinder::getLogicalGridSize( const std::vector<double> &res ) const
{
    AMP_INSIST( res.size() == 3u, "Resolution must be an array of length 3" );
    size_t Nxy = d_r / std::min( res[0], res[1] );
    size_t Nz  = ( d_z_max - d_z_min ) / res[2];
    return { Nxy, Nxy, Nz };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void Cylinder::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
    d_offset[2] += x[2];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> Cylinder::clone() const
{
    return std::make_unique<Cylinder>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool Cylinder::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const Cylinder *>( &rhs );
    if ( !geom )
        return false;
    return d_r == geom->d_r && d_z_min == geom->d_z_min && d_z_max == geom->d_z_max &&
           d_offset == geom->d_offset && d_chamfer == geom->d_chamfer;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void Cylinder::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "r", d_r );
    AMP::IO::writeHDF5( fid, "z_min", d_z_min );
    AMP::IO::writeHDF5( fid, "z_max", d_z_max );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "chamfer", d_chamfer );
}
Cylinder::Cylinder( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "r", d_r );
    AMP::IO::readHDF5( fid, "z_min", d_z_min );
    AMP::IO::readHDF5( fid, "z_max", d_z_max );
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "chamfer", d_chamfer );
}


} // namespace AMP::Geometry
