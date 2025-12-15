#include "AMP/geometry/shapes/Tube.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>


namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
Tube::Tube( std::shared_ptr<const AMP::Database> db )
    : LogicalGeometry( 3, 3, { 8, 4, -1, -1, 2, 1 } )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    d_offset[2] = 0;
    auto range  = db->getVector<double>( "Range" );
    AMP_INSIST( range.size() == 4u, "Range must be an array of length 4" );
    d_r_min = range[0];
    d_r_max = range[1];
    d_z_min = range[2];
    d_z_max = range[3];
}
Tube::Tube( double r_min, double r_max, double z_min, double z_max )
    : LogicalGeometry( 3, 3, { 8, 4, -1, -1, 2, 1 } ),
      d_r_min( r_min ),
      d_r_max( r_max ),
      d_z_min( z_min ),
      d_z_max( z_max )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    d_offset[2] = 0;
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
Point Tube::nearest( const Point &pos ) const
{
    // Get the current point in the reference frame of the tube
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2];
    // Calculate the nearest point
    z        = std::min( z, d_z_max );
    z        = std::max( z, d_z_min );
    double r = std::sqrt( x * x + y * y );
    if ( r == 0 ) {
        x = d_r_min;
    } else if ( r < d_r_min ) {
        x *= d_r_min / r;
        y *= d_r_min / r;
    } else if ( r > d_r_max ) {
        x *= d_r_max / r;
        y *= d_r_max / r;
    }
    return { x + d_offset[0], y + d_offset[1], z + d_offset[2] };
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double Tube::distance( const Point &pos, const Point &ang ) const
{
    // Get the current point in the reference frame of the cylinder
    double h = d_z_max - d_z_min;
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2] - 0.5 * ( d_z_min + d_z_max );
    // Compute the distance to the tube
    return GeometryHelpers::distanceToTube( d_r_min, d_r_max, h, { x, y, z }, ang );
}


/********************************************************
 * Check if the point is inside the geometry             *
 ********************************************************/
bool Tube::inside( const Point &pos ) const
{
    double x  = pos.x() - d_offset[0];
    double y  = pos.y() - d_offset[1];
    double z  = pos.z() - d_offset[2];
    double r2 = x * x + y * y;
    double R1 = d_r_min * d_r_min;
    double R2 = d_r_max * d_r_max;
    double t1 = 1e-12 * std::max( R1, R2 );
    double t2 = 1e-12 * std::max( fabs( d_z_min ), fabs( d_z_max ) );
    bool in_r = r2 >= ( R1 - t1 ) && r2 <= ( R2 + t1 );
    bool in_z = z >= ( d_z_min - t2 ) && z <= ( d_z_max + t2 );
    return in_r && in_z;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
int Tube::surface( const Point &pos ) const
{
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double z = pos.z() - d_offset[2];
    if ( z <= d_z_min )
        return 0;
    if ( z >= d_z_max )
        return 1;
    double r  = std::sqrt( x * x + y * y );
    double d1 = fabs( z - d_z_min );
    double d2 = fabs( z - d_z_max );
    double d3 = fabs( r - d_r_min );
    double d4 = fabs( r - d_r_max );
    if ( d1 <= std::min( { d2, d3, d4 } ) )
        return 0;
    if ( d2 <= std::min( { d1, d3, d4 } ) )
        return 1;
    if ( d3 <= std::min( { d1, d2, d4 } ) )
        return 2;
    if ( d4 <= std::min( { d1, d2, d3 } ) )
        return 3;
    AMP_ERROR( "Internal error" );
    return -1;
}
Point Tube::surfaceNorm( const Point &pos ) const
{
    int s = surface( pos );
    if ( s == 0 )
        return { 0, 0, -1 };
    if ( s == 1 )
        return { 0, 0, 1 };
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    double n = std::sqrt( x * x + y * y );
    if ( s == 2 )
        return { -x / n, -y / n, 0 };
    if ( s == 3 )
        return { x / n, y / n, 0 };
    AMP_ERROR( "Internal error" );
    return { 0, 0, 0 };
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point Tube::physical( const Point &pos ) const
{
    constexpr double pi = 3.141592653589793116;
    // Compute r, theta
    double r     = d_r_min + pos[0] * ( d_r_max - d_r_min );
    double theta = 2.0 * pi * ( pos[1] - 0.5 );
    // Compute the physical coordinate
    double x = r * cos( theta ) + d_offset[0];
    double y = r * sin( theta ) + d_offset[1];
    double z = d_z_min + pos[2] * ( d_z_max - d_z_min ) + d_offset[2];
    return { x, y, z };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point Tube::logical( const Point &pos ) const
{
    constexpr double pi = 3.141592653589793116;
    // Compute r, theta
    double r     = std::sqrt( ( pos[0] - d_offset[0] ) * ( pos[0] - d_offset[0] ) +
                          ( pos[1] - d_offset[1] ) * ( pos[1] - d_offset[1] ) );
    double theta = acos( ( pos[0] - d_offset[0] ) / r );
    if ( asin( ( pos[1] - d_offset[1] ) / r ) < 0 )
        theta = -theta;
    // Compute the logical coordinate
    double x = ( r - d_r_min ) / ( d_r_max - d_r_min );
    double y = 0.5 + theta / ( 2.0 * pi );
    double z = ( pos[2] - d_z_min - d_offset[2] ) / ( d_z_max - d_z_min );
    return { x, y, z };
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point Tube::centroid() const
{
    return { d_offset[0], d_offset[1], d_offset[2] + 0.5 * ( d_z_max + d_z_min ) };
}
std::pair<Point, Point> Tube::box() const
{
    Point lb = { d_offset[0] - d_r_max, d_offset[1] - d_r_max, d_offset[2] + d_z_min };
    Point ub = { d_offset[0] + d_r_max, d_offset[1] + d_r_max, d_offset[2] + d_z_max };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double Tube::volume() const
{
    constexpr double pi = 3.141592653589793;
    double h            = d_z_max - d_z_min;
    double V1           = h * pi * d_r_max * d_r_max;
    double V2           = h * pi * d_r_min * d_r_min;
    return V1 - V2;
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize Tube::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.size() == 3u, "Size must be an array of length 1" );
    return { x[0], x[1], x[2] };
}
ArraySize Tube::getLogicalGridSize( const std::vector<double> &res ) const
{
    constexpr double pi = 3.141592653589793;
    AMP_INSIST( res.size() == 3u, "Resolution must be an array of length 3" );
    int Nx = ( d_r_max - d_r_min ) / std::min( res[0], res[1] );
    int Ny = 2 * pi * d_r_max / std::min( res[0], res[1] );
    int Nz = ( d_z_max - d_z_min ) / res[2];
    return { Nx, Ny, Nz };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void Tube::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
    d_offset[2] += x[2];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> Tube::clone() const
{
    return std::make_unique<Tube>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool Tube::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const Tube *>( &rhs );
    if ( !geom )
        return false;
    return d_r_min == geom->d_r_min && d_r_max == geom->d_r_max && d_z_min == geom->d_z_min &&
           d_z_max == geom->d_z_max && d_offset[0] == geom->d_offset[0] &&
           d_offset[1] == geom->d_offset[1] && d_offset[2] == geom->d_offset[2];
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void Tube::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "r_min", d_r_min );
    AMP::IO::writeHDF5( fid, "r_max", d_r_max );
    AMP::IO::writeHDF5( fid, "z_min", d_z_min );
    AMP::IO::writeHDF5( fid, "z_max", d_z_max );
}
Tube::Tube( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "r_min", d_r_min );
    AMP::IO::readHDF5( fid, "r_max", d_r_max );
    AMP::IO::readHDF5( fid, "z_min", d_z_min );
    AMP::IO::readHDF5( fid, "z_max", d_z_max );
}


} // namespace AMP::Geometry
