#include "AMP/geometry/shapes/RegularPolygon.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
RegularPolygon::RegularPolygon( std::shared_ptr<const AMP::Database> db )
    : LogicalGeometry( 2, 2, { 1, 1, 1, 1, -3, -3 } )
{
    d_N = db->getScalar<double>( "N" );
    d_R = db->getScalar<double>( "R" );
    computeNorms();
}
RegularPolygon::RegularPolygon( int N, double R )
    : LogicalGeometry( 2, 2, { 1, 1, 1, 1, -3, -3 } ), d_N( N ), d_R( R )
{
    d_offset[0] = 0;
    d_offset[1] = 0;
    computeNorms();
}
void RegularPolygon::computeNorms()
{
    // Get the vertices
    d_vertices = GeometryHelpers::get_poly_vertices( d_N, d_R );
    for ( auto &p : d_vertices ) {
        p[0] += d_offset[0];
        p[1] += d_offset[1];
    }
    // Calculate the normals
    d_norm.resize( d_N );
    d_norm[0] = GeometryHelpers::normal( d_vertices.back(), d_vertices[0] );
    for ( size_t i = 1; i < d_vertices.size(); i++ )
        d_norm[i] = GeometryHelpers::normal( d_vertices[i - 1], d_vertices[i] );
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
std::tuple<Point, double, int> RegularPolygon::nearest2( const Point &pos ) const
{
    std::array<double, 2> p0 = { pos.x(), pos.y() };
    // Check the intersection with each line segment (keeping the closest)
    Point p  = GeometryHelpers::nearest( d_vertices[0], d_vertices.back(), p0 );
    double d = ( p - pos ).norm();
    int k    = 0;
    for ( size_t i = 1; i < d_vertices.size(); i++ ) {
        Point p2  = GeometryHelpers::nearest( d_vertices[i], d_vertices[i - 1], p0 );
        double d2 = ( p2 - pos ).norm();
        if ( d2 < d ) {
            d = d2;
            p = p2;
            k = i;
        }
    }
    return std::tie( p, d, k );
}
Point RegularPolygon::nearest( const Point &pos ) const
{
    if ( inside( pos ) )
        return pos;
    auto r = nearest2( pos );
    return std::get<0>( r );
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double RegularPolygon::distance( const Point &pos, const Point &ang ) const
{
    // Check the intersection with each line segment (keeping the closest)
    double d = GeometryHelpers::distanceToLine( pos, ang, d_vertices[0], d_vertices.back() );
    for ( size_t i = 1; i < d_vertices.size(); i++ ) {
        double d2 = GeometryHelpers::distanceToLine( pos, ang, d_vertices[i], d_vertices[i - 1] );
        if ( d2 < d )
            d = d2;
    }
    if ( d == std::numeric_limits<double>::infinity() )
        return d;
    if ( inside( pos ) )
        return -d;
    return d;
}


/********************************************************
 * Check if the ray is inside the geometry               *
 ********************************************************/
bool RegularPolygon::inside( const Point &pos ) const
{
    auto L    = logical( pos );
    double t1 = -1e-12;
    double t2 = 1.0 + 1e-12;
    return L.x() >= t1 && L.y() >= t1 && L.z() >= t1 && L.x() <= t2 && L.y() <= t2 && L.z() <= t2;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
int RegularPolygon::NSurface() const { return d_N; }
int RegularPolygon::surface( const Point &pos ) const
{
    auto r = nearest2( pos );
    return std::get<2>( r );
}
Point RegularPolygon::surfaceNorm( const Point &pos ) const
{
    int i = surface( pos );
    return d_norm[i];
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point RegularPolygon::physical( const Point &pos ) const
{
    auto tmp = GeometryHelpers::map_logical_poly( d_N, d_R, pos.x(), pos.y() );
    double x = tmp[0] + d_offset[0];
    double y = tmp[1] + d_offset[1];
    return { x, y };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point RegularPolygon::logical( const Point &pos ) const
{
    double x = pos.x() - d_offset[0];
    double y = pos.y() - d_offset[1];
    auto tmp = GeometryHelpers::map_poly_logical( d_N, d_R, x, y );
    return Point( tmp[0], tmp[1] );
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point RegularPolygon::centroid() const { return { d_offset[0], d_offset[1] }; }
std::pair<Point, Point> RegularPolygon::box() const
{
    Point lb = { d_offset[0] - d_R, d_offset[1] - d_R };
    Point ub = { d_offset[0] + d_R, d_offset[1] + d_R };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double RegularPolygon::volume() const
{
    constexpr double pi = 3.141592653589793;
    return 0.5 * d_N * d_R * d_R * sin( 2 * pi / d_N );
}


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize RegularPolygon::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.ndim() == 1u, "Size must be an array of length 1" );
    return { 2 * x[0], 2 * x[0] };
}
ArraySize RegularPolygon::getLogicalGridSize( const std::vector<double> &res ) const
{
    AMP_INSIST( res.size() == 2u, "Resolution must be an array of length 2" );
    return { (size_t) ( d_R / res[0] ), (size_t) ( d_R / res[1] ) };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void RegularPolygon::displace( const double *x )
{
    // Update the offsets
    d_offset[0] += x[0];
    d_offset[1] += x[1];
    // Get the offsets
    d_vertices = GeometryHelpers::get_poly_vertices( d_N, d_R );
    for ( auto &p : d_vertices ) {
        p[0] += d_offset[0];
        p[1] += d_offset[1];
    }
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> RegularPolygon::clone() const
{
    return std::make_unique<RegularPolygon>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
static bool approx_equal( const std::array<double, 2> &x, const std::array<double, 2> &y )
{
    double d2 = ( x[0] - y[0] ) * ( x[0] - y[0] ) + ( x[1] - y[1] ) * ( x[1] - y[1] );
    return d2 < ( 1e-12 * 1e-12 );
}
bool RegularPolygon::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const RegularPolygon *>( &rhs );
    if ( !geom )
        return false;
    if ( d_N != geom->d_N || d_R != geom->d_R || d_offset != geom->d_offset )
        return false;
    bool equal = true;
    for ( size_t i = 0; i < d_vertices.size(); i++ )
        equal = equal && approx_equal( d_vertices[i], geom->d_vertices[i] );
    for ( size_t i = 0; i < d_norm.size(); i++ )
        equal = equal && approx_equal( d_norm[i], geom->d_norm[i] );
    return equal;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void RegularPolygon::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "N", d_N );
    AMP::IO::writeHDF5( fid, "R", d_R );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
}
RegularPolygon::RegularPolygon( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "N", d_N );
    AMP::IO::readHDF5( fid, "R", d_R );
    AMP::IO::readHDF5( fid, "offset", d_offset );
    computeNorms();
}


} // namespace AMP::Geometry
