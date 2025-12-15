#include "AMP/geometry/shapes/Parallelepiped.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Geometry {


/********************************************************
 * Constructor                                           *
 ********************************************************/
Parallelepiped::Parallelepiped( std::shared_ptr<const AMP::Database> db ) : LogicalGeometry( 3, 3 )
{
    // Fill some basic properties
    d_offset[0] = 0;
    d_offset[1] = 0;
    d_offset[2] = 0;
    d_offset[0] = 0;
    d_offset[1] = 0;
    d_offset[2] = 0;
    // Define the
    auto a = db->getVector<double>( "a" );
    auto b = db->getVector<double>( "b" );
    auto c = db->getVector<double>( "c" );
    AMP_ASSERT( a.size() == b.size() );
    AMP_ASSERT( a.size() == c.size() );
    if ( a.size() == 1u ) {
        AMP_ERROR( "Not finished" );
    } else if ( a.size() == 3u ) {
        for ( int i = 0; i < 3; i++ ) {
            d_a[i] = a[i];
            d_b[i] = b[i];
            d_c[i] = c[i];
        }
    } else {
        AMP_ERROR( "Either specify scalar values for a, b, c, theta, and gamna or vectors for a, "
                   "b, and c " );
    }
    // Compute the inverse matrix used to convert to logical coordinates
    // [ x ]   [ ax  bx  cx ] [ L1 ]
    // [ y ] = [ ay  by  cy ] [ L2 ]
    // [ z ]   [ az  bz  cz ] [ L3 ]
    d_M_inv[0] = d_b[1] * d_c[2] - d_c[1] * d_b[2];
    d_M_inv[1] = d_c[1] * d_a[2] - d_a[1] * d_c[2];
    d_M_inv[2] = d_a[1] * d_b[2] - d_b[1] * d_a[2];
    d_M_inv[3] = d_c[0] * d_b[2] - d_b[0] * d_c[2];
    d_M_inv[4] = d_a[0] * d_c[2] - d_c[0] * d_a[2];
    d_M_inv[5] = d_b[0] * d_a[2] - d_a[0] * d_b[2];
    d_M_inv[6] = d_b[0] * d_c[1] - d_c[0] * d_b[1];
    d_M_inv[7] = d_c[0] * d_a[1] - d_a[0] * d_c[1];
    d_M_inv[8] = d_a[0] * d_b[1] - d_b[0] * d_a[1];
    double det = d_a[0] * d_M_inv[0] + d_b[0] * d_M_inv[1] + d_c[0] * d_M_inv[2];
    for ( auto &x : d_M_inv )
        x /= det;
    // Compute the volume: abs(det(M))
    d_V = std::abs( det );
    // Compute the normals to the surface (pointing out from the center)
    auto computeNormal = []( auto a, auto b, auto c ) {
        auto n = AMP::Mesh::cross<double>( { a[0], a[1], a[2] }, { b[0], b[1], b[2] } );
        n      = normalize( n );
        if ( dot( n, { c[0], c[1], c[2] } ) < 0.0 )
            n = -n;
        return n;
    };
    d_n_ab = computeNormal( d_a, d_b, d_c );
    d_n_ac = computeNormal( d_a, d_c, d_b );
    d_n_bc = computeNormal( d_b, d_c, d_a );
}


/********************************************************
 * Compute the nearest point on the surface              *
 ********************************************************/
Point Parallelepiped::nearest( const Point &pos ) const
{
    auto L = logical( pos );
    L.x()  = std::max( L.x(), 0.0 );
    L.x()  = std::min( L.x(), 1.0 );
    L.y()  = std::max( L.y(), 0.0 );
    L.y()  = std::min( L.y(), 1.0 );
    L.z()  = std::max( L.z(), 0.0 );
    L.z()  = std::min( L.z(), 1.0 );
    return physical( L );
}


/********************************************************
 * Compute the distance to the object                    *
 ********************************************************/
double Parallelepiped::distance( const Point &pos0, const Point &ang ) const
{
    auto pos = pos0 - Point( d_offset[0], d_offset[1], d_offset[2] );
    Point p0 = { 0, 0, 0 };
    Point p1 = { d_a[0] + d_b[0] + d_c[0], d_a[1] + d_b[1] + d_c[1], d_a[2] + d_b[2] + d_c[2] };
    double d = std::numeric_limits<double>::infinity();
    auto fun = [this, pos, pos0, ang]( double d0, auto n, auto p ) {
        double d1 = GeometryHelpers::distanceToPlane( n, p, pos, ang );
        if ( d1 < d0 ) {
            auto pos2 = pos0 + d1 * ang;
            auto L    = logical( pos2 );
            double t1 = -1e-12;
            double t2 = 1.0 + 1e-12;
            if ( L.x() >= t1 && L.y() >= t1 && L.z() >= t1 && L.x() <= t2 && L.y() <= t2 &&
                 L.z() <= t2 )
                return d1;
        }
        return d0;
    };
    d = fun( d, d_n_ab, p0 );
    d = fun( d, d_n_ab, p1 );
    d = fun( d, d_n_ac, p0 );
    d = fun( d, d_n_ac, p1 );
    d = fun( d, d_n_bc, p0 );
    d = fun( d, d_n_bc, p1 );
    if ( inside( pos0 ) && d < 1e100 )
        d = -d;
    return d;
}


/********************************************************
 * Check if the ray is inside the geometry               *
 ********************************************************/
bool Parallelepiped::inside( const Point &pos ) const
{
    auto L    = logical( pos );
    double t1 = -1e-12;
    double t2 = 1.0 + 1e-12;
    return L.x() >= t1 && L.y() >= t1 && L.z() >= t1 && L.x() <= t2 && L.y() <= t2 && L.z() <= t2;
}


/********************************************************
 * Return the closest surface                            *
 ********************************************************/
int Parallelepiped::surface( const Point &pos ) const
{
    const auto L      = logical( pos );
    const double d[6] = { fabs( L.x() ),       fabs( 1.0 - L.x() ), fabs( L.y() ),
                          fabs( 1.0 - L.y() ), fabs( L.z() ),       fabs( 1.0 - L.z() ) };
    int index         = 0;
    double d_min      = d[0];
    for ( int i = 1; i < 6; i++ ) {
        if ( d[i] < d_min ) {
            index = i;
            d_min = d[i];
        }
    }
    return index;
}
Point Parallelepiped::surfaceNorm( const Point &pos ) const
{
    auto index = surface( pos );
    if ( index == 0 )
        return d_n_bc;
    else if ( index == 1 )
        return -d_n_bc;
    else if ( index == 2 )
        return d_n_ac;
    else if ( index == 3 )
        return -d_n_ac;
    else if ( index == 4 )
        return -d_n_ab;
    else if ( index == 5 )
        return -d_n_ab;
    else
        AMP_ERROR( "Internal error" );
    return {};
}


/********************************************************
 * Return the physical coordinates                       *
 ********************************************************/
Point Parallelepiped::physical( const Point &pos ) const
{
    const double L[3] = { pos.x(), pos.y(), pos.z() };
    const double x    = d_a[0] * L[0] + d_b[0] * L[1] + d_c[0] * L[2];
    const double y    = d_a[1] * L[0] + d_b[1] * L[1] + d_c[1] * L[2];
    const double z    = d_a[2] * L[0] + d_b[2] * L[1] + d_c[2] * L[2];
    return { x + d_offset[0], y + d_offset[1], z + d_offset[2] };
}


/********************************************************
 * Return the logical coordinates                        *
 ********************************************************/
Point Parallelepiped::logical( const Point &pos ) const
{
    double x  = pos.x() - d_offset[0];
    double y  = pos.y() - d_offset[1];
    double z  = pos.z() - d_offset[2];
    double L1 = d_M_inv[0] * x + d_M_inv[3] * y + d_M_inv[6] * z;
    double L2 = d_M_inv[1] * x + d_M_inv[4] * y + d_M_inv[7] * z;
    double L3 = d_M_inv[2] * x + d_M_inv[5] * y + d_M_inv[8] * z;
    return { L1, L2, L3 };
}


/********************************************************
 * Return the centroid and bounding box                  *
 ********************************************************/
Point Parallelepiped::centroid() const
{
    double x = 0.5 * ( d_a[0] + d_b[0] + d_c[0] );
    double y = 0.5 * ( d_a[1] + d_b[1] + d_c[1] );
    double z = 0.5 * ( d_a[2] + d_b[2] + d_c[2] );
    return { d_offset[0] + x, d_offset[1] + y, d_offset[2] + z };
}
std::pair<Point, Point> Parallelepiped::box() const
{
    double x = d_a[0] + d_b[0] + d_c[0];
    double y = d_a[1] + d_b[1] + d_c[1];
    double z = d_a[2] + d_b[2] + d_c[2];
    Point lb = { d_offset[0] + std::min( x, 0.0 ),
                 d_offset[1] + std::min( y, 0.0 ),
                 d_offset[2] + std::min( z, 0.0 ) };
    Point ub = { d_offset[0] + std::max( x, 0.0 ),
                 d_offset[1] + std::max( y, 0.0 ),
                 d_offset[2] + std::max( z, 0.0 ) };
    return { lb, ub };
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double Parallelepiped::volume() const { return d_V; }


/********************************************************
 * Return the logical grid                               *
 ********************************************************/
ArraySize Parallelepiped::getLogicalGridSize( const ArraySize &x ) const
{
    AMP_INSIST( x.size() == 3u, "Size must be an array of length 3" );
    return { x[0], x[1], x[2] };
}
ArraySize Parallelepiped::getLogicalGridSize( const std::vector<double> &res ) const
{
    AMP_INSIST( res.size() == 3u, "Resolution must be an array of length 3" );
    double a  = std::sqrt( d_a[0] * d_a[0] + d_a[1] * d_a[1] + d_a[2] * d_a[2] );
    double b  = std::sqrt( d_b[0] * d_b[0] + d_b[1] * d_b[1] + d_b[2] * d_b[2] );
    double c  = std::sqrt( d_c[0] * d_c[0] + d_c[1] * d_c[1] + d_c[2] * d_c[2] );
    size_t N1 = std::max<int>( a / res[0], 1 );
    size_t N2 = std::max<int>( b / res[1], 1 );
    size_t N3 = std::max<int>( c / res[2], 1 );
    return { N1, N2, N3 };
}


/********************************************************
 * Displace the mesh                                     *
 ********************************************************/
void Parallelepiped::displace( const double *x )
{
    d_offset[0] += x[0];
    d_offset[1] += x[1];
    d_offset[2] += x[2];
}


/********************************************************
 * Clone the object                                      *
 ********************************************************/
std::unique_ptr<AMP::Geometry::Geometry> Parallelepiped::clone() const
{
    return std::make_unique<Parallelepiped>( *this );
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool Parallelepiped::operator==( const Geometry &rhs ) const
{
    if ( &rhs == this )
        return true;
    auto geom = dynamic_cast<const Parallelepiped *>( &rhs );
    if ( !geom )
        return false;
    return d_a == geom->d_a && d_b == geom->d_b && d_c == geom->d_c && d_offset == geom->d_offset &&
           d_M_inv == geom->d_M_inv && d_V == geom->d_V && d_n_ab == geom->d_n_ab &&
           d_n_ac == geom->d_n_ac && d_n_bc == geom->d_n_bc;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void Parallelepiped::writeRestart( int64_t fid ) const
{
    LogicalGeometry::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "a", d_a );
    AMP::IO::writeHDF5( fid, "b", d_b );
    AMP::IO::writeHDF5( fid, "c", d_c );
    AMP::IO::writeHDF5( fid, "offset", d_offset );
    AMP::IO::writeHDF5( fid, "M_inv", d_M_inv );
    AMP::IO::writeHDF5( fid, "V", d_V );
    AMP::IO::writeHDF5( fid, "n_ab", d_n_ab );
    AMP::IO::writeHDF5( fid, "n_ac", d_n_ac );
    AMP::IO::writeHDF5( fid, "n_bc", d_n_bc );
}
Parallelepiped::Parallelepiped( int64_t fid ) : LogicalGeometry( fid )
{
    AMP::IO::readHDF5( fid, "a", d_a );
    AMP::IO::readHDF5( fid, "b", d_b );
    AMP::IO::readHDF5( fid, "c", d_c );
    AMP::IO::readHDF5( fid, "offset", d_offset );
    AMP::IO::readHDF5( fid, "M_inv", d_M_inv );
    AMP::IO::readHDF5( fid, "V", d_V );
    AMP::IO::readHDF5( fid, "n_ab", d_n_ab );
    AMP::IO::readHDF5( fid, "n_ac", d_n_ac );
    AMP::IO::readHDF5( fid, "n_bc", d_n_bc );
}


} // namespace AMP::Geometry
