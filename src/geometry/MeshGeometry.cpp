#include "AMP/geometry/MeshGeometry.h"
#include "AMP/IO/HDF.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/MeshUtilities.h"
#include "AMP/utils/arrayHelpers.h"
#include "AMP/utils/kdtree2.h"

#include "ProfilerApp.h"

#include <mutex>
#include <random>


namespace AMP::Geometry {


/********************************************************
 * Constructors                                          *
 ********************************************************/
MeshGeometry::MeshGeometry( std::shared_ptr<AMP::Mesh::Mesh> mesh )
    : Geometry( mesh->getDim() ),
      d_mesh( mesh ),
      d_pos_hash( static_cast<size_t>( -1 ) ),
      d_isConvex( false ),
      d_volume( 0 )
{
    PROFILE( "MeshGeometry::MeshGeometry" );
    AMP_ASSERT( d_mesh );
    AMP_ASSERT( static_cast<int>( d_mesh->getGeomType() ) == d_mesh->getDim() - 1 );
    AMP_ASSERT( d_mesh->getDim() == 3 );
    // Get the block ids (they will translate to surface ids)
    d_surfaceIds = d_mesh->getBlockIDs();
    // Initialize position related data
    d_find = AMP::Mesh::ElementFinder( d_mesh );
}
std::unique_ptr<AMP::Geometry::Geometry> MeshGeometry::clone() const
{
    return std::unique_ptr<AMP::Geometry::Geometry>( new MeshGeometry( d_mesh->clone() ) );
}


/********************************************************
 * Cache internal data                                   *
 ********************************************************/
void MeshGeometry::updateCache() const
{
    if ( d_pos_hash == d_mesh->positionHash() )
        return;
    // Lock a mutex to ensure only 1 thread updates the internal data
    PROFILE( "updateCache" );
    static std::mutex mtx;
    mtx.lock();
    d_pos_hash = d_mesh->positionHash();
    // Check that the element finder is up-to-date
    d_find.update();
    // Get a quick approximation to the centroid
    auto [lb, ub]  = box();
    Point centroid = 0.5 * ( lb + ub );
    // Initialize the tree to store inside data
    if ( d_inside.empty() ) {
        PROFILE( "updateCache-interior", 1 );
        // Choose a random direction to use for ray-cast to ensure we don't hit edges
        std::random_device rd;
        std::mt19937 gen( rd() );
        std::uniform_real_distribution<double> dis( 0, 1 );
        std::array<double, 3> dir = { dis( gen ), dis( gen ), dis( gen ) };
        dir                       = normalize( dir );
        // Get the points associated with each mesh element
        const auto &tree = d_find.getTree();
        std::map<AMP::Mesh::MeshElementID, std::vector<std::array<double, 3>>> map;
        for ( auto &[p, id] : tree.getPointsAndData() )
            map[id].push_back( p );
        // Add points on each side of the surface to identify if we we are inside/outside
        std::vector<std::array<double, 3>> points;
        points.emplace_back( centroid );
        for ( auto &[id, elemPoints] : map ) {
            auto elem               = d_mesh->getElement( id );
            std::array<double, 3> n = elem.norm();
            for ( auto &p : elemPoints ) {
                points.push_back( p + 1e-6 * n );
                points.push_back( p - 1e-6 * n );
            }
        }
        std::vector<bool> data( points.size() );
        for ( size_t i = 0; i < points.size(); i++ )
            data[i] = d_find.distance( points[i], dir ) <= 0;
        d_inside = kdtree2<3, bool>( points, data );
    }
    // Check that the approximate centroid is inside the volume
    if ( !std::get<1>( d_inside.findNearest( centroid ) ) ) {
        PROFILE( "updateCache-centroid", 1 );
        auto tmp = d_inside.findNearest( centroid, 5 );
        for ( auto it = tmp.rbegin(); it != tmp.rend(); ++it ) {
            if ( std::get<1>( *it ) )
                centroid = std::get<0>( *it );
        }
        AMP_ASSERT( std::get<1>( d_inside.findNearest( centroid ) ) );
    }
    // Check if the mesh is convex
    {
        PROFILE( "updateCache-isConvex", 1 );
        d_isConvex = true;
        double tol = 1e-4 * abs( ub - lb );
        std::vector<Point> vertices;
        for ( const auto &elem : d_mesh->getIterator( d_mesh->getGeomType() ) ) {
            // Get the normal to the plane of the element (pointing away from the centroid)
            auto n = elem.norm();
            auto a = elem.centroid();
            if ( dot( n, a - centroid ) < 0 )
                n = -n;
            // Check the neighboring vertices to ensure they are not in front of the plane
            elem.getNeighborVertices( vertices );
            for ( const auto &p : vertices ) {
                double v   = dot( n, p - a );
                d_isConvex = d_isConvex && v >= -tol;
            }
        }
        printf( "isConvex = %s\n", d_isConvex ? "true" : "false" );
    }
    // Calculate the volume/centroid
    if ( d_isConvex ) {
        // Loop through the elements creating pyramids and using them to compute the centroid/volume
        PROFILE( "updateCache-volume-1", 1 );
        d_volume   = 0;
        d_centroid = { 0, 0, 0 };
        for ( const auto &elem : d_mesh->getIterator( d_mesh->getGeomType() ) ) {
            auto A = elem.volume();
            auto n = elem.norm();
            auto h = fabs( dot( n, centroid ) );
            auto V = 1.0 / 3.0 * A * h;
            d_volume += V;
            auto c = 0.75 * elem.centroid() + 0.25 * centroid;
            d_centroid += V * c;
        }
        d_centroid *= 1.0 / d_volume;
    } else {
        // Get the volume overlap
        PROFILE( "updateCache-volume-2", 1 );
        std::vector<int> N( d_mesh->getDim(), 100 );
        auto V = AMP::Mesh::volumeOverlap( *this, N );
        // Calculate the total volume
        d_volume = V.sum();
        // Calculate the centroid
        auto xy    = V.sum( 2 );
        auto x     = xy.sum( 1 );
        auto y     = xy.sum( 0 );
        auto z     = V.sum( 1 ).sum( 0 );
        d_centroid = { 0, 0, 0 };
        for ( size_t i = 0; i < x.length(); i++ )
            d_centroid.x() += i * x( i );
        for ( size_t i = 0; i < y.length(); i++ )
            d_centroid.y() += i * y( i );
        for ( size_t i = 0; i < z.length(); i++ )
            d_centroid.z() += i * z( i );
        d_centroid.x() = lb[0] + ( ub[0] - lb[0] ) * d_centroid.x() / ( x.length() * d_volume );
        d_centroid.y() = lb[1] + ( ub[1] - lb[1] ) * d_centroid.y() / ( y.length() * d_volume );
        d_centroid.z() = lb[2] + ( ub[2] - lb[2] ) * d_centroid.z() / ( z.length() * d_volume );
    }
    // Unlock the mutex
    mtx.unlock();
}


/********************************************************
 * Get the distance to the surface                       *
 ********************************************************/
Point MeshGeometry::nearest( const Point &pos ) const
{
    if ( inside( pos ) )
        return pos;
    return d_find.nearest( pos ).second;
}
double MeshGeometry::distance( const Point &pos, const Point &dir ) const
{
    return d_find.distance( pos, dir );
}
bool MeshGeometry::inside( const Point &pos ) const
{
    updateCache();
    auto nearest = d_inside.findNearest( pos );
    bool inside  = std::get<1>( nearest );
    return inside;
}


/********************************************************
 * Get the surface                                       *
 ********************************************************/
int MeshGeometry::NSurface() const { return d_surfaceIds.size(); }
int MeshGeometry::surface( const Point &x ) const
{
    if ( d_surfaceIds.empty() )
        return 0;
    if ( d_surfaceIds.size() == 1 )
        return d_surfaceIds[0];
    auto elem = d_find.nearest( x ).first;
    AMP_ASSERT( !elem.isNull() );
    for ( auto id : d_surfaceIds ) {
        if ( elem.isInBlock( id ) )
            return id;
    }
    return 0;
}
Point MeshGeometry::surfaceNorm( const Point &x ) const
{
    auto elem = d_find.nearest( x ).first;
    AMP_ASSERT( !elem.isNull() );
    return elem.norm();
}


/********************************************************
 * Get the centroid/box                                  *
 ********************************************************/
Point MeshGeometry::centroid() const
{
    updateCache();
    return d_centroid;
}
std::pair<Point, Point> MeshGeometry::box() const
{
    auto box = d_mesh->getBoundingBox();
    Point p0( box.size() / 2 ), p1( box.size() / 2 );
    for ( size_t d = 0; d < box.size() / 2; d++ ) {
        p0[d] = box[2 * d];
        p1[d] = box[2 * d + 1];
    }
    return std::make_pair( p0, p1 );
}


/********************************************************
 * Return the volume                                     *
 ********************************************************/
double MeshGeometry::volume() const
{
    updateCache();
    return d_volume;
}


/********************************************************
 * Get the centroid/box                                  *
 ********************************************************/
void MeshGeometry::displace( const double *x )
{
    std::vector<double> x2( x, x + d_mesh->getDim() );
    d_mesh->displaceMesh( x2 );
    d_inside = kdtree2<3, bool>();
}


/********************************************************
 * Get the nearest element                               *
 ********************************************************/
bool MeshGeometry::isConvex() const
{
    updateCache();
    return d_isConvex;
}


/********************************************************
 * Compare the geometry                                  *
 ********************************************************/
bool MeshGeometry::operator==( const Geometry &rhs ) const
{
    auto geom = dynamic_cast<const MeshGeometry *>( &rhs );
    if ( !geom )
        return false;
    return d_mesh == geom->d_mesh;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MeshGeometry::writeRestart( int64_t fid ) const
{
    Geometry::writeRestart( fid );
    AMP_ERROR( "Not finished" );
}
MeshGeometry::MeshGeometry( int64_t fid )
    : Geometry( fid ), d_pos_hash( static_cast<size_t>( -1 ) ), d_isConvex( false ), d_volume( 0 )
{
    AMP_ERROR( "Not finished" );
}

} // namespace AMP::Geometry
