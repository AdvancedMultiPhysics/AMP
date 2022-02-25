#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/mesh/MeshUtilities.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <random>


namespace AMP::Geometry {


// Generate a random direction
static inline Point genRandDir( int ndim )
{
    static std::mt19937 gen( 47253 );
    static std::uniform_real_distribution<double> dis( -1, 1 );
    return normalize( Point( ndim, { dis( gen ), dis( gen ), dis( gen ) } ) );
}


// Add the surface points from the rays
static bool addSurfacePoints( const AMP::Geometry::Geometry &geom,
                              const Point &x0,
                              const Point &dir,
                              std::vector<Point> &surfacePoints )
{
    double d = geom.distance( x0, dir );
    AMP_ASSERT( d == d );
    bool inside = d <= 0;
    double d2   = 0;
    int it      = 0;
    while ( fabs( d ) < 1e100 ) {
        d2 += fabs( d );
        surfacePoints.push_back( x0 + d2 * dir );
        d2 += 1e-6;
        d = geom.distance( x0 + d2 * dir, dir );
        ++it;
        if ( it > 100 )
            AMP_ERROR( "Infinite surfaces" );
    }
    return inside;
}


// Run logical geometry specific tests
static bool testLogicalGeometry( const AMP::Geometry::LogicalGeometry &geom, AMP::UnitTest &ut )
{
    bool pass = true;
    auto name = geom.getName();
    int ndim  = geom.getDim();
    // Test logical/physical transformations
    PROFILE_START( "testGeometry-logical " + name );
    auto [lb, ub] = geom.box();
    std::mt19937 gen( 54612 );
    std::uniform_real_distribution<> dis[3];
    for ( int d = 0; d < geom.getDim(); d++ ) {
        double dx = ub[d] - lb[d];
        dis[d]    = std::uniform_real_distribution<>( lb[d] - 0.2 * dx, ub[d] + 0.2 * dx );
    }
    auto p     = lb;
    size_t N   = 10000;
    bool pass2 = true;
    for ( size_t i = 0; i < N; i++ ) {
        for ( int d = 0; d < geom.getDim(); d++ )
            p[d] = dis[d]( gen );
        auto p2 = geom.logical( p );
        auto p3 = geom.physical( p2 );
        for ( int d = 0; d < ndim; d++ )
            pass2 = pass2 && fabs( p[d] - p3[d] ) < 1e-6;
    }
    pass = pass && pass2;
    if ( !pass2 && geom.getLogicalDim() == ndim )
        ut.failure( "testGeometry physical-logical-physical: " + name );
    else if ( !pass2 )
        ut.expected_failure( "testGeometry physical-logical-physical: " + name );
    PROFILE_STOP( "testGeometry-logical " + name );
    return pass;
}


// Test the centroid of an object
static bool testCentroid( const AMP::Geometry::Geometry &geom, AMP::UnitTest &ut )
{
    auto name = geom.getName();
    int ndim  = geom.getDim();
    // Get the centroid
    auto centroid = geom.centroid();
    // Check that the centroid is within the bounding box
    auto box  = geom.box();
    bool pass = centroid.ndim() == ndim;
    for ( int d = 0; d < ndim; d++ )
        pass = pass && centroid[d] >= box.first[d] && centroid[d] <= box.second[d];
    if ( !pass ) {
        ut.failure( "testGeometry centroid/box: " + name );
        return false;
    }
    // Check if we are dealing with a surface in a volume
    if ( ndim != static_cast<int>( geom.getGeomType() ) ) {
        ut.expected_failure( "testGeometry-centroid is not complete for ndim()!=geomType(): " +
                             name );
        return false;
    }
    // Estimate the centroid by randomly sampling space if it is inside the object
    // Note we use a non-random seed to ensure test doesn't fail periodically due to tolerances
    Point p( ndim, { 0, 0, 0 } );
    static std::mt19937 gen( 56871 );
    std::uniform_real_distribution<double> dist[3];
    for ( int d = 0; d < ndim; d++ )
        dist[d] = std::uniform_real_distribution<double>( box.first[d], box.second[d] );
    size_t N = 100000;
    for ( size_t i = 0; i < N; ) {
        Point pos( ndim, { 0, 0, 0 } );
        for ( int d = 0; d < ndim; d++ )
            pos[d] = dist[d]( gen );
        if ( geom.inside( pos ) ) {
            p += pos;
            i++;
        }
    }
    p *= 1.0 / N;
    double err = 0;
    for ( int d = 0; d < ndim; d++ ) {
        double dx = box.second[d] - box.first[d];
        err       = std::max( err, fabs( p[d] - centroid[d] ) / dx );
    }
    pass = err < 0.01;
    using AMP::Utilities::stringf;
    if ( !pass )
        ut.failure( stringf( "testGeometry centroid: %s (%f)", name.data(), err ) );
    return pass;
}


// Run all geometry based tests
void testGeometry( const AMP::Geometry::Geometry &geom, AMP::UnitTest &ut )
{
    auto multigeom = dynamic_cast<const MultiGeometry *>( &geom );
    if ( multigeom ) {
        for ( const auto geom2 : multigeom->getGeometries() )
            testGeometry( *geom2, ut );
    }
    // Get the physical dimension
    int ndim  = geom.getDim();
    auto name = geom.getName();
    PROFILE_SCOPED( timer, "testGeometry " + name );
    // Test logical geometries
    auto logicalGeom = dynamic_cast<const AMP::Geometry::LogicalGeometry *>( &geom );
    if ( logicalGeom ) {
        bool pass2 = testLogicalGeometry( *logicalGeom, ut );
        if ( !pass2 )
            return;
    }
    // Test the centroid of the object
    bool pass = testCentroid( geom, ut );
    // First get the centroid and the range
    auto center = geom.centroid();
    // Use a series of rays projecting from the centroid to get points on the surface
    PROFILE_START( "testGeometry-surface " + name );
    size_t N = 10000;
    std::vector<Point> surfacePoints;
    surfacePoints.reserve( N );
    bool all_hit = true;
    while ( surfacePoints.size() < N ) {
        auto dir  = genRandDir( ndim );
        bool test = addSurfacePoints( geom, center, dir, surfacePoints );
        all_hit   = all_hit && test;
    }
    pass = pass && !surfacePoints.empty();
    if ( surfacePoints.empty() )
        ut.failure( "testGeometry unable to get surface: " + name );
    if ( geom.inside( center ) && !all_hit )
        ut.failure( "testGeometry failed all rays hit: " + name );
    // Add points propagating from box surface
    if ( ndim == 3 && !multigeom ) {
        int n         = 11;
        auto [lb, ub] = geom.box();
        auto dx       = 1.0 / n * ( ub - lb );
        for ( int i = 0; i < n; i++ ) {
            for ( int j = 0; j < n; j++ ) {
                Point x0 = { lb[0] - dx[0],
                             lb[1] + ( i + 0.5 ) * dx[1],
                             lb[2] + ( j + 0.5 ) * dx[2] };
                Point y0 = { lb[0] + ( i + 0.5 ) * dx[0],
                             lb[1] - dx[1],
                             lb[2] + ( j + 0.5 ) * dx[2] };
                Point z0 = { lb[0] + ( i + 0.5 ) * dx[0],
                             lb[1] + ( j + 0.5 ) * dx[1],
                             lb[2] - dx[2] };
                addSurfacePoints( geom, x0, { 1, 0, 0 }, surfacePoints );
                addSurfacePoints( geom, y0, { 0, 1, 0 }, surfacePoints );
                addSurfacePoints( geom, z0, { 0, 0, 1 }, surfacePoints );
            }
        }
    }
    PROFILE_STOP( "testGeometry-surface " + name );
    // Verify each surface point is "inside" the object
    PROFILE_START( "testGeometry-inside " + name );
    bool pass_inside = true;
    for ( const auto &tmp : surfacePoints ) {
        bool inside = geom.inside( tmp );
        if ( !inside ) {
            pass_inside = false;
            std::cout << "testGeometry-inside: " << tmp << std::endl;
            break;
        }
    }
    pass = pass && pass_inside;
    if ( !pass_inside )
        ut.failure( "testGeometry surface inside geometry: " + name );
    PROFILE_STOP( "testGeometry-inside " + name );
    // Project each surface point in a random direction and back propagate to get the same point
    PROFILE_START( "testGeometry-distance " + name );
    bool pass_projection = true;
    auto box             = geom.box();
    auto length          = box.second - box.first;
    const double d0      = 0.2 * std::max( { length.x(), length.y(), length.z() } );
    for ( const auto &tmp : surfacePoints ) {
        auto ang = genRandDir( ndim );
        auto pos = tmp - d0 * ang;
        double d = fabs( geom.distance( pos, ang ) );
        for ( int it = 0; it < 1000 && d < d0 - 1e-5; it++ ) {
            // We may have crossed multiple surfaces, find the original
            d += 1e-6;
            auto pos2 = pos + d * ang;
            d += fabs( geom.distance( pos2, ang ) );
        }
        if ( fabs( d - d0 ) > 1e-5 ) {
            std::cout << "testGeometry-distance: " << d0 << " " << d << " " << tmp << " " << pos
                      << std::endl;
            pass_projection = false;
            break;
        }
    }
    pass = pass && pass_projection;
    if ( !pass_projection )
        ut.failure( "testGeometry distances do not match: " + name );
    PROFILE_STOP( "testGeometry-distance " + name );
    // Get a set of interior points by randomly sampling the space
    // Note we use a non-random seed to ensure test doesn't fail periodically due to tolerances
    PROFILE_START( "testGeometry-sample " + name );
    static std::mt19937 gen( 84397 );
    std::uniform_real_distribution<double> dist[3];
    for ( int d = 0; d < ndim; d++ )
        dist[d] = std::uniform_real_distribution<double>( box.first[d], box.second[d] );
    std::vector<Point> interiorPoints;
    for ( int i = 0; i < 10000; i++ ) {
        Point pos( ndim, { 0, 0, 0 } );
        for ( int d = 0; d < ndim; d++ )
            pos[d] = dist[d]( gen );
        if ( geom.inside( pos ) )
            interiorPoints.push_back( pos );
    }
    PROFILE_STOP( "testGeometry-sample " + name );
    // Check that nearest returns the surface/interior points
    PROFILE_START( "testGeometry-nearest " + name );
    bool pass_nearest = true;
    for ( const auto &p0 : surfacePoints ) {
        auto p   = geom.nearest( p0 );
        double d = ( p - p0 ).abs();
        if ( d > 1e-8 ) {
            bool test = geom.inside( p0 );
            p         = geom.nearest( p0 );
            NULL_USE( p );
            NULL_USE( test );
            pass_nearest = false;
        }
    }
    for ( const auto &p0 : interiorPoints ) {
        auto p   = geom.nearest( p0 );
        double d = ( p - p0 ).abs();
        if ( d > 1e-8 ) {
            p = geom.nearest( p0 );
            NULL_USE( p );
            pass_nearest = false;
        }
    }
    pass = pass && pass_nearest;
    if ( !pass_nearest )
        ut.failure( "testGeometry-nearest: " + name );
    PROFILE_STOP( "testGeometry-nearest " + name );
    // Test getting surface normals
    if ( !multigeom ) {
        bool passNorm = true;
        for ( const auto &p : surfacePoints ) {
            auto norm = geom.surfaceNorm( p );
            double n  = sqrt( norm.x() * norm.x() + norm.y() * norm.y() + norm.z() * norm.z() );
            // auto p1   = p - 1e-5 * norm;
            // auto p2   = p + 1e-5 * norm;
            passNorm = passNorm && fabs( n - 1.0 ) < 1e-6;
            // passNorm  = passNorm && geom.inside( p1 ) && !geom.inside( p2 );
        }
        pass = pass && passNorm;
        if ( !passNorm )
            ut.failure( "testGeometry surfaceNorm: " + name );
    }
    // Test getting the volume
    {
        PROFILE_START( "testGeometry-volume " + name );
        double volume    = geom.volume();
        double boxVolume = 1.0;
        for ( int d = 0; d < ndim; d++ )
            boxVolume *= box.second[d] - box.first[d];
        bool passVol = volume > 0;
        if ( ndim == static_cast<int>( geom.getGeomType() ) )
            passVol = passVol && volume <= boxVolume;
        pass = pass && passVol;
        if ( !passVol )
            ut.failure( "testGeometry volume: " + name );
        // Test mesh utilities volume overlap
        if ( ndim == static_cast<int>( geom.getGeomType() ) && !multigeom ) {
            auto tmp      = AMP::Mesh::volumeOverlap( geom, std::vector<int>( ndim, 35 ) );
            double vol2   = tmp.sum();
            bool passVol2 = fabs( vol2 - volume ) < 0.01 * volume;
            pass          = pass && passVol2;
            if ( !passVol2 )
                ut.failure( "testGeometry volumeOverlap: " + name );
        }
        PROFILE_STOP( "testGeometry-volume " + name );
    }
    // Test getting the surface id
    if ( !multigeom ) {
        PROFILE_START( "testGeometry-surfaceID " + name );
        std::set<int> ids;
        for ( const auto &p : surfacePoints )
            ids.insert( geom.surface( p ) );
        if ( (int) ids.size() != geom.NSurface() ) {
            using AMP::Utilities::stringf;
            auto msg = stringf( "testGeometry surface: %s (%i,%i)\n",
                                name.data(),
                                geom.NSurface(),
                                (int) ids.size() );
            msg += "           ids = ";
            msg += AMP::Utilities::to_string( std::vector<int>( ids.begin(), ids.end() ) );
            ut.failure( msg );
        }
        PROFILE_STOP( "testGeometry-surfaceID " + name );
    }
    // Finished with all tests
    if ( pass )
        ut.passes( "testGeometry: " + name );
}


} // namespace AMP::Geometry