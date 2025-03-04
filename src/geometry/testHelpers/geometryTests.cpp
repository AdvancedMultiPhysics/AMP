#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/mesh/MeshUtilities.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cmath>
#include <random>


namespace AMP::Geometry {


// Generate a random direction
static inline Point genRandDir( int ndim )
{
    static std::random_device rd;
    static std::mt19937 gen( rd() );
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
    while ( std::abs( d ) < 1e100 ) {
        d2 += std::abs( d );
        surfacePoints.push_back( x0 + d2 * dir );
        d2 += 1e-6;
        d = geom.distance( x0 + d2 * dir, dir );
        ++it;
        if ( it > 100 )
            AMP_ERROR( "Infinite surfaces" );
    }
    return inside;
}


// Generate points in a box
static std::vector<Point>
generatePoints( size_t N, int ndim, const Point &lb, const Point &ub, size_t seed = 0 )
{
    std::vector<Point> p( N, Point( ndim, { 0, 0, 0 } ) );
    std::mt19937 gen;
    if ( seed == 0 ) {
        std::random_device rd;
        gen = std::mt19937( rd() );
    } else {
        gen = std::mt19937( seed );
    }
    for ( int d = 0; d < ndim; d++ ) {
        std::uniform_real_distribution<> dis( lb[d], ub[d] );
        for ( size_t i = 0; i < N; i++ )
            p[i][d] = dis( gen );
    }
    return p;
}


// Generate random points in the box containing the geometry
static std::vector<Point> generatePoints( size_t N, const AMP::Geometry::Geometry &geom )
{
    auto logicalGeom = dynamic_cast<const AMP::Geometry::LogicalGeometry *>( &geom );
    const auto ndim  = geom.getDim();
    auto [lb, ub]    = geom.box();
    if ( logicalGeom ) {
        if ( ndim == logicalGeom->getLogicalDim() ) {
            // All physical points should map
            return generatePoints( N, ndim, lb, ub );
        } else {
            // Not all physical points can map, choose points in the logical domain
            auto p = generatePoints( N, logicalGeom->getLogicalDim(), { 0, 0, 0 }, { 1, 1, 1 } );
            for ( size_t i = 0; i < N; i++ )
                p[i] = logicalGeom->physical( p[i] );
            return p;
        }
    }
    return generatePoints( N, ndim, lb, ub );
}


// Generate points inside the geometry
static std::vector<Point> generateInteriorPoints( size_t N, const AMP::Geometry::Geometry &geom )
{
    const auto ndim  = geom.getDim();
    auto [lb, ub]    = geom.box();
    auto logicalGeom = dynamic_cast<const AMP::Geometry::LogicalGeometry *>( &geom );
    if ( ndim == static_cast<int>( geom.getGeomType() ) ) {
        // Estimate the centroid by randomly sampling space and test if it is inside the object
        // Note we use a non-random seed to ensure test doesn't fail periodically due to tolerances
        std::vector<Point> p;
        size_t seed = 56871;
        while ( p.size() < N ) {
            auto p2 = generatePoints( N, ndim, lb, ub, seed );
            for ( size_t i = 0; i < N && p.size() < N; i++ ) {
                if ( geom.inside( p2[i] ) )
                    p.push_back( p2[i] );
            }
            seed++;
        }
        return p;
    } else if ( logicalGeom ) {
        // Generate points in the logical domain and map to physical
        // Note this is less reliable than the first method because it assume the
        //    the mapping from logical to physical maintains the approximate same volume
        auto p = generatePoints( N, logicalGeom->getLogicalDim(), { 0, 0, 0 }, { 1, 1, 1 } );
        for ( size_t i = 0; i < p.size(); i++ )
            p[i] = logicalGeom->physical( p[i] );
        return p;
    } else {
        // Unable to generate points
        AMP_WARN_ONCE( "Unable to generate points in geometry (not finished): " + geom.getName() );
    }
    return {};
}


// Run logical geometry specific tests
static bool testLogicalGeometry( const AMP::Geometry::LogicalGeometry &geom, AMP::UnitTest &ut )
{
    bool pass = true;
    auto name = geom.getName();
    int ndim  = geom.getDim();
    // Test logical/physical transformations
    PROFILE2( "testGeometry-logical " + name );
    size_t N   = 10000;
    auto p     = generatePoints( N, geom );
    bool pass2 = true;
    for ( size_t i = 0; i < N; i++ ) {
        auto p2 = geom.logical( p[i] );
        auto p3 = geom.physical( p2 );
        for ( int d = 0; d < ndim; d++ ) {
            pass2 = pass2 && std::abs( p[i][d] - p3[d] ) < 1e-8;
            if ( std::abs( p[i][d] - p3[d] ) > 1e-8 ) {
                geom.logical( p[i] );
                geom.physical( p2 );
            }
        }
    }
    pass = pass && pass2;
    if ( !pass2 )
        ut.failure( "testGeometry physical-logical-physical: " + name );
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
    // Estimate the centroid by randomly points in object
    size_t N = 100000;
    auto p   = generateInteriorPoints( N, geom );
    if ( p.empty() ) {
        ut.failure( "testGeometry-centroid did not generate interior points: " + name );
        return false;
    }
    Point c( ndim, { 0, 0, 0 } );
    for ( size_t i = 0; i < p.size(); i++ )
        c += p[i];
    c *= 1.0 / p.size();
    double err = 0;
    for ( int d = 0; d < ndim; d++ ) {
        double dx = geom.box().second[d] - geom.box().first[d];
        err       = std::max( err, std::abs( c[d] - centroid[d] ) / dx );
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
        for ( const auto &geom2 : multigeom->getGeometries() )
            testGeometry( *geom2, ut );
    }
    // Get the physical dimension
    int ndim  = geom.getDim();
    auto name = geom.getName();
    PROFILE2( "testGeometry " + name );
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
    std::vector<Point> surfacePoints;
    {
        PROFILE( "testGeometry-surface " );
        size_t N = 10000;
        surfacePoints.reserve( N );
        size_t N_missed = 0;
        while ( surfacePoints.size() < N ) {
            auto dir  = genRandDir( ndim );
            bool test = addSurfacePoints( geom, center, dir, surfacePoints );
            if ( !test )
                N_missed++;
        }
        pass = pass && !surfacePoints.empty();
        if ( surfacePoints.empty() )
            ut.failure( "testGeometry unable to get surface: " + name );
        if ( geom.inside( center ) && N_missed != 0 ) {
            auto msg = AMP::Utilities::stringf(
                "testGeometry failed all rays hit (%i): %s", N_missed, name.data() );
            if ( N_missed < 10 )
                ut.expected_failure( msg );
            else if ( name == "MultiGeometry" )
                ut.expected_failure( msg );
            else
                ut.failure( msg );
        }
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
    }
    // Verify each surface point is "inside" the object
    {
        PROFILE( "testGeometry-inside" );
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
    }
    // Project each surface point in a random direction and back propagate to get the same point
    {
        PROFILE( "testGeometry-distance" );
        auto box        = geom.box();
        auto length     = box.second - box.first;
        const double d0 = 0.2 * std::max( { length.x(), length.y(), length.z() } );
        int N_failed    = 0;
        int N_repeat    = 10;
        for ( const auto &tmp : surfacePoints ) {
            for ( int i = 0; i < N_repeat; i++ ) {
                auto ang = genRandDir( ndim );
                auto pos = tmp - d0 * ang;
                double d = std::abs( geom.distance( pos, ang ) );
                for ( int it = 0; it < 1000 && d < d0 - 1e-5; it++ ) {
                    // We may have crossed multiple surfaces, find the original
                    d += 1e-6;
                    auto pos2 = pos + d * ang;
                    d += std::abs( geom.distance( pos2, ang ) );
                }
                if ( std::abs( d - d0 ) > 1e-5 )
                    N_failed++;
            }
        }
        if ( N_failed > 0 ) {
            using AMP::Utilities::stringf;
            int N_test = N_repeat * surfacePoints.size();
            auto msg   = stringf( "testGeometry distances do not match (%i of %i): %s",
                                N_failed,
                                N_test,
                                name.data() );
            if ( N_failed > 0.001 * N_test ) {
                ut.failure( msg );
                pass = false;
            } else if ( N_failed > 0 ) {
                ut.expected_failure( msg );
            }
        }
    }
    // Get a set of interior points by randomly sampling the space
    // Note we use a non-random seed to ensure test doesn't fail periodically due to tolerances
    std::vector<Point> interiorPoints;
    {
        PROFILE( "testGeometry-sample" );
        auto box = geom.box();
        std::mt19937 gen( 84397 );
        std::uniform_real_distribution<double> dist[3];
        for ( int d = 0; d < ndim; d++ )
            dist[d] = std::uniform_real_distribution<double>( box.first[d], box.second[d] );
        for ( int i = 0; i < 10000; i++ ) {
            Point pos( ndim, { 0, 0, 0 } );
            for ( int d = 0; d < ndim; d++ )
                pos[d] = dist[d]( gen );
            if ( geom.inside( pos ) )
                interiorPoints.push_back( pos );
        }
    }
    // Check that nearest returns the surface/interior points
    {
        PROFILE( "testGeometry-nearest " );
        bool pass_nearest = true;
        for ( const auto &p0 : surfacePoints ) {
            [[maybe_unused]] auto p   = geom.nearest( p0 );
            [[maybe_unused]] double d = ( p - p0 ).abs();
            if ( d > 1e-8 ) {
                [[maybe_unused]] bool test = geom.inside( p0 );
                p                          = geom.nearest( p0 );
                pass_nearest               = false;
            }
        }
        for ( const auto &p0 : interiorPoints ) {
            [[maybe_unused]] auto p   = geom.nearest( p0 );
            [[maybe_unused]] double d = ( p - p0 ).abs();
            if ( d > 1e-8 ) {
                p            = geom.nearest( p0 );
                pass_nearest = false;
            }
        }
        pass = pass && pass_nearest;
        if ( !pass_nearest )
            ut.failure( "testGeometry-nearest: " + name );
    }
    // Test getting surface normals
    if ( !multigeom ) {
        bool passNorm = true;
        for ( const auto &p : surfacePoints ) {
            auto norm = geom.surfaceNorm( p );
            double n = std::sqrt( norm.x() * norm.x() + norm.y() * norm.y() + norm.z() * norm.z() );
            // auto p1   = p - 1e-5 * norm;
            // auto p2   = p + 1e-5 * norm;
            passNorm = passNorm && std::abs( n - 1.0 ) < 1e-6;
            // passNorm  = passNorm && geom.inside( p1 ) && !geom.inside( p2 );
        }
        pass = pass && passNorm;
        if ( !passNorm )
            ut.failure( "testGeometry surfaceNorm: " + name );
    }
    // Test getting the volume
    {
        PROFILE( "testGeometry-volume " );
        auto box         = geom.box();
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
            bool passVol2 = std::abs( vol2 - volume ) < 0.01 * volume;
            pass          = pass && passVol2;
            if ( !passVol2 )
                ut.failure( "testGeometry volumeOverlap: " + name );
        }
    }
    // Test getting the surface id
    if ( !multigeom ) {
        PROFILE( "testGeometry-surfaceID" );
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
    }
    // Finished with all tests
    if ( pass )
        ut.passes( "testGeometry: " + name );
}


} // namespace AMP::Geometry
