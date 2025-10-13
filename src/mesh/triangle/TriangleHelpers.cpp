#include "AMP/mesh/triangle/TriangleHelpers.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/geometry/MeshGeometry.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/geometry/shapes/Circle.h"
#include "AMP/geometry/shapes/Sphere.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MeshUtilities.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/triangle/TriangleMesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/DelaunayTessellation.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/kdtree.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <map>
#include <random>


namespace AMP::Mesh::TriangleHelpers {


// Factorial
static inline size_t factorial( int x )
{
    if ( x <= 1 )
        return 1;
    size_t y  = 0;
    size_t x2 = x;
    for ( size_t i = 2; i <= x2; i++ )
        y *= x;
    return y;
}


// Helper function to create null triangles
template<size_t N>
static constexpr std::array<int, N + 1> createNullTri()
{
    std::array<int, N + 1> null = { -1 };
    for ( size_t i = 0; i < null.size(); i++ )
        null[i] = -1;
    return null;
}


// Helper function to wrap fread
static inline void fread2( void *ptr, size_t size, size_t count, FILE *stream )
{
    size_t N = fread( ptr, size, count, stream );
    AMP_ASSERT( N == count );
}


// Helper functions to see if two points are ~ the same
template<size_t N>
static inline bool approx_equal( const std::array<double, N> &x,
                                 const std::array<double, N> &y,
                                 const std::array<double, N> &tol )
{
    if constexpr ( N == 1 )
        return fabs( x[0] - y[0] ) <= tol[0];
    else if constexpr ( N == 2 )
        return fabs( x[0] - y[0] ) <= tol[0] && fabs( x[1] - y[1] ) <= tol[1];
    else if constexpr ( N == 3 )
        return fabs( x[0] - y[0] ) <= tol[0] && fabs( x[1] - y[1] ) <= tol[1] &&
               fabs( x[2] - y[2] ) <= tol[2];
    return false;
}


/****************************************************************
 * Count the number of unique triangles                          *
 ****************************************************************/
template<size_t NDIM, bool ordered>
static uint64_t hash( const std::array<int, NDIM> &x )
{
    uint64_t hash = 0;
    for ( size_t i = 0; i < NDIM; i++ ) {
        // Use hashing function: 2^64*0.5*(sqrt(5)-1)
        uint64_t z = static_cast<uint64_t>( x[i] ) * 0x9E3779B97F4A7C15;
        if constexpr ( ordered ) {
            hash = ( ( hash << 5 ) + hash ) ^ z;
        } else {
            hash = hash ^ z;
        }
    }
    return hash;
}
template<size_t NG>
size_t count( const std::vector<std::array<int, NG + 1>> &tri )
{
    std::vector<uint64_t> x( tri.size(), 0 );
    for ( size_t i = 0; i < tri.size(); i++ )
        x[i] = hash<NG + 1, false>( tri[i] );
    std::sort( x.begin(), x.end() );
    x.erase( std::unique( x.begin(), x.end() ), x.end() );
    return x.size();
}


/****************************************************************
 * Read stl file                                                 *
 ****************************************************************/
size_t readSTLHeader( const std::string &filename )
{
    char header[80];
    uint32_t N;
    auto fid = fopen( filename.c_str(), "rb" );
    AMP_INSIST( fid, "Unable to open " + filename );
    fread2( header, sizeof( header ), 1, fid );
    fread2( &N, sizeof( N ), 1, fid );
    fclose( fid );
    return N;
}

std::vector<std::array<std::array<double, 3>, 3>> readSTL( const std::string &filename,
                                                           double scale )
{
    char header[80];
    uint32_t N;
    // Read the file
    auto fid = fopen( filename.c_str(), "rb" );
    AMP_INSIST( fid, "Unable to open " + filename );
    fread2( header, sizeof( header ), 1, fid );
    fread2( &N, sizeof( N ), 1, fid );
    auto tmp = new char[N * 50];
    fread2( tmp, 50, N, fid );
    fclose( fid );
    // Get a list of the local triangles based on their coordinates
    std::vector<std::array<std::array<double, 3>, 3>> tri_coord( N );
    for ( size_t i = 0; i < N; i++ ) {
        [[maybe_unused]] uint16_t attrib    = 0;
        [[maybe_unused]] float normal[3]    = { 0, 0, 0 };
        [[maybe_unused]] float vertex[3][3] = { { 0 } };
        memcpy( normal, &tmp[50 * i], sizeof( normal ) );
        memcpy( vertex, &tmp[50 * i + 12], sizeof( vertex ) );
        memcpy( &attrib, &tmp[50 * i + 48], sizeof( attrib ) );
        tri_coord[i][0][0] = scale * vertex[0][0];
        tri_coord[i][0][1] = scale * vertex[0][1];
        tri_coord[i][0][2] = scale * vertex[0][2];
        tri_coord[i][1][0] = scale * vertex[1][0];
        tri_coord[i][1][1] = scale * vertex[1][1];
        tri_coord[i][1][2] = scale * vertex[1][2];
        tri_coord[i][2][0] = scale * vertex[2][0];
        tri_coord[i][2][1] = scale * vertex[2][1];
        tri_coord[i][2][2] = scale * vertex[2][2];
    }
    delete[] tmp;
    return tri_coord;
}


/****************************************************************
 * Create triangles/vertices from a set of triangles specified  *
 * by their coordinates                                          *
 ****************************************************************/
template<size_t NG, size_t NP>
void createTriangles( const std::vector<std::array<std::array<double, NP>, NG + 1>> &tri_list,
                      std::vector<std::array<double, NP>> &vertices,
                      std::vector<std::array<int, NG + 1>> &triangles,
                      double tol )
{
    // Get the range of points and tolerance to use
    std::array<double, 2 * NP> range;
    for ( size_t d = 0; d < NP; d++ ) {
        range[2 * d + 0] = tri_list[0][0][d];
        range[2 * d + 1] = tri_list[0][0][d];
    }
    for ( const auto &tri : tri_list ) {
        for ( const auto &point : tri ) {
            for ( size_t d = 0; d < NP; d++ ) {
                range[2 * d + 0] = std::min( range[2 * d + 0], point[d] );
                range[2 * d + 1] = std::max( range[2 * d + 1], point[d] );
            }
        }
    }
    std::array<double, NP> tol2;
    for ( size_t d = 0; d < NP; d++ )
        tol2[d] = tol * ( range[2 * d + 1] - range[2 * d + 0] );
    // Get the unique vertices and create triangle indicies
    vertices.clear();
    triangles.clear();
    triangles.resize( tri_list.size(), createNullTri<NG>() );
    for ( size_t i = 0; i < tri_list.size(); i++ ) {
        for ( size_t j = 0; j < NG + 1; j++ ) {
            auto &point   = tri_list[i][j];
            int64_t index = -1;
            for ( size_t k = 0; k < vertices.size() && index == -1; k++ ) {
                if ( approx_equal( point, vertices[k], tol2 ) )
                    index = k;
            }
            if ( index == -1 ) {
                index = vertices.size();
                vertices.push_back( point );
            }
            triangles[i][j] = index;
        }
    }
}


/****************************************************************
 * Create triangles/vertices from a set of triangles specified  *
 * by their coordinates                                          *
 ****************************************************************/
static inline std::array<double, 3> calcNorm( const std::vector<std::array<double, 3>> &x,
                                              const std::array<int, 3> &tri )
{
    return AMP::Geometry::GeometryHelpers::normal( x[tri[0]], x[tri[1]], x[tri[2]] );
}
static inline double dot( const std::array<double, 3> &x, const std::array<double, 3> &y )
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}
template<size_t NG, size_t NP>
static std::vector<int> createBlockIDs( const std::vector<std::array<double, NP>> &vertices,
                                        const std::vector<std::array<int, NG + 1>> &tri,
                                        const std::vector<std::array<int, NG + 1>> &tri_nab )
{
    if ( tri.empty() )
        return std::vector<int>();
    // Calculate the normal for each triangle face
    using Point = std::array<double, NP>;
    std::vector<Point> norm( tri.size() );
    for ( size_t i = 0; i < tri.size(); i++ )
        norm[i] = calcNorm( vertices, tri[i] );
    // Identify different blocks by the change in the normal
    int nextBlockID = 0;
    std::vector<int> blockID( tri.size(), -1 );
    std::vector<bool> finished( tri.size(), false );
    std::set<size_t> queued;
    double tol = 0.1;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        if ( finished[i] )
            continue;
        blockID[i] = nextBlockID++;
        queued.insert( i );
        while ( !queued.empty() ) {
            auto it  = queued.begin();
            size_t j = *it;
            queued.erase( it );
            finished[j] = true;
            for ( auto k : tri_nab[j] ) {
                if ( k == -1 )
                    continue; // There is no neighbor
                if ( finished[k] )
                    continue; // We already examined this neighbor
                auto theta = acos( dot( norm[j], norm[k] ) );
                if ( theta <= tol ) {
                    // The norm is within tol, set the block id
                    blockID[k] = blockID[j];
                    queued.insert( k );
                }
            }
        }
    }
    return blockID;
}


/****************************************************************
 * Try to split the mesh into seperate independent domains       *
 ****************************************************************/
static inline std::array<int, 2> getFace( const std::array<int, 3> &tri, size_t i )
{
    return { tri[( i + 1 ) % 3], tri[( i + 2 ) % 3] };
}
static inline std::array<int, 3> getFace( const std::array<int, 4> &tri, size_t i )
{
    return { tri[( i + 1 ) % 4], tri[( i + 2 ) % 4], tri[( i + 3 ) % 4] };
}
template<size_t NG>
static void addFaces( const std::array<int, NG + 1> &tri,
                      int64_t index,
                      std::vector<std::pair<uint64_t, int64_t>> &faces )
{
    for ( size_t i = 0; i <= NG; i++ ) {
        // Get each face of the triangle
        auto face = getFace( tri, i );
        auto id1  = hash<NG, true>( face );
        // Reverse the order
        std::reverse( face.begin(), face.end() );
        auto id2   = hash<NG, true>( face );
        bool found = false;
        size_t k   = 0;
        for ( size_t j = 0; j < faces.size(); j++ ) {
            if ( faces[j].first == id1 )
                found = true;
            else
                faces[k++] = faces[j];
        }
        faces.resize( k );
        if ( !found ) {
            // Face does not exist, add it
            int64_t tmp = ( index << 4 ) + i;
            faces.push_back( std::make_pair( id2, tmp ) );
        }
    }
}
template<class TYPE>
static inline void erase( TYPE &faceMap, int64_t i )
{
    for ( auto it = faceMap.begin(); it != faceMap.end(); ) {
        if ( it->second >> 4 == i )
            it = faceMap.erase( it );
        else
            ++it;
    }
}
template<size_t NG>
static std::vector<std::array<int, NG + 1>>
removeSubDomain( std::vector<std::array<int, NG + 1>> &tri )
{
    // For each triangle get a hash id for each face
    std::multimap<uint64_t, int64_t> faceMap;
    for ( size_t i = 0, k = 0; i < tri.size(); i++ ) {
        for ( size_t j = 0; j <= NG; j++, k++ ) {
            auto face   = getFace( tri[i], j );
            uint64_t id = hash<NG, true>( face );
            int64_t tmp = ( i << 4 ) + j;
            faceMap.insert( std::make_pair( id, tmp ) );
        }
    }
    // Choose an initial triangle
    size_t i0 = 0;
    int count = 100;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        int Nf_max = 0;
        for ( size_t j = 0; j <= NG; j++ ) {
            // Get each face of the triangle
            auto face = getFace( tri[i], j );
            // auto id   = hash<NG, true>( face );
            // Reverse the order
            std::reverse( face.begin(), face.end() );
            auto id2 = hash<NG, true>( face );
            // Get the number of matching faces
            int Nf = faceMap.count( id2 );
            Nf_max = std::max( Nf_max, Nf );
        }
        if ( Nf_max < count ) {
            count = Nf_max;
            i0    = i;
        }
    }
    // Add the initial triangle store the edges
    std::vector<bool> used( tri.size(), false );
    std::vector<std::array<int, NG + 1>> tri2;
    std::vector<std::pair<uint64_t, int64_t>> faces;
    used[i0] = true;
    tri2.push_back( tri[i0] );
    addFaces<NG>( tri[i0], i0, faces );
    erase( faceMap, i0 );
    // Add triangles until all faces have been filled
    while ( !faces.empty() ) {
        bool found = false;
        for ( size_t i = 0; i < faces.size(); i++ ) {
            int Nf = faceMap.count( faces[i].first );
            AMP_ASSERT( Nf > 0 );
            if ( Nf == 1 ) {
                // We are dealing with a unique match, add the triangle
                auto it = faceMap.find( faces[i].first );
                int j   = it->second >> 4;
                used[j] = true;
                tri2.push_back( tri[j] );
                addFaces<NG>( tri[j], j, faces );
                erase( faceMap, j );
                found = true;
                break;
            }
        }
        if ( found )
            continue;
        // We have multiple faces to choose from, try to remove a subdomain from the remaining faces
        try {
            std::vector<std::array<int, NG + 1>> tri3;
            for ( size_t j = 0; j < tri.size(); j++ ) {
                if ( !used[j] )
                    tri3.push_back( tri[j] );
            }
            auto tri4 = removeSubDomain<NG>( tri3 );
            for ( const auto t : tri2 )
                tri3.push_back( t );
            std::swap( tri, tri3 );
            return tri4;
        } catch ( ... ) {
        }
        // Still no luck
        AMP_ERROR( "Unable to resolve multiple faces" );
    }
    // Remove triangles that were used
    size_t k = 0;
    for ( size_t j = 0; j < tri.size(); j++ ) {
        if ( !used[j] )
            tri[k++] = tri[j];
    }
    tri.resize( k );
    return tri2;
}
template<size_t NG>
std::vector<std::vector<std::array<int, NG + 1>>>
splitDomains( std::vector<std::array<int, NG + 1>> tri )
{
    std::vector<std::vector<std::array<int, NG + 1>>> tri_sets;
    while ( !tri.empty() ) {
        tri_sets.emplace_back( removeSubDomain<NG>( tri ) );
    }
    return tri_sets;
}
template<>
std::vector<std::vector<std::array<int, 2>>> splitDomains<1>( std::vector<std::array<int, 2>> )
{
    AMP_ERROR( "1D splitting of domains is not supported" );
    return std::vector<std::vector<std::array<int, 2>>>();
}


/********************************************************
 *  Generate mesh for STL file                           *
 ********************************************************/
using triset = std::vector<std::array<int, 3>>;
static std::vector<AMP::AMP_MPI>
loadbalance( const std::vector<triset> &tri, const AMP_MPI &comm, int method )
{
    // Check if we are not load-balancing the meshes
    if ( method == 0 || comm.getSize() == 1 )
        return std::vector<AMP::AMP_MPI>( tri.size(), comm );
    // Perform the load balance
    std::vector<loadBalanceSimulator> list( tri.size() );
    for ( size_t i = 0; i < tri.size(); i++ )
        list[i] = loadBalanceSimulator( tri[i].size() );
    loadBalanceSimulator mesh( list, 1 );
    mesh.setProcs( comm.getSize() );
    // Create the communicators
    std::vector<AMP::AMP_MPI> comm2( tri.size(), AMP_COMM_NULL );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        bool found = false;
        for ( int r : mesh.getRanks( i ) )
            found = found || r == comm.getRank();
        int key  = found ? 1 : -1;
        comm2[i] = comm.split( comm.getRank(), key );
    }
    return comm2;
}
std::shared_ptr<AMP::Mesh::Mesh> generateSTL( std::shared_ptr<const MeshParameters> params )
{
    auto db       = params->getDatabase();
    auto filename = db->getWithDefault<std::string>( "FileName", "" );
    auto name     = db->getWithDefault<std::string>( "MeshName", "NULL" );
    auto comm     = params->getComm();
    // Read the STL file
    std::vector<std::array<double, 3>> vert;
    std::vector<triset> tri( 1 ), tri_nab;
    if ( comm.getRank() == 0 ) {
        auto scale     = db->getWithDefault<double>( "scale", 1.0 );
        auto triangles = TriangleHelpers::readSTL( filename, scale );
        // Create triangles from the points
        double tol = 1e-6;
        TriangleHelpers::createTriangles<2, 3>( triangles, vert, tri[0], tol );
        // Find the number of unique triangles (duplicates may indicate multiple objects)
        bool multidomain = TriangleHelpers::count<2>( tri[0] ) > 1;
        if ( multidomain && db->getWithDefault<bool>( "split", true ) ) {
            // Try to split the domains
            tri         = TriangleHelpers::splitDomains<2>( tri[0] );
            multidomain = false;
        }
        // Create the triangle neighbors
        tri_nab.resize( tri.size() );
        if ( !multidomain ) {
            for ( size_t i = 0; i < tri.size(); i++ ) {
                // Get the triangle neighbors
                tri_nab[i] = DelaunayHelpers::create_tri_neighbors<2>( tri[i] );
                // Check if the geometry is closed
                bool closed = true;
                for ( const auto &t : tri_nab[i] ) {
                    for ( const auto &p : t )
                        closed = closed && p >= 0;
                }
                if ( !closed )
                    AMP_WARNING( "Geometry is not closed" );
            }
        } else {
            AMP_WARNING( "Not splitting multi-domain, no neighbor info will be created" );
            for ( size_t i = 0; i < tri.size(); i++ ) {
                tri_nab[i].resize( tri[i].size() );
                for ( auto &t : tri_nab[i] )
                    t.fill( -1 );
            }
        }
    }
    int N_domains = comm.bcast( tri.size(), 0 );
    tri.resize( N_domains );
    tri_nab.resize( N_domains );
    // Create the mesh
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    if ( N_domains == 1 ) {
        auto id = createBlockIDs<2, 3>( vert, tri[0], tri_nab[0] );
        mesh    = TriangleMesh<2>::generate<3>( vert, tri[0], tri_nab[0], comm, nullptr, id );
    } else {
        // We are dealing with multiple sub-domains, choose the load balance method
        int method = db->getWithDefault<int>( "LoadBalanceMethod", 1 );
        auto comm2 = loadbalance( tri, comm, method );
        // Send the triangle data to all ranks
        vert = comm.bcast( std::move( vert ), 0 );
        tri.resize( N_domains );
        tri_nab.resize( N_domains );
        for ( int i = 0; i < N_domains; i++ ) {
            tri[i]     = comm.bcast( std::move( tri[i] ), 0 );
            tri_nab[i] = comm.bcast( std::move( tri_nab[i] ), 0 );
        }
        // Build the sub meshes
        std::vector<std::shared_ptr<AMP::Mesh::Mesh>> submeshes;
        for ( size_t i = 0; i < tri.size(); i++ ) {
            if ( !comm2[i].isNull() ) {
                std::shared_ptr<AMP::Mesh::Mesh> mesh2;
                if ( comm2[i].getRank() == 0 ) {
                    auto id = createBlockIDs<2, 3>( vert, tri[i], tri_nab[i] );
                    mesh2   = TriangleMesh<2>::generate<3>(
                        vert, tri[i], tri_nab[i], comm2[i], nullptr, id );
                } else {
                    mesh2 = TriangleMesh<2>::generate<3>( {}, {}, {}, comm2[i], nullptr, {} );
                }
                mesh2->setName( name + "_" + std::to_string( i + 1 ) );
                submeshes.push_back( mesh2 );
            }
        }
        mesh.reset( new MultiMesh( name, comm, submeshes ) );
    }
    // Displace the mesh
    std::vector<double> disp( 3, 0.0 );
    if ( db->keyExists( "x_offset" ) )
        disp[0] = db->getScalar<double>( "x_offset" );
    if ( db->keyExists( "y_offset" ) )
        disp[1] = db->getScalar<double>( "y_offset" );
    if ( db->keyExists( "z_offset" ) )
        disp[2] = db->getScalar<double>( "z_offset" );
    if ( disp[0] != 0.0 || disp[1] != 0.0 || disp[2] != 0.0 )
        mesh->displaceMesh( disp );
    // Set the mesh name
    mesh->setName( name );
    return mesh;
}


/********************************************************
 *  Generate mesh for geometry                           *
 ********************************************************/
static inline void check_nearest( const std::vector<Point> &x )
{
    if ( x.empty() )
        return;
    auto index = find_min_dist( x );
    auto dx    = x[index.first] - x[index.second];
    double d   = dx.abs();
    AMP_ASSERT( d > 1e-8 );
}
template<uint8_t NDIM>
static inline void checkTri( const std::vector<std::array<int, NDIM + 1>> &tri )
{
    AMP_ASSERT( !tri.empty() );
    // Check the triangles for duplicate nodes within a triangle
    for ( const auto &tri2 : tri ) {
        for ( size_t i = 0; i <= NDIM; i++ ) {
            for ( size_t j = 0; j < i; j++ ) {
                AMP_ASSERT( tri2[i] != -1 );
                AMP_ASSERT( tri2[i] != tri2[j] );
            }
        }
    }
}
static inline std::vector<Point> getVolumePoints( const AMP::Geometry::Geometry &geom,
                                                  double resolution )
{
    // Create interior points from an arbitrary geometry
    // Note: we can adjust the points so that they are not aligned
    //    on xyz grids which may help with the tessellation
    std::vector<Point> points;
    auto [x0, x1] = geom.box();
    // auto xc = 0.7 * ( x0 + x1 );
    // double tol = 0.7 * resolution / ( x1 - x0 ).abs();
    for ( double x = x0.x(); x <= x1.x(); x += resolution ) {
        for ( double y = x0.y(); y <= x1.y(); y += resolution ) {
            for ( double z = x0.z(); z <= x1.z(); z += resolution ) {
                Point p( x0.size(), { x, y, z } );
                // auto p2 = p - xc;
                // p -= ( tol * p2.abs() ) * p2;
                if ( geom.inside( p ) )
                    points.push_back( p );
            }
        }
    }
    return points;
}
static inline std::vector<Point> getSurfacePoints( const AMP::Geometry::Geometry &geom, int N )
{
    // Create surface points for an arbitrary geometry
    std::vector<Point> points;
    const int ndim      = geom.getDim();
    const auto [x1, x2] = geom.box();
    const auto x0       = 0.5 * ( x1 + x2 );
    const auto dx       = x2 - x1;
    if ( ndim == 3 ) {
        double r = std::sqrt( dx.x() * dx.x() + dx.y() * dx.y() + dx.z() * dx.z() );
        int n    = ceil( std::sqrt( N ) );
        for ( int i = 0; i < n; i++ ) {
            double x = ( 0.5 + i ) / (double) n;
            for ( int j = 0; j < n; j++ ) {
                double y = ( 0.5 + j ) / (double) n;
                Point s  = AMP::Geometry::GeometryHelpers::map_logical_sphere_surface( 1, r, x, y );
                auto dir = normalize( s );
                double d = geom.distance( x0, dir );
                AMP_ASSERT( d < 0 );
                points.push_back( x0 - d * dir );
            }
        }
    } else {
        AMP_ERROR( "Not finished" );
    }
    check_nearest( points );
    return points;
}
static inline double getPos( int i, int N, bool isPeriodic )
{
    if ( N <= 1 )
        return 0;
    if ( isPeriodic )
        return ( i + 0.5 ) / N;
    return i / ( N - 1.0 );
}
static inline std::vector<Point> getLogicalPoints( const AMP::Geometry::LogicalGeometry &geom,
                                                   double resolution )
{
    // Create surface/interior points for a logical geometry
    std::vector<Point> points;
    int ndim      = geom.getDim();
    auto N        = geom.getLogicalGridSize( std::vector<double>( ndim, resolution ) );
    auto periodic = geom.getPeriodicDim();
    for ( size_t i = 0; i < N[0]; i++ ) {
        double x = getPos( i, N[0], periodic[0] );
        for ( size_t j = 0; j < N[1]; j++ ) {
            double y = getPos( j, N[1], periodic[1] );
            for ( size_t k = 0; k < N[2]; k++ ) {
                double z = getPos( k, N[2], periodic[2] );
                Point p  = geom.physical( { x, y, z } );
                points.push_back( p );
            }
        }
    }
    return points;
}
static inline std::vector<Point> combineSurfaceVolumePoints( const std::vector<Point> &volume,
                                                             const std::vector<Point> &surface,
                                                             const AMP::Geometry::Geometry &geom,
                                                             double resolution )
{
    // Add the points in the volume
    std::vector<Point> points = volume;
    // Remove volume points that are close to the surface
    size_t k    = 0;
    double tol  = 0.6 * resolution;
    double tol2 = tol * tol;
    for ( size_t i = 0; i < points.size(); i++ ) {
        auto ps   = geom.nearest( points[i] );
        double d2 = ( points[i] - ps ).norm();
        if ( d2 >= tol2 )
            points[k++] = points[i];
    }
    // Add the surface points
    points.resize( k );
    for ( const auto &s : surface )
        points.push_back( s );
    // Check the distance
    check_nearest( points );
    return points;
}
template<uint8_t NDIM>
static void removeTriangles( std::vector<std::array<int, NDIM + 1>> &tri,
                             std::vector<std::array<int, NDIM + 1>> &tri_nab,
                             const std::vector<bool> &remove )
{
    std::vector<int> map( tri.size(), -1 );
    size_t N = 0;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        if ( !remove[i] )
            map[i] = N++;
    }
    if ( N == tri.size() )
        return;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        if ( !remove[i] ) {
            tri[map[i]] = tri[i];
            for ( int d = 0; d <= NDIM; d++ ) {
                if ( tri_nab[i][d] != -1 ) {
                    if ( remove[tri_nab[i][d]] )
                        tri_nab[i][d] = -1;
                    else
                        tri_nab[i][d] = map[tri_nab[i][d]];
                }
            }
            tri_nab[map[i]] = tri_nab[i];
        }
    }
    tri.resize( N );
    tri_nab.resize( N );
}
template<uint8_t NDIM>
static std::tuple<std::vector<std::array<int, NDIM + 1>>, std::vector<std::array<int, NDIM + 1>>>
createTessellation( const std::vector<Point> &points )
{
    // Convert coordinates
    int N = points.size();
    AMP::Array<double> x1( NDIM, points.size() );
    for ( int i = 0; i < N; i++ ) {
        for ( int d = 0; d < NDIM; d++ )
            x1( d, i ) = points[i][d];
    }
    // Tessellate
    auto [tri, nab] = DelaunayTessellation::create_tessellation( x1 );
    // Convert to output format
    constexpr auto nullTri = createNullTri<NDIM>();
    std::vector<std::array<int, NDIM + 1>> tri2( tri.size( 1 ), nullTri );
    std::vector<std::array<int, NDIM + 1>> nab2( tri.size( 1 ), nullTri );
    for ( size_t i = 0; i < tri2.size(); i++ ) {
        for ( size_t d = 0; d <= NDIM; d++ ) {
            tri2[i][d] = tri( d, i );
            nab2[i][d] = nab( d, i );
        }
    }
    checkTri<NDIM>( tri2 );
    return std::tie( tri2, nab2 );
}
template<uint8_t NDIM>
static std::shared_ptr<AMP::Mesh::Mesh>
generateGeom2( std::shared_ptr<AMP::Geometry::Geometry> geom,
               const std::vector<Point> &points,
               const AMP_MPI &comm )
{
    if ( comm.getRank() != 0 )
        TriangleMesh<NDIM>::template generate<NDIM>( {}, {}, {}, comm, geom );
    // Tessellate
    auto [tri, tri_nab] = createTessellation<NDIM>( points );
    // Delete triangles that have duplicate neighbors
    {
        // Identify the triangles that need to be removed
        std::vector<bool> remove( tri.size(), false );
        for ( size_t i = 0; i < tri.size(); i++ ) {
            for ( int d = 0; d <= NDIM; d++ ) {
                if ( tri_nab[i][d] == -1 )
                    continue;
                int count = 0;
                for ( int d2 = 0; d2 <= NDIM; d2++ ) {
                    if ( tri_nab[i][d] == tri_nab[i][d2] )
                        count++;
                }
                if ( count != 1 )
                    remove[i] = true;
            }
        }
        // Remove the triangles
        removeTriangles<NDIM>( tri, tri_nab, remove );
    }
    // Delete surface triangles that have zero volume
    if constexpr ( NDIM == 3 ) {
        // Identify the triangles that need to be removed
        std::vector<bool> remove( tri.size(), false );
        for ( size_t i = 0; i < tri.size(); i++ ) {
            if ( tri_nab[i][0] >= 0 && tri_nab[i][1] >= 0 && tri_nab[i][2] >= 0 &&
                 tri_nab[i][3] >= 0 )
                continue;
            Point x[4];
            for ( int j = 0; j <= NDIM; j++ )
                x[j] = points[tri[i][j]];
            double M[9];
            for ( size_t k = 0; k < 3; k++ ) {
                for ( size_t d = 0; d < 3; d++ )
                    M[d + k * 3] = x[k][d] - x[3][d];
            }
            constexpr double C = 1.0 / 6.0;
            double V           = std::abs( C * DelaunayHelpers::det<double, NDIM>( M ) );
            remove[i]          = V < 1e-6;
        }
        // Remove the triangles
        removeTriangles<NDIM>( tri, tri_nab, remove );
    }
    // Try to remove triangles outside the domain
    bool isConvex = geom->isConvex();
    if ( !isConvex ) {
        AMP_WARNING( "non-convex domains are not fully supported yet" );
        // Identify the triangles that need to be removed
        std::vector<bool> remove( tri.size(), false );
        const double tmp = 1.0 / ( NDIM + 1.0 );
        for ( size_t i = 0; i < tri.size(); i++ ) {
            Point center( NDIM, { 0, 0, 0 } );
            for ( int j = 0; j <= NDIM; j++ )
                center += points[tri[i][j]];
            center *= tmp;
            remove[i] = !geom->inside( center );
        }
        // Remove the triangles
        removeTriangles<NDIM>( tri, tri_nab, remove );
    }
    checkTri<NDIM>( tri );
    // Generate the mesh
    std::vector<std::array<int, NDIM + 1>> tri2( tri.size() ), tri_nab2( tri.size() );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( int d = 0; d <= NDIM; d++ ) {
            tri2[i][d]     = tri[i][d];
            tri_nab2[i][d] = tri_nab[i][d];
        }
    }
    std::vector<std::array<double, NDIM>> x1( points.size() );
    for ( size_t i = 0; i < points.size(); i++ ) {
        for ( int d = 0; d < NDIM; d++ )
            x1[i][d] = points[i][d];
    }
    return TriangleMesh<NDIM>::template generate<NDIM>( x1, tri2, tri_nab2, comm, geom );
}
std::shared_ptr<AMP::Mesh::Mesh> generateGeom( std::shared_ptr<AMP::Geometry::Geometry> geom,
                                               const AMP_MPI &comm,
                                               double resolution )
{
    AMP_ASSERT( geom );
    auto multigeom = std::dynamic_pointer_cast<AMP::Geometry::MultiGeometry>( geom );
    if ( multigeom ) {
        std::vector<std::shared_ptr<AMP::Mesh::Mesh>> submeshes;
        for ( auto &geom2 : multigeom->getGeometries() )
            submeshes.push_back( generateGeom( geom2, comm, resolution ) );
        return std::make_shared<MultiMesh>( "name", comm, submeshes );
    }
    // Perform some basic checks
    int ndim         = geom->getDim();
    auto meshGeom    = std::dynamic_pointer_cast<AMP::Geometry::MeshGeometry>( geom );
    auto logicalGeom = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( geom );
    // Create the grid vertices
    std::vector<Point> points;
    if ( logicalGeom ) {
        // We are dealing with a logical geometry
        points = getLogicalPoints( *logicalGeom, resolution );
    } else if ( meshGeom ) {
        // Get the volume points
        auto interior = getVolumePoints( *geom, resolution );
        // Get the surface points
        auto &mesh   = meshGeom->getMesh();
        auto data    = sample( mesh, resolution );
        auto surface = std::get<0>( data );
        // Combine
        points = combineSurfaceVolumePoints( interior, surface, *geom, resolution );
    } else {
        // Get the volume points
        auto interior = getVolumePoints( *geom, resolution );
        // Get the surface points
        auto surface = getSurfacePoints( *geom, 0.1 * interior.size() );
        // Combine
        points = combineSurfaceVolumePoints( interior, surface, *geom, resolution );
    }
    // Smooth the points to try and make the distance between all points ~ equal

    // Tessellate and generate the mesh
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    if ( ndim == 2 ) {
        mesh = generateGeom2<2>( geom, points, comm );
    } else if ( ndim == 3 ) {
        mesh = generateGeom2<3>( geom, points, comm );
    } else {
        AMP_ERROR( "Not supported yet" );
    }
    if ( meshGeom )
        mesh->setName( meshGeom->getMesh().getName() );
    return mesh;
}
std::shared_ptr<AMP::Mesh::Mesh> generate( std::shared_ptr<const MeshParameters> params )
{
    auto db = params->getDatabase();
    if ( db->keyExists( "FileName" ) ) {
        auto filename = db->getWithDefault<std::string>( "FileName", "" );
        auto suffix   = IO::getSuffix( filename );
        if ( suffix == "stl" ) {
            // We are reading an stl file
            return generateSTL( params );
        } else {
            AMP_ERROR( "Unknown format for TriangleMesh" );
        }
    } else if ( db->keyExists( "Geometry" ) ) {
        // We will build a triangle mesh from a geometry
        auto geom_db   = db->getDatabase( "Geometry" );
        double dist[3] = { db->getWithDefault<double>( "x_offset", 0.0 ),
                           db->getWithDefault<double>( "y_offset", 0.0 ),
                           db->getWithDefault<double>( "z_offset", 0.0 ) };
        auto geom      = AMP::Geometry::Geometry::buildGeometry( geom_db );
        geom->displace( dist );
        auto res = db->getScalar<double>( "Resolution" );
        return generateGeom( geom, params->getComm(), res );
    } else {
        AMP_ERROR( "Unknown parameters for TriangleMesh" );
    }
    return nullptr;
}
size_t estimateMeshSize( std::shared_ptr<const MeshParameters> params )
{
    auto db = params->getDatabase();
    if ( db->keyExists( "FileName" ) ) {
        auto filename = db->getScalar<std::string>( "FileName", "" );
        auto suffix   = IO::getSuffix( filename );
        if ( suffix == "stl" ) {
            // We are reading an stl file
            return TriangleHelpers::readSTLHeader( filename );
        } else {
            AMP_ERROR( "Unknown format for TriangleMesh" );
        }
    } else if ( db->keyExists( "Geometry" ) ) {
        auto geom_db    = db->getDatabase( "Geometry" );
        auto geometry   = AMP::Geometry::Geometry::buildGeometry( geom_db );
        double volume   = geometry->volume();
        int geomDim     = geometry->getDim();
        auto resolution = db->getScalar<double>( "Resolution" );
        double triVol   = std::pow( resolution, geomDim ) / factorial( geomDim );
        return std::max<size_t>( 1, volume / triVol );
    } else {
        AMP_ERROR( "Unknown method for TriangleMesh" );
    }
    return 0;
}
size_t maxProcs( std::shared_ptr<const MeshParameters> ) { return 1; }


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
// clang-format off
using point1D = std::array<double, 1>;
using point2D = std::array<double, 2>;
using point3D = std::array<double, 3>;
using pointset1D = std::vector<point1D>;
using pointset2D = std::vector<point2D>;
using pointset3D = std::vector<point3D>;
using triset1D = std::vector<std::array<int, 2>>;
using triset2D = std::vector<std::array<int, 3>>;
using triset3D = std::vector<std::array<int, 4>>;
template size_t count<1>( const triset1D & );
template size_t count<2>( const triset2D & );
template size_t count<3>( const triset3D & );
template void createTriangles<1,1>( const std::vector<std::array<point1D,2>>&, pointset1D&, triset1D&, double );
template void createTriangles<1,2>( const std::vector<std::array<point2D,2>>&, pointset2D&, triset1D&, double );
template void createTriangles<1,3>( const std::vector<std::array<point3D,2>>&, pointset3D&, triset1D&, double );
template void createTriangles<2,2>( const std::vector<std::array<point2D,3>>&, pointset2D&, triset2D&, double );
template void createTriangles<2,3>( const std::vector<std::array<point3D,3>>&, pointset3D&, triset2D&, double );
template void createTriangles<3,3>( const std::vector<std::array<point3D,4>>&, pointset3D&, triset3D&, double );
template std::vector<triset2D> splitDomains<2>( triset2D );
template std::vector<triset3D> splitDomains<3>( triset3D );
// clang-format on

} // namespace AMP::Mesh::TriangleHelpers
