#include "AMP/ampmesh/triangle/TriangleHelpers.h"
#include "AMP/ampmesh/MeshParameters.h"
#include "AMP/ampmesh/MultiGeometry.h"
#include "AMP/ampmesh/MultiMesh.h"
#include "AMP/ampmesh/triangle/TriangleMesh.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <map>


namespace AMP {
namespace Mesh {
namespace TriangleHelpers {


// Helper function to create constexpr std::array with a single value
template<class T, std::size_t N, std::size_t... I>
static constexpr std::array<std::remove_cv_t<T>, N> to_array_impl( const T *a,
                                                                   std::index_sequence<I...> )
{
    return { { a[I]... } };
}
template<class TYPE, std::size_t N>
static constexpr std::array<TYPE, N> make_array( const TYPE &x )
{
    TYPE tmp[N] = { x };
    return to_array_impl<TYPE, N>( tmp, std::make_index_sequence<N>{} );
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
}


/****************************************************************
 * Find the first n intersections in multiple lists              *
 * This function assumes the lists are in sorted order           *
 ****************************************************************/
static int intersect_sorted(
    const int N_lists, const int size[], int64_t *list[], const int N_max, int64_t *intersection )
{
    if ( N_max <= 0 )
        return ~( (unsigned int) 0 );
    int N_int = 0;
    std::vector<int> index( N_lists );
    for ( int i = 0; i < N_lists; i++ )
        index[i] = 0;
    unsigned int current_val = list[0][0];
    bool finished            = false;
    while ( true ) {
        unsigned int min_val = 2147483647;
        bool in_intersection = true;
        for ( int i = 0; i < N_lists; i++ ) {
            if ( index[i] >= size[i] ) {
                finished = true;
                break;
            }
            while ( list[i][index[i]] < current_val ) {
                index[i]++;
                if ( index[i] >= size[i] ) {
                    finished = true;
                    break;
                }
            }
            if ( list[i][index[i]] == current_val ) {
                index[i]++;
            } else {
                in_intersection = false;
            }
            if ( index[i] < size[i] ) {
                if ( list[i][index[i]] < min_val )
                    min_val = list[i][index[i]];
            }
        }
        if ( finished )
            break;
        if ( in_intersection ) {
            intersection[N_int] = current_val;
            N_int++;
            if ( N_int >= N_max )
                break;
        }
        current_val = min_val;
    }
    return N_int;
}


/****************************************************************
 * Count the number of unique triangles                          *
 ****************************************************************/
template<size_t NDIM, bool ordered>
static uint64_t hash( const std::array<int64_t, NDIM> &x )
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
size_t count( const std::vector<std::array<int64_t, NG + 1>> &tri )
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
        uint16_t attrib    = 0;
        float normal[3]    = { 0, 0, 0 };
        float vertex[3][3] = { { 0 } };
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
        NULL_USE( attrib );
        NULL_USE( normal );
    }
    delete[] tmp;
    return tri_coord;
}


/****************************************************************
 * Create triangles/verticies from a set of triangles specified  *
 * by their coordinates                                          *
 ****************************************************************/
template<size_t NG, size_t NP>
void createTriangles( const std::vector<std::array<std::array<double, NP>, NG + 1>> &tri_list,
                      std::vector<std::array<double, NP>> &verticies,
                      std::vector<std::array<int64_t, NG + 1>> &triangles,
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
    // Get the unique verticies and create triangle indicies
    verticies.clear();
    triangles.clear();
    constexpr auto null_tri = make_array<int64_t, NG + 1>( -1 );
    triangles.resize( tri_list.size(), null_tri );
    for ( size_t i = 0; i < tri_list.size(); i++ ) {
        for ( size_t j = 0; j < NG + 1; j++ ) {
            auto &point   = tri_list[i][j];
            int64_t index = -1;
            for ( size_t k = 0; k < verticies.size() && index == -1; k++ ) {
                if ( approx_equal( point, verticies[k], tol2 ) )
                    index = k;
            }
            if ( index == -1 ) {
                index = verticies.size();
                verticies.push_back( point );
            }
            triangles[i][j] = index;
        }
    }
}


/****************************************************************
 * Create triangles neighbors from the triangles                 *
 ****************************************************************/
template<size_t NG>
std::vector<std::array<int64_t, NG + 1>>
create_tri_neighbors( const std::vector<std::array<int64_t, NG + 1>> &tri )
{
    // Allocate memory
    constexpr auto null_tri = make_array<int64_t, NG + 1>( -1 );
    std::vector<std::array<int64_t, NG + 1>> tri_nab( tri.size(), null_tri );
    if ( tri.size() == 1 )
        return tri_nab;
    // 1D is a special easy case
    if constexpr ( NG == 1 ) {
        for ( size_t i = 0; i < tri.size(); i++ ) {
            tri_nab[i][0] = i + 1;
            tri_nab[i][1] = i - 1;
        }
        tri_nab[0][1]              = -1;
        tri_nab[tri.size() - 1][0] = -1;
        return tri_nab;
    }
    PROFILE_START( "create_tri_neighbors", 1 );
    // Get the number of verticies
    size_t N_vertex = 0;
    for ( const auto &t : tri ) {
        for ( size_t i = 0; i < NG + 1; i++ )
            N_vertex = std::max<size_t>( N_vertex, t[i] + 1 );
    }
    // Count the number of triangles connected to each vertex
    std::vector<int64_t> N_tri_nab( N_vertex, 0 );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( size_t d = 0; d < NG + 1; d++ )
            N_tri_nab[tri[i][d]]++;
    }
    // For each node, get a list of the triangles that connect to that node
    auto tri_list = new int64_t *[N_vertex]; // List of triangles connected each node (N)
    tri_list[0]   = new int64_t[( NG + 1 ) * tri.size()];
    for ( size_t i = 1; i < N_vertex; i++ )
        tri_list[i] = &tri_list[i - 1][N_tri_nab[i - 1]];
    for ( size_t i = 0; i < ( NG + 1 ) * tri.size(); i++ )
        tri_list[0][i] = -1;
    // Create a sorted list of all triangles that have each node as a vertex
    for ( size_t i = 0; i < N_vertex; i++ )
        N_tri_nab[i] = 0;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( size_t j = 0; j <= NG; j++ ) {
            int64_t k                 = tri[i][j];
            tri_list[k][N_tri_nab[k]] = i;
            N_tri_nab[k]++;
        }
    }
    for ( size_t i = 0; i < N_vertex; i++ )
        AMP::Utilities::quicksort( N_tri_nab[i], tri_list[i] );
    int64_t N_tri_max = 0;
    for ( size_t i = 0; i < N_vertex; i++ ) {
        if ( N_tri_nab[i] > N_tri_max )
            N_tri_max = N_tri_nab[i];
    }
    // Note, if a triangle is a neighbor, it will share all but the current node
    int size[NG];
    int error = 0;
    for ( int64_t i = 0; i < (int64_t) tri.size(); i++ ) {
        // Loop through the different faces of the triangle
        for ( size_t j = 0; j <= NG; j++ ) {
            int64_t *list[NG] = { nullptr };
            int64_t k1        = 0;
            for ( size_t k2 = 0; k2 <= NG; k2++ ) {
                if ( k2 == j )
                    continue;
                int64_t k = tri[i][k2];
                list[k1]  = tri_list[k];
                size[k1]  = N_tri_nab[k];
                k1++;
            }
            // Find the intersection of all triangle lists except the current node
            int64_t intersection[5] = { -1, -1, -1, -1, -1 };
            int64_t N_int           = intersect_sorted( NG, size, list, 5, intersection );
            int64_t m               = 0;
            if ( N_int == 0 || N_int > 2 ) {
                // We cannot have less than 1 triangle or more than 2 triangles sharing NDIM nodes
                error = 1;
                break;
            } else if ( intersection[0] == i ) {
                m = intersection[1];
            } else if ( intersection[1] == i ) {
                m = intersection[0];
            } else {
                // One of the triangles must be the current triangle
                error = 1;
                break;
            }
            tri_nab[i][j] = m;
        }
        if ( error != 0 )
            break;
    }
    // Check tri_nab
    for ( int64_t i = 0; i < (int64_t) tri.size(); i++ ) {
        for ( size_t d = 0; d <= NG; d++ ) {
            if ( tri_nab[i][d] < -1 || tri_nab[i][d] >= (int64_t) tri.size() || tri_nab[i][d] == i )
                error = 2;
        }
    }
    delete[] tri_list[0];
    delete[] tri_list;
    if ( error == 1 ) {
        throw std::logic_error( "Error in create_tri_neighbors detected" );
    } else if ( error == 2 ) {
        throw std::logic_error( "Internal error" );
    }
    PROFILE_STOP( "create_tri_neighbors", 1 );
    return tri_nab;
}


/****************************************************************
 * Try to split the mesh into seperate independent domains       *
 ****************************************************************/
static inline std::array<int64_t, 2> getFace( const std::array<int64_t, 3> &tri, size_t i )
{
    return { tri[( i + 1 ) % 3], tri[( i + 2 ) % 3] };
}
static inline std::array<int64_t, 3> getFace( const std::array<int64_t, 4> &tri, size_t i )
{
    return { tri[( i + 1 ) % 4], tri[( i + 2 ) % 4], tri[( i + 3 ) % 4] };
}
template<size_t NG>
static void addFaces( const std::array<int64_t, NG + 1> &tri,
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
static std::vector<std::array<int64_t, NG + 1>>
    removeSubDomain( std::vector<std::array<int64_t, NG + 1>> &tri )
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
    // Choose a triangle at random and store the edges
    std::vector<bool> used( tri.size(), false );
    std::vector<std::array<int64_t, NG + 1>> tri2;
    std::vector<std::pair<uint64_t, int64_t>> faces;
    used[0] = true;
    tri2.push_back( tri[0] );
    addFaces<NG>( tri[0], 0, faces );
    erase( faceMap, 0 );
    // Add triangles until all faces have been filled
    while ( !faces.empty() ) {
        bool found = false;
        for ( size_t i = 0; i < faces.size(); i++ ) {
            int Nf = faceMap.count( faces[i].first );
            AMP_ASSERT( Nf > 0 );
            if ( Nf == 1 ) {
                // We are dealing with a unique match, add the triangle
                auto it = faceMap.find( faces[0].first );
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
        // We have multiple faces to choose from, need to choose wisely
        AMP_ERROR( "Not finished" );
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
std::vector<std::vector<std::array<int64_t, NG + 1>>>
    splitDomains( std::vector<std::array<int64_t, NG + 1>> tri )
{
    std::vector<std::vector<std::array<int64_t, NG + 1>>> tri_sets;
    while ( !tri.empty() )
        tri_sets.emplace_back( removeSubDomain<NG>( tri ) );
    return tri_sets;
}
template<>
std::vector<std::vector<std::array<int64_t, 2>>>
    splitDomains<1>( std::vector<std::array<int64_t, 2>> )
{
    AMP_ERROR( "1D splitting of domains is not supported" );
    return std::vector<std::vector<std::array<int64_t, 2>>>();
}


/********************************************************
 *  Generate mesh for STL file                           *
 ********************************************************/
std::shared_ptr<AMP::Mesh::Mesh> generateSTL( MeshParameters::shared_ptr params )
{
    auto db       = params->getDatabase();
    auto filename = db->getWithDefault<std::string>( "FileName", "" );
    auto name     = db->getWithDefault<std::string>( "MeshName", "NULL" );
    auto comm     = params->getComm();
    // Read the STL file
    typedef std::vector<std::array<double, 3>> pointset;
    typedef std::vector<std::array<int64_t, 3>> triset;
    pointset vert;
    std::vector<triset> tri( 1 );
    if ( comm.getRank() == 0 ) {
        auto scale     = db->getWithDefault<double>( "scale", 1.0 );
        auto triangles = TriangleHelpers::readSTL( filename, scale );
        // Create triangles from the points
        double tol = 1e-6;
        TriangleHelpers::createTriangles<2, 3>( triangles, vert, tri[0], tol );
        // Find the number of unique triangles (duplicates may indicate multiple objects
        size_t N2 = TriangleHelpers::count<2>( tri[0] );
        if ( N2 > 1 ) {
            // Try to split the domains
            tri = TriangleHelpers::splitDomains<2>( tri[0] );
        }
    }
    size_t N_domains = comm.bcast( tri.size(), 0 );
    tri.resize( N_domains );
    // Create the triangle neighbors
    std::vector<triset> tri_nab( N_domains );
    if ( comm.getRank() == 0 ) {
        for ( size_t i = 0; i < N_domains; i++ ) {
            // Get the triangle neighbors
            tri_nab[i] = TriangleHelpers::create_tri_neighbors<2>( tri[i] );
            // Check if the geometry is closed
            bool closed = true;
            for ( const auto &t : tri_nab[i] ) {
                for ( const auto &p : t )
                    closed = closed && p >= 0;
            }
            if ( !closed )
                AMP_WARNING( "Geometry is not closed" );
        }
    }
    // Create the mesh
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    if ( N_domains == 1 ) {
        mesh = TriangleMesh<2, 3>::generate( vert, tri[0], tri_nab[0], comm );
    } else {
        // For now have all meshes on the same communicator
        std::vector<std::shared_ptr<AMP::Mesh::Mesh>> submeshes( N_domains );
        for ( size_t i = 0; i < N_domains; i++ ) {
            submeshes[i] = TriangleMesh<2, 3>::generate( vert, tri[i], tri_nab[i], comm );
            submeshes[i]->setName( name + "_" + std::to_string( i + 1 ) );
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
    if ( disp[0] != 0.0 && disp[1] != 0.0 && disp[2] != 0.0 )
        mesh->displaceMesh( disp );
    // Set the mesh name
    mesh->setName( name );
    return mesh;
}


/********************************************************
 *  Generate mesh for geometry                           *
 ********************************************************/
std::shared_ptr<AMP::Mesh::Mesh> generate( std::shared_ptr<AMP::Geometry::Geometry> geom,
                                           const AMP_MPI &comm,
                                           const Point &resolution )
{
    auto multigeom = std::dynamic_pointer_cast<AMP::Geometry::MultiGeometry>( geom );
    if ( multigeom ) {
        std::vector<std::shared_ptr<AMP::Mesh::Mesh>> submeshes;
        for ( auto &geom2 : multigeom->getGeometries() ) {
            submeshes.push_back( generate( geom2, comm, resolution ) );
            // submeshes[i]->setName( name + "_" + std::to_string( i + 1 ) );
        }
        return std::make_shared<MultiMesh>( "name", comm, submeshes );
    }
    AMP_INSIST( geom->isConvex(), "Geometry must be convex" );
    // Get the volume points
    std::vector<Point> points;
    auto [x0, x1] = geom->box();
    for ( double x = x0.x(); x <= x1.x(); x += resolution.x() ) {
        for ( double y = x0.y(); y <= x1.y(); y += resolution.y() ) {
            for ( double z = x0.z(); z <= x1.z(); z += resolution.z() ) {
                Point p( x0.size(), { x, y, z } );
                if ( geom->inside( p ) )
                    points.push_back( p );
            }
        }
    }
    // Get the surface points

    // Tessellate

    // Generate the mesh

    AMP_ERROR( "Not finished" );
    return std::shared_ptr<AMP::Mesh::Mesh>();
}


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
// clang-format off
typedef std::array<double,1> point1D;
typedef std::array<double,2> point2D;
typedef std::array<double,3> point3D;
typedef std::vector<point1D> pointset1D;
typedef std::vector<point2D> pointset2D;
typedef std::vector<point3D> pointset3D;
typedef std::vector<std::array<int64_t,2>> triset1D;
typedef std::vector<std::array<int64_t,3>> triset2D;
typedef std::vector<std::array<int64_t,4>> triset3D;
template size_t count<1>( const triset1D & );
template size_t count<2>( const triset2D & );
template size_t count<3>( const triset3D & );
template void createTriangles<1,1>( const std::vector<std::array<point1D,2>>&, pointset1D&, triset1D&, double );
template void createTriangles<1,2>( const std::vector<std::array<point2D,2>>&, pointset2D&, triset1D&, double );
template void createTriangles<1,3>( const std::vector<std::array<point3D,2>>&, pointset3D&, triset1D&, double );
template void createTriangles<2,2>( const std::vector<std::array<point2D,3>>&, pointset2D&, triset2D&, double );
template void createTriangles<2,3>( const std::vector<std::array<point3D,3>>&, pointset3D&, triset2D&, double );
template void createTriangles<3,3>( const std::vector<std::array<point3D,4>>&, pointset3D&, triset3D&, double );
template triset1D create_tri_neighbors<1>( const triset1D& );
template triset2D create_tri_neighbors<2>( const triset2D& );
template triset3D create_tri_neighbors<3>( const triset3D& );
template std::vector<triset2D> splitDomains<2>( triset2D );
template std::vector<triset3D> splitDomains<3>( triset3D );
// clang-format on

} // namespace TriangleHelpers
} // namespace Mesh
} // namespace AMP