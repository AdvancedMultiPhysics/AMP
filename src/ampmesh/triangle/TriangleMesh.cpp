#include "AMP/ampmesh/triangle/TriangleMesh.h"
#include "AMP/ampmesh/MeshParameters.h"
#include "AMP/ampmesh/MultiIterator.h"
#include "AMP/ampmesh/triangle/TriangleHelpers.h"
#include "AMP/ampmesh/triangle/TriangleMeshIterator.h"
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.h"
#ifdef USE_AMP_VECTORS
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#endif

#include "ProfilerApp.h"

#include <algorithm>
#include <cstring>
#include <iostream>


namespace AMP {
namespace Mesh {


typedef std::array<ElementID, 2> Edge;
typedef std::array<ElementID, 3> Triangle;
typedef std::array<ElementID, 4> Tetrahedron;


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


/****************************************************************
 * Get the number of n-Simplex elements of each type             *
 ****************************************************************/
// clang-format off
static constexpr uint8_t n_Simplex_elements[10][10] = {
        {  1,  0,   0,   0,   0,   0,   0,   0,  0,  0 },
        {  2,  1,   0,   0,   0,   0,   0,   0,  0,  0 },
        {  3,  3,   1,   0,   0,   0,   0,   0,  0,  0 },
        {  4,  6,   4,   1,   0,   0,   0,   0,  0,  0 },
        {  5, 10,  10,   5,   1,   0,   0,   0,  0,  0 },
        {  6, 15,  20,  15,   6,   1,   0,   0,  0,  0 },
        {  7, 21,  35,  35,  21,   7,   1,   0,  0,  0 },
        {  8, 28,  56,  70,  56,  28,   8,   1,  0,  0 },
        {  9, 36,  84, 126, 126,  84,  36,   9,  1,  0 },
        { 10, 45, 120, 210, 252, 210, 120,  45, 10,  1 },
};
// clang-format on


/****************************************************************
 * Get the children from an element                              *
 ****************************************************************/
template<uint8_t N1, uint8_t N2>
static std::array<std::array<ElementID, N2 + 1>, n_Simplex_elements[N1][N2]>
getChildren( const std::array<ElementID, N1 + 1> &parent )
{
    std::array<std::array<ElementID, N2 + 1>, n_Simplex_elements[N1][N2]> children;
    if constexpr ( N1 == N2 ) {
        children[0] = parent[0];
    } else if constexpr ( N2 == 0 ) {
        for ( size_t i = 0; i < N1 + 1; i++ )
            children[i] = parent[i];
    } else if constexpr ( N2 == 1 ) {
        int k = 0;
        for ( size_t i = 0; i < N1; i++ )
            for ( size_t j = i + 1; j <= N1; j++ )
                children[k++] = { parent[i], parent[j] };
    } else if constexpr ( N2 == 2 && N1 == 3 ) {
        children[0] = { parent[1], parent[2], parent[3] };
        children[1] = { parent[2], parent[3], parent[0] };
        children[2] = { parent[3], parent[0], parent[1] };
        children[3] = { parent[0], parent[1], parent[2] };
    } else {
        AMP_ERROR( "Not finished" );
    }
    for ( auto &child : children )
        std::sort( child.begin(), child.end() );
    return children;
}
template<std::size_t N1, std::size_t N2>
static void getChildren( const std::vector<std::array<ElementID, N1 + 1>> &tri,
                         std::vector<std::array<ElementID, N2 + 1>> &local,
                         std::vector<std::array<ElementID, N2 + 1>> &remote )
{
    local.clear();
    remote.clear();
    local.reserve( n_Simplex_elements[N1][N2] * tri.size() / 2 );
    for ( auto obj : tri ) {
        for ( const auto &child : getChildren<N1, N2>( obj ) ) {
            if ( child[0].is_local() )
                local.push_back( child );
            else
                remote.push_back( child );
        }
    }
    std::sort( local.begin(), local.end() );
    std::sort( remote.begin(), remote.end() );
}


/****************************************************************
 * Remove unused verticies                                       *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
static void removeUnusedVerticies( std::vector<std::array<double, NP>> &verticies,
                                   std::vector<std::array<int64_t, NG + 1>> &tri )
{
    // Check which verticies are used
    std::vector<bool> used( verticies.size() );
    for ( auto &t : tri ) {
        for ( auto i : t )
            used[i] = true;
    }
    // Create a map to renumber and remove unused verticies
    std::vector<size_t> map( used.size(), -1 );
    size_t N = 0;
    for ( size_t i = 0; i < used.size(); i++ ) {
        if ( used[i] ) {
            map[i]       = N;
            verticies[N] = verticies[i];
            N++;
        }
    }
    verticies.resize( N );
    // Renumber triangles
    for ( auto &t : tri ) {
        for ( auto &i : t )
            i = map[i];
    }
}


/****************************************************************
 * Perform load balancing                                        *
 * At exit:                                                      *
 *    only local data will remain                                *
 *    local verticies will be stored in sorted order             *
 *    indexing is the global                                     *
 ****************************************************************/
template<size_t NP>
static std::vector<size_t> splitDomain( const std::vector<std::array<double, NP>> &center )
{
    AMP_WARNING( "Not finished" );
    return std::vector<size_t>( center.size(), 0 );
}
static std::vector<size_t> mapOwner( const std::vector<size_t> &rank, const AMP_MPI &comm )
{
    // Get the mapping for each triangle to the correct processor
    std::vector<size_t> count( comm.getSize(), 0 );
    for ( const auto &r : rank )
        count[r]++;
    std::vector<size_t> offset( comm.getSize(), 0 );
    for ( int i = 1; i < comm.getSize(); i++ )
        offset[i] = offset[i - 1] + count[i - 1];
    std::fill( count.begin(), count.end(), 0 );
    std::vector<size_t> map( rank.size(), 0 );
    for ( size_t i = 0; i < rank.size(); i++ )
        map[i] = offset[rank[i]] + count[rank[i]]++;
    return map;
}
template<class TYPE>
static std::vector<TYPE>
sendData( const std::vector<TYPE> &data, const std::vector<size_t> &rank, const AMP_MPI &comm )
{
    std::vector<TYPE> out;
    if ( comm.getRank() == 0 ) {
        // Pack the data to send to each rank
        std::vector<std::vector<TYPE>> data2( comm.getSize() );
        for ( size_t i = 0; i < data.size(); i++ )
            data2[rank[i]].push_back( data[i] );
        std::vector<size_t> count( comm.getSize(), 0 );
        for ( int i = 0; i < comm.getSize(); i++ )
            count[i] = data2[i].size();
        comm.bcast( count.data(), count.size(), 0 );
        // Send the data
        std::vector<MPI_Request> request;
        for ( int i = 1; i < comm.getSize(); i++ ) {
            if ( count[i] > 0 ) {
                auto req = comm.Isend<TYPE>( data2[i].data(), data2[i].size(), i, 125 );
                request.push_back( req );
            }
        }
        comm.waitAll( request.size(), request.data() );
        out = std::move( data2[0] );
    } else {
        // Recieve the data
        std::vector<size_t> count( comm.getSize(), 0 );
        comm.bcast( count.data(), count.size(), 0 );
        if ( count[comm.getRank()] > 0 ) {
            out.resize( count[comm.getRank()] );
            int length = out.size();
            comm.recv<TYPE>( out.data(), length, 0, false, 125 );
        }
    }
    return out;
}
template<uint8_t NG, uint8_t NP>
static void loadBalance( std::vector<std::array<double, NP>> &verticies,
                         std::vector<std::array<int64_t, NG + 1>> &tri,
                         std::vector<std::array<int64_t, NG + 1>> &tri_nab,
                         std::vector<int> &block,
                         const AMP_MPI &comm )
{
    // Perform the load balancing
    if ( comm.getSize() == 1 )
        return;
    // Check that only rank 0 has data (may relax this in the future)
    if ( comm.getRank() != 0 )
        AMP_ASSERT( verticies.empty() && tri.empty() && tri_nab.empty() );
    // Get the number of subdomains in each direction
    auto factors    = AMP::Utilities::factor( comm.getSize() );
    int N_domain[3] = { 1, 1, 1 };
    for ( auto it = factors.rbegin(); it != factors.rend(); ++it ) {
        if ( N_domain[0] <= N_domain[1] && N_domain[1] <= N_domain[2] )
            N_domain[0] *= *it;
        else if ( N_domain[1] <= N_domain[2] )
            N_domain[1] *= *it;
        else
            N_domain[2] *= *it;
    }
    // Get the triangle centers (used for load balance)
    std::vector<std::array<double, NP>> center( tri.size(), make_array<double, NP>( 0 ) );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( size_t j = 0; j < NG + 1; j++ ) {
            const auto &point = verticies[tri[i][j]];
            for ( size_t d = 0; d < NP; d++ )
                center[i][d] += point[d] / NP;
        }
    }
    // Recursively split the domain
    auto rank_tri  = splitDomain( center );
    auto rank_node = splitDomain( center );
    // Get the mapping for each triangle to the correct processor
    auto map_tri  = mapOwner( rank_tri, comm );
    auto map_node = mapOwner( rank_tri, comm );
    // Remap the triangles
    for ( auto &t : tri ) {
        for ( auto &v : t )
            if ( v != -1 )
                v = map_node[v];
    }
    // Remap the triangle neighbors
    for ( auto &t : tri_nab ) {
        for ( auto &v : t )
            if ( v != -1 )
                v = map_tri[v];
    }
    // Move the data
    verticies = sendData( verticies, rank_node, comm );
    tri       = sendData( tri, rank_node, comm );
    tri_nab   = sendData( tri_nab, rank_node, comm );
    block     = sendData( block, rank_node, comm );
}
template<size_t NDIM, class TYPE>
static void sortData( std::vector<TYPE> &data,
                      std::vector<std::array<int64_t, NDIM>> &index,
                      const AMP_MPI &comm )
{
    // Sort the local data updating the indicies
    std::vector<size_t> I, J;
    AMP::Utilities::unique( data, I, J );
    AMP_ASSERT( I.size() == J.size() );
    auto N        = comm.allGather( data.size() );
    size_t offset = 0;
    for ( int i = 0; i < comm.getRank(); i++ )
        offset += N[i];
    for ( auto &v : J )
        v += offset;
    auto map = comm.allGather( J );
    for ( auto &x : index ) {
        for ( auto &y : x )
            if ( y != -1 )
                y = map[y];
    }
}
template<size_t NDIM, class TYPE>
static void sortData( std::vector<TYPE> &data,
                      std::vector<std::array<int64_t, NDIM>> &index,
                      std::vector<int> &block,
                      const AMP_MPI &comm )
{
    // Sort the local data updating the indicies
    std::vector<size_t> I, J;
    AMP::Utilities::unique( data, I, J );
    AMP_ASSERT( I.size() == J.size() );
    auto N        = comm.allGather( data.size() );
    size_t offset = 0;
    for ( int i = 0; i < comm.getRank(); i++ )
        offset += N[i];
    for ( auto &v : J )
        v += offset;
    auto map = comm.allGather( J );
    for ( auto &x : index ) {
        for ( auto &y : x )
            if ( y != -1 )
                y = map[y];
    }
    AMP_ASSERT( I.size() == block.size() );
    std::vector<int> tmp = block;
    for ( size_t i = 0; i < block.size(); i++ )
        block[i] = tmp[I[i]];
}


/****************************************************************
 * Generator                                                     *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
std::shared_ptr<TriangleMesh<NG, NP>>
TriangleMesh<NG, NP>::generate( std::shared_ptr<MeshParameters> params )
{
    auto db = params->getDatabase();
    // Create the mesh
    auto filename = db->getWithDefault<std::string>( "FileName", "" );
    auto suffix   = Utilities::getSuffix( filename );
    std::shared_ptr<TriangleMesh<NG, NP>> mesh;
    if ( suffix == "stl" ) {
        AMP_ERROR( "Use AMP::Mesh::TriangleHelpers::generateSTL to load stl meshes" );
    } else {
        AMP_ERROR( "Not finished" );
    }
    // Displace the mesh
    std::vector<double> displacement( NP, 0.0 );
    if ( db->keyExists( "x_offset" ) && NP >= 1 )
        displacement[0] = db->getScalar<double>( "x_offset" );
    if ( db->keyExists( "y_offset" ) && NP >= 2 )
        displacement[1] = db->getScalar<double>( "y_offset" );
    if ( db->keyExists( "z_offset" ) && NP >= 3 )
        displacement[2] = db->getScalar<double>( "z_offset" );
    bool test = false;
    for ( auto &elem : displacement ) {
        if ( elem != 0.0 )
            test = true;
    }
    if ( test )
        mesh->displaceMesh( displacement );
    return mesh;
}
template<uint8_t NG, uint8_t NP>
std::shared_ptr<TriangleMesh<NG, NP>>
TriangleMesh<NG, NP>::generate( std::vector<std::array<double, NP>> vert,
                                std::vector<std::array<int64_t, NG + 1>> tri,
                                std::vector<std::array<int64_t, NG + 1>> tri_nab,
                                const AMP_MPI &comm,
                                std::shared_ptr<Geometry::Geometry> geom,
                                std::vector<int> block )
{
    if ( comm.getRank() != 0 )
        AMP_ASSERT( vert.empty() && tri.empty() && tri_nab.empty() );
    std::shared_ptr<TriangleMesh<NG, NP>> mesh( new TriangleMesh<NG, NP>( std::move( vert ),
                                                                          std::move( tri ),
                                                                          std::move( tri_nab ),
                                                                          comm,
                                                                          std::move( geom ),
                                                                          std::move( block ) ) );
    return mesh;
}
template<uint8_t NG, uint8_t NP>
std::shared_ptr<TriangleMesh<NG, NP>> TriangleMesh<NG, NP>::generate(
    const std::vector<std::array<std::array<double, NP>, NG + 1>> &tri_list,
    const AMP_MPI &comm,
    double tol )
{
    // Get the global list of tri_list
    auto global_list = comm.allGather( tri_list );
    std::vector<std::array<double, NP>> verticies;
    std::vector<std::array<int64_t, NG + 1>> triangles;
    std::vector<std::array<int64_t, NG + 1>> neighbors;

    if ( comm.getRank() == 0 ) {
        // Create triangles from the points
        TriangleHelpers::createTriangles<NG, NP>( global_list, verticies, triangles, tol );
        // Find the number of unique triangles (duplicates may indicate multiple objects
        size_t N2 = TriangleHelpers::count<NG>( triangles );
        if ( N2 == tri_list.size() ) {
            // Get the triangle neighbors
            neighbors = TriangleHelpers::create_tri_neighbors<NG>( triangles );
            // Check if the geometry is closed
            if constexpr ( NG == NP ) {
                bool closed = true;
                for ( const auto &t : neighbors ) {
                    for ( const auto &p : t )
                        closed = closed && p >= 0;
                }
                if ( !closed )
                    AMP_WARNING( "Geometry is not closed" );
            }
        } else {
            // auto triangles2 = TriangleHelpers::splitDomains<NG>( triangles );
            AMP_WARNING(
                "Duplicate triangles detected, no connectivity information will be stored" );
            neighbors.resize( triangles.size(), make_array<int64_t, NG + 1>( -1 ) );
        }
    }
    // Create the mesh
    std::shared_ptr<TriangleMesh<NG, NP>> mesh( new TriangleMesh<NG, NP>( std::move( verticies ),
                                                                          std::move( triangles ),
                                                                          std::move( neighbors ),
                                                                          comm,
                                                                          nullptr,
                                                                          std::vector<int>() ) );
    return mesh;
}
template<uint8_t NG>
static std::vector<std::array<ElementID, NG + 1>>
createGlobalIDs( const std::vector<std::array<int64_t, NG + 1>> &index,
                 size_t N_local,
                 GeomType type,
                 const AMP_MPI &comm )
{
    // Get the index offsets for each rank
    auto N = comm.allGather( N_local );
    std::vector<size_t> size( N.size(), 0 );
    size[0] = N[0];
    for ( size_t i = 1; i < N.size(); i++ )
        size[i] = size[i - 1] + N[i];
    std::vector<size_t> offset( N.size(), 0 );
    for ( size_t i = 1; i < N.size(); i++ )
        offset[i] = offset[i - 1] + N[i - 1];
    // Create the global ids
    int myRank = comm.getRank();
    std::vector<std::array<ElementID, NG + 1>> ids( index.size() );
    for ( size_t i = 0; i < index.size(); i++ ) {
        for ( size_t d = 0; d <= NG; d++ ) {
            if ( index[i][d] == -1 )
                continue;
            int rank  = AMP::Utilities::findfirst<size_t>( size, index[i][d] );
            int local = index[i][d] - offset[rank];
            ids[i][d] = ElementID( rank == myRank, type, local, rank );
        }
    }
    return ids;
}
template<uint8_t NG, uint8_t NP>
TriangleMesh<NG, NP>::TriangleMesh( std::vector<std::array<double, NP>> verticies,
                                    std::vector<std::array<int64_t, NG + 1>> tri,
                                    std::vector<std::array<int64_t, NG + 1>> tri_nab,
                                    const AMP_MPI &comm,
                                    std::shared_ptr<Geometry::Geometry> geom_in,
                                    std::vector<int> block )
    : d_pos_hash( 0 )
{
    // Run some basic checks
    AMP_ASSERT( tri.size() == 0 || comm.getRank() == 0 );
    AMP_ASSERT( tri_nab.size() == tri.size() );
    if ( block.empty() )
        block = std::vector<int>( tri.size(), 0 );
    AMP_ASSERT( block.size() == tri.size() );
    // Set basic mesh info
    d_db        = nullptr;
    d_params    = nullptr;
    d_geometry  = nullptr;
    GeomDim     = static_cast<GeomType>( NG );
    PhysicalDim = NP;
    d_max_gcw   = 0;
    d_comm      = comm;
    d_name      = "NULL";
    d_geometry  = std::move( geom_in );
    setMeshID();
    // Remove verticies that are not used
    if ( comm.getRank() == 0 )
        removeUnusedVerticies<NG, NP>( verticies, tri );
    else
        AMP_ASSERT( tri.empty() && verticies.empty() );
    // Perform the load balancing
    loadBalance<NG, NP>( verticies, tri, tri_nab, block, comm );
    sortData( verticies, tri, comm );
    // Create the global ids
    d_vert    = std::move( verticies );
    auto tri2 = createGlobalIDs<NG>( tri, d_vert.size(), GeomType::Vertex, comm );
    sortData( tri2, tri_nab, block, comm );
    d_neighbors = createGlobalIDs<NG>( tri_nab, tri2.size(), static_cast<GeomType>( NG ), comm );
    if constexpr ( NG == 1 )
        std::swap( d_edge, tri2 );
    else if constexpr ( NG == 2 )
        std::swap( d_tri, tri2 );
    else if constexpr ( NG == 3 )
        std::swap( d_tet, tri2 );
    // Fill remote data
    d_max_gcw = 3;
    if ( comm.getSize() > 1 ) {
        AMP_ERROR( "Not finished" );
    }
    // Initialize the iterators and some common data
    initialize();
    // Create the block iterators
    d_blockID = std::move( block );
    std::set<int> blockSet( d_blockID.begin(), d_blockID.end() );
    d_block_ids = std::vector<int>( blockSet.begin(), blockSet.end() );
    d_block_iterators.resize( d_block_ids.size() );
    for ( size_t i = 0; i < d_block_ids.size(); i++ ) {
        auto id0 = d_block_ids[i];
        d_block_iterators[i].resize( d_iterators.size() );
        for ( size_t gcw = 0; gcw < d_iterators.size(); gcw++ ) {
            for ( int type = 0; type <= NG; type++ ) {
                auto list_ptr = std::make_shared<std::vector<ElementID>>();
                auto &list    = *list_ptr;
                for ( const auto &elem : d_iterators[gcw][type] ) {
                    auto id = elem.globalID().elemID();
                    if ( isInBlock( id, id0 ) )
                        list.push_back( id );
                }
                std::sort( list.begin(), list.end() );
                d_block_iterators[i][gcw][type] = createIterator( list_ptr );
            }
        }
    }
    // Create the surface iterators
    d_surface_iterators.resize( d_iterators.size() );
    for ( size_t gcw = 0; gcw < d_iterators.size(); gcw++ ) {
        for ( int type = 0; type <= NG; type++ ) {
            auto list_ptr = std::make_shared<std::vector<ElementID>>();
            auto &list    = *list_ptr;
            for ( const auto &elem : d_iterators[gcw][type] ) {
                auto id = elem.globalID().elemID();
                if ( isOnSurface( id ) )
                    list.push_back( id );
            }
            std::sort( list.begin(), list.end() );
            d_surface_iterators[gcw][type] = createIterator( list_ptr );
        }
    }
    // Create the boundary iterators
    if ( d_geometry ) {
        int Ns = d_geometry->NSurface();
        d_boundary_ids.resize( Ns );
        d_boundary_iterators.resize( Ns );
        for ( int i = 0; i < Ns; i++ ) {
            d_boundary_ids[i] = i;
            d_boundary_iterators[i].resize( d_iterators.size() );
        }
        for ( size_t gcw = 0; gcw < d_iterators.size(); gcw++ ) {
            for ( int type2 = 0; type2 < NG; type2++ ) {
                auto type = static_cast<GeomType>( type2 );
                std::vector<std::vector<ElementID>> list( Ns );
                for ( const auto &elem : getSurfaceIterator( type, gcw ) ) {
                    auto p = elem.centroid();
                    int s  = d_geometry->surface( p );
                    list[s].push_back( elem.globalID().elemID() );
                }
                for ( int i = 0; i < Ns; i++ ) {
                    std::sort( list[i].begin(), list[i].end() );
                    auto ptr = std::make_shared<std::vector<ElementID>>( std::move( list[i] ) );
                    d_boundary_iterators[i][gcw][type2] = createIterator( ptr );
                }
            }
        }
    } else {
        if ( d_surface_iterators.back()[0].size() == 0 ) {
            d_boundary_ids.clear();
            d_boundary_iterators.clear();
        } else {
            d_boundary_ids = std::vector<int>( 1, 0 );
            d_boundary_iterators.resize( 1 );
            d_boundary_iterators[0] = d_surface_iterators;
        }
    }
}


/****************************************************************
 * Initialize mesh data                                          *
 ****************************************************************/
template<class TYPE>
static inline size_t find( const std::vector<TYPE> &x, TYPE y )
{
    if ( x.empty() )
        return 0;
    size_t k = AMP::Utilities::findfirst( x, y );
    k        = std::min( k, x.size() - 1 );
    if ( x[k] != y )
        k = x.size();
    return k;
}
static std::shared_ptr<std::vector<ElementID>> createLocalList( size_t N, GeomType type, int rank )
{
    // Create a local list of element ids
    auto list = std::make_shared<std::vector<ElementID>>( N );
    for ( size_t i = 0; i < N; i++ )
        ( *list )[i] = ElementID( true, type, i, rank );
    return list;
}
template<uint8_t NG>
static StoreCompressedList<ElementID>
computeNodeParents( size_t N_points,
                    const std::vector<std::array<ElementID, NG + 1>> &tri,
                    const std::map<ElementID, std::array<ElementID, NG + 1>> &remote_tri,
                    int rank )
{
    // Construct a parent list
    std::vector<std::vector<ElementID>> parents( N_points );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        ElementID tri_id( true, static_cast<GeomType>( NG ), i, rank );
        for ( size_t j = 0; j <= NG; j++ ) {
            const auto &node_id = tri[i][j];
            if ( node_id.is_local() )
                parents[node_id.local_id()].push_back( tri_id );
        }
    }
    for ( const auto &tmp : remote_tri ) {
        const auto &tri_id = tmp.first;
        for ( size_t j = 0; j <= NG; j++ ) {
            const auto &node_id = tmp.second[j];
            if ( node_id.is_local() )
                parents[node_id.local_id()].push_back( tri_id );
        }
    }
    // Return the parents
    return StoreCompressedList<ElementID>( parents );
}
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::initializeBoundingBox()
{
    // Initialize the bounding box
    d_box.resize( 2 * NP );
    d_box_local.resize( 2 * NP );
    for ( size_t d = 0; d < NP; d++ ) {
        d_box_local[2 * d + 0] = 1e100;
        d_box_local[2 * d + 1] = -1e100;
    }
    for ( const auto &p : d_vert ) {
        for ( size_t d = 0; d < NP; d++ ) {
            d_box_local[2 * d + 0] = std::min( d_box_local[2 * d + 0], p[d] );
            d_box_local[2 * d + 1] = std::max( d_box_local[2 * d + 1], p[d] );
        }
    }
    for ( size_t d = 0; d < NP; d++ ) {
        d_box[2 * d + 0] = d_comm.minReduce( d_box_local[2 * d + 0] );
        d_box[2 * d + 1] = d_comm.maxReduce( d_box_local[2 * d + 1] );
    }
}
template<std::size_t N1, std::size_t N2>
static StoreCompressedList<ElementID>
getParents( const std::vector<std::array<ElementID, N1 + 1>> &elements,
            const std::vector<std::array<ElementID, N2 + 1>> &objects,
            const std::map<ElementID, std::array<ElementID, N2 + 1>> &ghosts,
            int rank )
{
    std::vector<std::vector<ElementID>> parents( elements.size() );
    for ( size_t i = 0; i < objects.size(); i++ ) {
        const auto &obj = objects[i];
        ElementID id( true, static_cast<GeomType>( N2 ), i, rank );
        for ( const auto &child : getChildren<N2, N1>( obj ) ) {
            size_t k = find( elements, child );
            if ( k != elements.size() )
                parents[k].push_back( id );
        }
    }
    for ( const auto &[id, obj] : ghosts ) {
        for ( const auto &child : getChildren<N2, N1>( obj ) ) {
            size_t k = find( elements, child );
            if ( k != elements.size() )
                parents[k].push_back( id );
        }
    }
    return StoreCompressedList<ElementID>( parents );
}
template<std::size_t N1, std::size_t N2>
static std::vector<std::array<ElementID, n_Simplex_elements[N1][N2]>>
getChildrenIDs( const std::vector<std::array<ElementID, N1 + 1>> &elements,
                const std::vector<std::array<ElementID, N2 + 1>> &local,
                const std::map<ElementID, std::array<ElementID, N2 + 1>> &ghost,
                int rank )
{
    std::vector<std::array<ElementID, n_Simplex_elements[N1][N2]>> ids( elements.size() );
    for ( size_t i = 0; i < elements.size(); i++ ) {
        auto children = getChildren<N1, N2>( elements[i] );
        for ( size_t j = 0; j < children.size(); j++ ) {
            size_t k = find( local, children[j] );
            if ( k != local.size() ) {
                ids[i][j] = ElementID( true, GeomType::Edge, k, rank );
            } else {
                for ( const auto &[id, edge] : ghost ) {
                    if ( edge == children[j] )
                        ids[i][j] = id;
                }
            }
            AMP_ASSERT( ids[i][j] != ElementID() );
        }
    }
    return ids;
}
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::initialize()
{
    int rank = d_comm.getRank();
    int size = d_comm.getSize();
    // Re-sort the points and triangles
    AMP_ASSERT( std::is_sorted( d_vert.begin(), d_vert.end() ) );
    if constexpr ( NG == 1 )
        AMP_ASSERT( std::is_sorted( d_edge.begin(), d_edge.end() ) );
    else if constexpr ( NG == 2 )
        AMP_ASSERT( std::is_sorted( d_tri.begin(), d_tri.end() ) );
    else if constexpr ( NG == 3 )
        AMP_ASSERT( std::is_sorted( d_tet.begin(), d_tet.end() ) );
    else
        static_assert( NG > 0 && NG <= 3, "More than 3 dimensions not yet supported" );
    // Create the edges/faces
    std::vector<Edge> remote_edges;
    std::vector<Triangle> remote_faces;
    if constexpr ( NG == 1 ) {
        // No edges
    } else if constexpr ( NG == 2 ) {
        getChildren<2, 1>( d_tri, d_edge, remote_edges );
    } else if constexpr ( NG == 3 ) {
        getChildren<3, 1>( d_tet, d_edge, remote_edges );
        getChildren<3, 2>( d_tet, d_tri, remote_faces );
    } else {
        static_assert( NG > 0 && NG <= 3, "More than 3 dimensions not yet supported" );
    }
    if ( !d_remote_tri.empty() )
        AMP_ERROR( "Not finished, need to fill d_remote_neighbors" );
    if ( !remote_edges.empty() )
        AMP_ERROR( "Not finished, need to fill d_remote_edge" );
    if ( !remote_faces.empty() )
        AMP_ERROR( "Not finished, need to fill d_remote_edge" );
    AMP_ASSERT( std::is_sorted( d_vert.begin(), d_vert.end() ) );
    AMP_ASSERT( std::is_sorted( d_edge.begin(), d_edge.end() ) );
    AMP_ASSERT( std::is_sorted( d_tri.begin(), d_tri.end() ) );
    AMP_ASSERT( std::is_sorted( d_tet.begin(), d_tet.end() ) );
    // Get the global size
    d_N_global[0] = d_comm.sumReduce( d_vert.size() );
    d_N_global[1] = d_comm.sumReduce( d_edge.size() );
    d_N_global[2] = d_comm.sumReduce( d_tri.size() );
    d_N_global[3] = d_comm.sumReduce( d_tet.size() );
    // Get the bounding boxes
    initializeBoundingBox();
    // Initialize the iterators
    int max_gcw = size == 1 ? 0 : d_max_gcw;
    d_iterators.resize( max_gcw + 1 );
    d_iterators[0][0] = createIterator( createLocalList( d_vert.size(), GeomType::Vertex, rank ) );
    if constexpr ( NG >= 1 )
        d_iterators[0][1] =
            createIterator( createLocalList( d_edge.size(), GeomType::Edge, rank ) );
    if constexpr ( NG >= 2 )
        d_iterators[0][2] = createIterator( createLocalList( d_tri.size(), GeomType::Face, rank ) );
    if constexpr ( NG >= 3 )
        d_iterators[0][3] =
            createIterator( createLocalList( d_tet.size(), GeomType::Volume, rank ) );
    for ( int gcw = 1; gcw <= max_gcw; gcw++ ) {
        AMP_ERROR( "Not finished" );
    }
    // Compute the parents
    if constexpr ( NG >= 1 ) {
        d_parents[0][1] = computeNodeParents<1>( d_vert.size(), d_edge, d_remote_edge, rank );
    }
    if constexpr ( NG >= 2 ) {
        d_parents[0][2] = computeNodeParents<2>( d_vert.size(), d_tri, d_remote_tri, rank );
        d_parents[1][2] = getParents<1, 2>( d_edge, d_tri, d_remote_tri, rank );
    }
    if constexpr ( NG >= 3 ) {
        d_parents[0][3] = computeNodeParents<3>( d_vert.size(), d_tet, d_remote_tet, rank );
        d_parents[1][3] = getParents<1, 3>( d_edge, d_tet, d_remote_tet, rank );
        d_parents[2][3] = getParents<2, 3>( d_tri, d_tet, d_remote_tet, rank );
    }
    // Compute the children
    if constexpr ( NG >= 2 ) {
        d_tri_edge = getChildrenIDs<2, 1>( d_tri, d_edge, d_remote_edge, rank );
    }
    if constexpr ( NG >= 3 ) {
        d_tet_edge = getChildrenIDs<3, 1>( d_tet, d_edge, d_remote_edge, rank );
        d_tet_tri  = getChildrenIDs<3, 2>( d_tet, d_tri, d_remote_tri, rank );
    }
}
template<uint8_t NG, uint8_t NP>
MeshIterator
TriangleMesh<NG, NP>::createIterator( std::shared_ptr<std::vector<ElementID>> list ) const
{
    if ( list->empty() )
        return MeshIterator();
    auto type = ( *list )[0].type();
    bool test = true;
    for ( const auto &id : *list )
        test = test && id.type() == type;
    AMP_ASSERT( test );
    if ( type == GeomType::Vertex )
        return TriangleMeshIterator<NG, NP, 0>( this, list );
    if constexpr ( NG >= 1 ) {
        if ( type == GeomType::Edge )
            return TriangleMeshIterator<NG, NP, 1>( this, list );
    }
    if constexpr ( NG >= 2 ) {
        if ( type == GeomType::Face )
            return TriangleMeshIterator<NG, NP, 2>( this, list );
    }
    if constexpr ( NG >= 3 ) {
        if ( type == GeomType::Volume )
            return TriangleMeshIterator<NG, NP, 3>( this, list );
    }
    AMP_ERROR( "Internal error" );
    return MeshIterator();
}


/****************************************************************
 * Estimate the mesh size                                        *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
size_t TriangleMesh<NG, NP>::estimateMeshSize( const std::shared_ptr<MeshParameters> &params )
{
    size_t N      = 0;
    auto db       = params->getDatabase();
    auto filename = db->getWithDefault<std::string>( "FileName", "" );
    auto suffix   = Utilities::getSuffix( filename );
    if ( suffix == "stl" ) {
        // We are reading an stl file
        char header[80];
        uint32_t N2;
        auto fid = fopen( filename.c_str(), "rb" );
        AMP_INSIST( fid, "Unable to open " + filename );
        fread2( header, sizeof( header ), 1, fid );
        fread2( &N2, sizeof( N2 ), 1, fid );
        fclose( fid );
        N = N2;
    } else {
        AMP_ERROR( "Not finished" );
    }
    return N;
}


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
TriangleMesh<NG, NP>::TriangleMesh( std::shared_ptr<MeshParameters> params_in ) : Mesh( params_in )
{
    // Check for valid inputs
    AMP_INSIST( d_params != nullptr, "Params must not be null" );
    AMP_INSIST( !d_comm.isNull(), "Communicator must be set" );
    AMP_INSIST( d_db.get(), "Database must exist" );
    AMP_ERROR( "Not finished" );
}
template<uint8_t NG, uint8_t NP>
TriangleMesh<NG, NP>::TriangleMesh( const TriangleMesh &rhs )
    : Mesh( rhs.d_params ),
      d_N_global{ rhs.d_N_global },
      d_vert{ rhs.d_vert },
      d_edge{ rhs.d_edge },
      d_tri{ rhs.d_tri },
      d_tet{ rhs.d_tet },
      d_neighbors{ rhs.d_neighbors },
      d_remote_vert{ rhs.d_remote_vert },
      d_remote_edge{ rhs.d_remote_edge },
      d_remote_tri{ rhs.d_remote_tri },
      d_remote_tet{ rhs.d_remote_tet }
{
    for ( size_t i = 0; i < NG; i++ ) {
        for ( size_t j = 0; j < NG; j++ ) {
            d_parents[i][j] = rhs.d_parents[i][j];
        }
    }
}
template<uint8_t NG, uint8_t NP>
std::unique_ptr<Mesh> TriangleMesh<NG, NP>::clone() const
{
    return std::unique_ptr<TriangleMesh<NG, NP>>( new TriangleMesh<NG, NP>( *this ) );
}


/****************************************************************
 * De-constructor                                                *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
TriangleMesh<NG, NP>::~TriangleMesh() = default;


/****************************************************************
 * Estimate the maximum number of processors                     *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
size_t TriangleMesh<NG, NP>::maxProcs( const std::shared_ptr<MeshParameters> &params )
{
    return estimateMeshSize( params );
}


/****************************************************************
 * Function to return the element given an ID                    *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
MeshElement *TriangleMesh<NG, NP>::getElement2( const MeshElementID &id ) const
{
    if ( id.type() == AMP::Mesh::GeomType::Vertex )
        return new TriangleMeshElement<NG, NP, 0>( id, this );
    if constexpr ( NG > 0 )
        if ( id.type() == AMP::Mesh::GeomType::Edge )
            return new TriangleMeshElement<NG, NP, 1>( id, this );
    if constexpr ( NG > 1 )
        if ( id.type() == AMP::Mesh::GeomType::Face )
            return new TriangleMeshElement<NG, NP, 2>( id, this );
    if constexpr ( NG > 2 )
        if ( id.type() == AMP::Mesh::GeomType::Volume )
            return new TriangleMeshElement<NG, NP, 3>( id, this );
    return nullptr;
}
template<uint8_t NG, uint8_t NP>
MeshElement TriangleMesh<NG, NP>::getElement( const MeshElementID &id ) const
{
    return MeshElement( getElement2( id ) );
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
template<uint8_t NG, uint8_t NP>
std::pair<const ElementID *, const ElementID *>
TriangleMesh<NG, NP>::getElementParents( const ElementID &id, const GeomType type ) const
{
    size_t type1 = static_cast<size_t>( id.type() );
    size_t type2 = static_cast<size_t>( type );
    // Perform some initial checks
    if ( !id.is_local() )
        AMP_ERROR( "Getting parents for non-owned elements is not supported" );
    if ( type1 == NG )
        AMP_ERROR( "Trying to get parents for largest geometric object" );
    if ( type2 <= type1 )
        AMP_ERROR( "Trying to get parents of the same or smaller type as the current element" );
    // Get the parents
    const auto index = id.local_id();
    const auto &list = d_parents[type1][type2];
    return std::make_pair( list.begin( index ), list.end( index ) );
}
template<uint8_t NG, uint8_t NP>
std::vector<MeshElement> TriangleMesh<NG, NP>::getElementParents( const MeshElement &elem,
                                                                  const GeomType type ) const
{
    auto ids = getElementParents( elem.globalID().elemID(), type );
    std::vector<MeshElement> parents( ids.second - ids.first );
    auto it = ids.first;
    for ( size_t i = 0; i < parents.size(); i++, ++it )
        parents[i] = getElement( MeshElementID( d_meshID, *it ) );
    return parents;
}


/****************************************************************
 * Functions to return the number of elements                    *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
size_t TriangleMesh<NG, NP>::numLocalElements( const GeomType type ) const
{
    if ( type == GeomType::Vertex )
        return d_vert.size();
    if ( type == GeomType::Edge )
        return d_edge.size();
    if ( type == GeomType::Face )
        return d_tri.size();
    if ( type == GeomType::Volume )
        return d_tet.size();
    return 0;
}
template<uint8_t NG, uint8_t NP>
size_t TriangleMesh<NG, NP>::numGlobalElements( const GeomType type ) const
{
    if ( static_cast<uint8_t>( type ) <= NG )
        return d_N_global[static_cast<uint8_t>( type )];
    return 0;
}
template<uint8_t NG, uint8_t NP>
size_t TriangleMesh<NG, NP>::numGhostElements( const GeomType type, int gcw ) const
{
    if ( gcw == 0 || d_comm.getSize() == 1 )
        return 0;
    int type2 = static_cast<int>( type );
    return d_iterators[gcw][type2].size() - d_iterators[0][type2].size();
}


/****************************************************************
 * Function to get an iterator                                   *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
MeshIterator TriangleMesh<NG, NP>::getIterator( const GeomType type, const int gcw ) const
{
    if ( static_cast<size_t>( type ) > NG || gcw > d_max_gcw )
        return MeshIterator();
    int gcw2  = d_comm.getSize() == 1 ? 0 : gcw;
    int type2 = static_cast<int>( type );
    return d_iterators[gcw2][type2];
}


/****************************************************************
 * Function to get an iterator over the surface                  *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
MeshIterator TriangleMesh<NG, NP>::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    int gcw2  = d_comm.getSize() == 1 ? 0 : gcw;
    int type2 = static_cast<int>( type );
    if ( type2 > NG || gcw > d_max_gcw )
        return MeshIterator();
    return d_surface_iterators[gcw2][type2];
}


/****************************************************************
 * Functions to get the boundaries                               *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
std::vector<int> TriangleMesh<NG, NP>::getBoundaryIDs() const
{
    return d_boundary_ids;
}
template<uint8_t NG, uint8_t NP>
MeshIterator TriangleMesh<NG, NP>::getBoundaryIDIterator( const GeomType type,
                                                          const int id,
                                                          const int gcw ) const
{
    int gcw2     = d_comm.getSize() == 1 ? 0 : gcw;
    int type2    = static_cast<int>( type );
    size_t index = d_boundary_iterators.size();
    for ( size_t i = 0; i < d_boundary_iterators.size(); i++ ) {
        if ( d_boundary_ids[i] == id )
            index = i;
    }
    if ( type2 > NG || gcw > d_max_gcw || index >= d_boundary_iterators.size() )
        return MeshIterator();
    return d_boundary_iterators[index][gcw2][type2];
}
template<uint8_t NG, uint8_t NP>
std::vector<int> TriangleMesh<NG, NP>::getBlockIDs() const
{
    std::vector<int> ids( d_block_iterators.size() );
    for ( size_t i = 0; i < d_block_iterators.size(); i++ )
        ids[i] = i;
    return ids;
}
template<uint8_t NG, uint8_t NP>
MeshIterator
TriangleMesh<NG, NP>::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    int gcw2     = d_comm.getSize() == 1 ? 0 : gcw;
    int type2    = static_cast<int>( type );
    size_t index = d_block_iterators.size();
    for ( size_t i = 0; i < d_block_iterators.size(); i++ ) {
        if ( d_block_ids[i] == id )
            index = i;
    }
    if ( type2 > NG || gcw > d_max_gcw || index >= d_block_iterators.size() )
        return MeshIterator();
    return d_block_iterators[index][gcw2][type2];
}


/****************************************************************
 * Functions to dispace the mesh                                 *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
uint64_t TriangleMesh<NG, NP>::positionHash() const
{
    return d_pos_hash;
}
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::displaceMesh( const std::vector<double> &x )
{
    AMP_ASSERT( x.size() == NP );
    for ( auto &p : d_vert ) {
        for ( size_t d = 0; d < NP; d++ )
            p[d] += x[d];
    }
    for ( auto &p : d_remote_vert ) {
        for ( size_t d = 0; d < NP; d++ )
            p.second[d] += x[d];
    }
    for ( size_t d = 0; d < NP; d++ ) {
        d_box[2 * d + 0] += x[d];
        d_box[2 * d + 1] += x[d];
        d_box_local[2 * d + 0] += x[d];
        d_box_local[2 * d + 1] += x[d];
    }
    d_pos_hash++;
}
#ifdef USE_AMP_VECTORS
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> x )
{
#ifdef USE_AMP_DISCRETIZATION
    // Update the local coordinates
    int rank  = d_comm.getRank();
    auto DOFs = x->getDOFManager();
    std::vector<size_t> dofs;
    double offset[NP];
    for ( size_t i = 0; i < d_vert.size(); i++ ) {
        MeshElementID id( true, AMP::Mesh::GeomType::Vertex, i, rank, d_meshID );
        DOFs->getDOFs( id, dofs );
        AMP_ASSERT( dofs.size() == NP );
        x->getValuesByGlobalID( NP, dofs.data(), offset );
        for ( size_t d = 0; d < NP; d++ )
            d_vert[i][d] += offset[d];
    }
    // Update the remote coordinates
    if ( !d_remote_vert.empty() ) {
        AMP_ERROR( "Not finished" );
    }
    // Update the bounding box
    initializeBoundingBox();
    d_pos_hash++;
#else
    AMP_ERROR( "displaceMesh requires DISCRETIZATION" );
#endif
}
#endif


/****************************************************************
 *  Get the coordinated of the given vertex or the centroid      *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
std::array<double, NP> TriangleMesh<NG, NP>::getPos( const ElementID &id ) const
{
    if ( id.is_local() )
        return d_vert[id.local_id()];
    auto it = d_remote_vert.find( id );
    return it->second;
}


/****************************************************************
 * Return the IDs of the elements composing the current element  *
 ****************************************************************/
template<uint8_t NG, uint8_t NP>
inline void TriangleMesh<NG, NP>::getVerticies( const ElementID &id, int &N, ElementID *IDs ) const
{
    auto type = id.type();
    if ( type == GeomType::Vertex ) {
        N      = 1;
        IDs[0] = id;
        return;
    }
    if ( type == GeomType::Edge ) {
        N = 2;
        Edge edge;
        if ( id.is_local() )
            edge = d_edge[id.local_id()];
        else
            edge = d_remote_edge.find( id )->second;
        IDs[0] = edge[0];
        IDs[1] = edge[1];
    } else if ( type == GeomType::Face ) {
        N = 3;
        Triangle tri;
        if ( id.is_local() )
            tri = d_tri[id.local_id()];
        else
            tri = d_remote_tri.find( id )->second;
        IDs[0] = tri[0];
        IDs[1] = tri[1];
        IDs[2] = tri[2];
    } else if ( type == GeomType::Volume ) {
        N = 4;
        Tetrahedron tet;
        if ( id.is_local() )
            tet = d_tet[id.local_id()];
        else
            tet = d_remote_tet.find( id )->second;
        IDs[0] = tet[0];
        IDs[1] = tet[1];
        IDs[2] = tet[2];
        IDs[3] = tet[3];
    } else {
        AMP_ERROR( "Not finished" );
    }
}
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::getElementsIDs( const ElementID &id,
                                           const GeomType type,
                                           ElementID *IDs ) const
{
    if ( type == id.type() ) {
        IDs[0] = id;
        return;
    }
    if ( type == GeomType::Vertex ) {
        int N_vertex = 0;
        getVerticies( id, N_vertex, IDs );
        return;
    }
    if ( !id.is_local() )
        AMP_ERROR( "Getting children elements is not supported for ghost data" );
    size_t index = id.local_id();
    if ( id.type() == GeomType::Face && type == GeomType::Edge ) {
        for ( int i = 0; i < 3; i++ )
            IDs[i] = d_tri_edge[index][i];
    } else if ( id.type() == GeomType::Volume && type == GeomType::Face ) {
        for ( int i = 0; i < 4; i++ )
            IDs[i] = d_tet_tri[index][i];
    } else if ( id.type() == GeomType::Volume && type == GeomType::Edge ) {
        for ( int i = 0; i < 6; i++ )
            IDs[i] = d_tet_edge[index][i];
    } else {
        AMP_ERROR( "Internal error" );
    }
}


/********************************************************
 *  Get the neighboring elements                         *
 ********************************************************/
template<uint8_t NG, uint8_t NP>
void TriangleMesh<NG, NP>::getNeighborIDs( const ElementID &id, std::vector<ElementID> &IDs ) const
{
    if ( !id.is_local() )
        AMP_ERROR( "Getting neighbors for non-owned elements is not supported" );
    // Check if we are dealing with the largest geometric type
    auto type = id.type();
    if ( static_cast<size_t>( type ) == NG ) {
        IDs.resize( NG + 1 );
        for ( size_t i = 0; i <= NG; i++ )
            IDs[i] = d_neighbors[id.local_id()][i];
        return;
    }
    // The neighbors are any elements that share a parent
    IDs.clear();
    IDs.reserve( 20 );
    int N        = n_Simplex_elements[NG][static_cast<size_t>( type )];
    auto parents = getElementParents( id, static_cast<GeomType>( NG ) );
    for ( auto p = parents.first; p != parents.second; ++p ) {
        ElementID tmp[6];
        getElementsIDs( *p, type, tmp );
        for ( int i = 0; i < N; i++ ) {
            if ( tmp[i] != id && !tmp[i].isNull() )
                IDs.push_back( tmp[i] );
        }
    }
    std::sort( IDs.begin(), IDs.end() );
    IDs.resize( std::unique( IDs.begin(), IDs.end() ) - IDs.begin() );
}


/********************************************************
 *  Check if element is on the boundary, block, etc.     *
 ********************************************************/
template<uint8_t NG, uint8_t NP>
bool TriangleMesh<NG, NP>::isOnSurface( const ElementID &id ) const
{
    if ( static_cast<uint8_t>( id.type() ) == NG ) {
        // Triangle is on the surface if any neighbor is null
        bool test = false;
        if ( id.is_local() ) {
            for ( const auto &tmp : d_neighbors[id.local_id()] )
                test = test || tmp.isNull();
        } else {
            auto it = d_remote_neighbors.find( id );
            for ( const auto &tmp : it->second )
                test = test || tmp.isNull();
        }
        return test;
    } else if ( static_cast<uint8_t>( id.type() ) == NG - 1 ) {
        // Face is on the surface if it has one parent
        auto parents = getElementParents( id, static_cast<GeomType>( NG ) );
        return parents.second - parents.first == 1;
    } else {
        // Node/edge is on the surface if any face is on the surface
        auto parents = getElementParents( id, static_cast<GeomType>( NG - 1 ) );
        for ( auto p = parents.first; p != parents.second; ++p ) {
            if ( isOnSurface( *p ) )
                return true;
        }
    }
    return false;
}
template<uint8_t NG, uint8_t NP>
bool TriangleMesh<NG, NP>::isOnBoundary( const ElementID &elemID, int id ) const
{
    int type     = static_cast<int>( elemID.type() );
    size_t index = d_boundary_iterators.size();
    for ( size_t i = 0; i < d_boundary_iterators.size(); i++ ) {
        if ( d_boundary_ids[i] == id )
            index = i;
    }
    if ( type > NG || index >= d_block_iterators.size() )
        return false;
    const auto &it = d_boundary_iterators[index].back()[type];
    return inIterator( elemID, &it );
}
template<uint8_t NG, uint8_t NP>
bool TriangleMesh<NG, NP>::isInBlock( const ElementID &elemID, int id ) const
{
    if ( static_cast<uint8_t>( elemID.type() ) == NG ) {
        if ( elemID.is_local() ) {
            return id == d_blockID[elemID.local_id()];
        } else {
            auto it = d_remote_blockID.find( elemID );
            AMP_ASSERT( it != d_remote_blockID.end() );
            return id == it->second;
        }
    } else {
        auto parents = getElementParents( elemID, static_cast<GeomType>( NG ) );
        for ( auto p = parents.first; p != parents.second; ++p ) {
            if ( isInBlock( *p, id ) )
                return true;
        }
    }
    return false;
}
template<uint8_t NG, uint8_t NP>
bool TriangleMesh<NG, NP>::inIterator( const ElementID &id, const MeshIterator *it )
{
    if ( it->size() == 0 )
        return false;
    auto type = id.type();
    auto find = []( auto id, auto it ) {
        const auto &list = *it->d_list;
        bool found       = false;
        for ( size_t i = 0; i < list.size(); i++ )
            found = found || list[i] == id;
        return found;
    };
    if ( type == AMP::Mesh::GeomType::Vertex ) {
        auto it2 = dynamic_cast<const TriangleMeshIterator<NG, NP, 0> *>( it );
        AMP_ASSERT( it2 );
        return find( id, it2 );
    } else if ( type == AMP::Mesh::GeomType::Edge ) {
        auto it2 = dynamic_cast<const TriangleMeshIterator<NG, NP, 1> *>( it );
        AMP_ASSERT( it2 );
        return find( id, it2 );
    } else if ( type == AMP::Mesh::GeomType::Face ) {
        auto it2 = dynamic_cast<const TriangleMeshIterator<NG, NP, 2> *>( it );
        AMP_ASSERT( it2 );
        return find( id, it2 );
    } else if ( type == AMP::Mesh::GeomType::Volume ) {
        auto it2 = dynamic_cast<const TriangleMeshIterator<NG, NP, 3> *>( it );
        AMP_ASSERT( it2 );
        return find( id, it2 );
    }
    return false;
}


/********************************************************
 *  Explicit instantiations of TriangleMesh              *
 ********************************************************/
template class TriangleMesh<1, 1>;
template class TriangleMesh<1, 2>;
template class TriangleMesh<1, 3>;
template class TriangleMesh<2, 2>;
template class TriangleMesh<2, 3>;
template class TriangleMesh<3, 3>;


} // namespace Mesh
} // namespace AMP
