#include "AMP/mesh/triangle/TriangleMesh.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/geometry/GeometryHelpers.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/triangle/TriangleHelpers.h"
#include "AMP/mesh/triangle/TriangleMeshIterator.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>


namespace AMP::Mesh {


using Edge        = std::array<ElementID, 2>;
using Triangle    = std::array<ElementID, 3>;
using Tetrahedron = std::array<ElementID, 4>;


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
 * Store the vertex data                                         *
 ****************************************************************/
static inline std::vector<std::array<double, 3>> convert( const std::vector<double> &x )
{
    std::vector<std::array<double, 3>> x2( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        x2[i] = { x[i], 0, 0 };
    return x2;
}
static inline std::vector<std::array<double, 3>>
convert( const std::vector<std::array<double, 2>> &x )
{
    std::vector<std::array<double, 3>> x2( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        x2[i] = { x[i][0], x[i][1], 0 };
    return x2;
}
template<class TYPE, size_t N>
StoreTriData<TYPE, N>::StoreTriData( std::vector<std::array<TYPE, N>> x,
                                     std::vector<int> offset,
                                     int rank,
                                     GeomType type )
    : d_type( type ),
      d_start( offset[rank] ),
      d_end( offset[rank + 1] ),
      d_rank( rank ),
      d_offset( std::move( offset ) ),
      d_x( std::move( x ) )
{
}
template<class TYPE, size_t N>
int StoreTriData<TYPE, N>::rank( int i ) const
{
    return i >= d_start && i < d_end ? d_rank : AMP::Utilities::findfirst( d_offset, i + 1 ) - 1;
}
template<class TYPE, size_t N>
ElementID StoreTriData<TYPE, N>::getID( int i ) const
{
    if ( i >= d_start && i < d_end ) {
        int local = i - d_start;
        return ElementID( true, d_type, local, d_rank );
    } else {
        int rank  = AMP::Utilities::findfirst( d_offset, i + 1 ) - 1;
        int local = i - d_offset[rank];
        return ElementID( false, d_type, local, rank );
    }
}


/****************************************************************
 * Get the children from an element                              *
 ****************************************************************/
template<uint8_t N1, uint8_t N2>
static std::array<std::array<int, N2 + 1>, n_Simplex_elements[N1][N2]>
getChildren( const std::array<int, N1 + 1> &parent )
{
    std::array<std::array<int, N2 + 1>, n_Simplex_elements[N1][N2]> children;
    if constexpr ( N1 == N2 ) {
        children[0] = parent;
    } else if constexpr ( N2 == 0 ) {
        for ( size_t i = 0; i < N1 + 1; i++ )
            children[i] = parent[i];
    } else if constexpr ( N2 == 1 ) {
        int k = 0;
        for ( size_t i = 0; i < N1; i++ ) {
            for ( size_t j = i + 1; j <= N1; j++ ) {
                if ( parent[i] < parent[j] )
                    children[k++] = { parent[i], parent[j] };
                else
                    children[k++] = { parent[j], parent[i] };
            }
        }
        for ( size_t i = 0; i < children.size(); i++ ) {
            if ( children[i][0] > children[i][1] )
                std::swap( children[i][0], children[i][1] );
        }
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


/****************************************************************
 * Remove unused vertices                                       *
 ****************************************************************/
template<uint8_t NG>
static void removeUnusedVerticies( std::vector<std::array<double, 3>> &vertices,
                                   std::vector<std::array<int, NG + 1>> &tri )
{
    // Check which vertices are used
    std::vector<bool> used( vertices.size() );
    for ( auto &t : tri ) {
        for ( auto i : t )
            used[i] = true;
    }
    // Create a map to renumber and remove unused vertices
    std::vector<size_t> map( used.size(), static_cast<size_t>( -1 ) );
    size_t N = 0;
    for ( size_t i = 0; i < used.size(); i++ ) {
        if ( used[i] ) {
            map[i]      = N;
            vertices[N] = vertices[i];
            N++;
        }
    }
    vertices.resize( N );
    // Renumber triangles
    for ( auto &t : tri ) {
        for ( auto &i : t )
            i = map[i];
    }
}


/********************************************************
 * Get the iterator list                                 *
 ********************************************************/
template<uint8_t NG>
static inline std::shared_ptr<const std::vector<ElementID>> getList( const MeshIterator &it0 )
{
    auto it = it0.rawIterator();
    if ( dynamic_cast<const TriangleMeshIterator<NG, 0> *>( it ) )
        return dynamic_cast<const TriangleMeshIterator<NG, 0> *>( it )->getList();
    if ( dynamic_cast<const TriangleMeshIterator<NG, 1> *>( it ) )
        return dynamic_cast<const TriangleMeshIterator<NG, 1> *>( it )->getList();
    if constexpr ( NG >= 2 ) {
        if ( dynamic_cast<const TriangleMeshIterator<NG, 2> *>( it ) )
            return dynamic_cast<const TriangleMeshIterator<NG, 2> *>( it )->getList();
    }
    if constexpr ( NG >= 3 ) {
        if ( dynamic_cast<const TriangleMeshIterator<NG, 3> *>( it ) )
            return dynamic_cast<const TriangleMeshIterator<NG, 3> *>( it )->getList();
    }
    return nullptr;
}


/****************************************************************
 * Perform load balancing                                        *
 * At entry:                                                     *
 *    Initial triangle/vertex info on rank 0                     *
 * At exit:                                                      *
 *    d_vertex:                                                  *
 *       will be filled with global vertices                     *
 *       vertices will be sorted by global rank                  *
 *    d_startVertex:                                             *
 *       will be filled with starting index for each rank        *
 *    d_startTri:                                                *
 *       will be filled with starting index for each rank        *
 *    d_globalTri:                                               *
 *       will be filled with global triangles                    *
 *       triangles will be sorted by global rank                 *
 *    d_globalNab:                                               *
 *       will be filled with global triangle neighbors           *
 *    d_blockID:                                                 *
 *       will be filled with block ides for each triangle        *
 ****************************************************************/
template<uint8_t NG>
void TriangleMesh<NG>::loadBalance( const std::vector<Point> &vertices,
                                    const std::vector<TRI> &tri,
                                    const std::vector<TRI> &tri_nab,
                                    const std::vector<int> &block )
{
    AMP_ASSERT( !vertices.empty() );
    AMP_ASSERT( !tri.empty() && tri.size() == tri_nab.size() && tri.size() == block.size() );
    // Get the owner rank for each node
    auto ranks = AMP::Geometry::GeometryHelpers::assignRanks( vertices, d_comm.getSize() );
    // Reorder the vertices so they are stored grouped by rank
    std::vector<int> I( vertices.size() );
    std::iota( I.begin(), I.end(), 0 );
    std::stable_sort(
        I.begin(), I.end(), [&ranks]( size_t i1, size_t i2 ) { return ranks[i1] < ranks[i2]; } );
    std::vector<Point> x( vertices.size() );
    for ( size_t i = 0; i < vertices.size(); i++ )
        x[i] = vertices[I[i]];
    auto tri2 = tri;
    tri2.resize( tri.size() );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( uint8_t d = 0; d < NG + 1; d++ )
            tri2[i][d] = I[tri[i][d]];
    }
    // Get the ranks
    std::vector<int> offset( d_comm.getSize() + 1, 0 );
    for ( auto r : ranks )
        offset[r + 1]++;
    for ( size_t i = 1; i < offset.size(); i++ )
        offset[i] += offset[i - 1];
    AMP_ASSERT( offset[d_comm.getSize()] == (int) x.size() );
    d_vertex = StoreTriData( x, offset, d_comm.getRank(), GeomType::Vertex );
    // Load balance the triangles
    d_globalNab.resize( tri.size() );
    d_blockID.resize( tri.size(), 0 );
    I = AMP::DelaunayHelpers::sortTri<NG>( tri2 );
    for ( size_t i = 0; i < tri.size(); i++ )
        d_globalNab[i] = tri_nab[I[i]];
    if ( !block.empty() ) {
        for ( size_t i = 0; i < tri.size(); i++ )
            d_blockID[i] = block[I[i]];
    }
    ranks.resize( tri.size() );
    for ( size_t i = 0; i < tri.size(); i++ )
        ranks[i] = d_vertex.rank( tri2[i][0] );
    // Store the triangle data
    offset.clear();
    offset.resize( d_comm.getSize() + 1, 0 );
    for ( auto r : ranks )
        offset[r + 1]++;
    for ( size_t i = 1; i < offset.size(); i++ )
        offset[i] += offset[i - 1];
    AMP_ASSERT( offset[d_comm.getSize()] == (int) tri.size() );
    d_globalTri = StoreTriData( tri2, offset, d_comm.getRank(), static_cast<GeomType>( NG ) );
}


/****************************************************************
 * Generator                                                     *
 ****************************************************************/
template<uint8_t NG>
template<uint8_t NP>
std::shared_ptr<TriangleMesh<NG>>
TriangleMesh<NG>::generate( const std::vector<std::array<double, NP>> &vert,
                            const std::vector<TRI> &tri,
                            const std::vector<TRI> &tri_nab,
                            const AMP_MPI &comm,
                            std::shared_ptr<Geometry::Geometry> geom,
                            std::vector<int> block )
{
    if ( comm.getRank() != 0 )
        AMP_INSIST( vert.empty() && tri.empty() && tri_nab.empty(),
                    "Initial triangle list must only be on rank 0" );
    std::vector<std::array<double, 3>> v2;
    if constexpr ( NP == 3 ) {
        v2 = vert;
    } else {
        v2.resize( vert.size(), { 0, 0, 0 } );
        for ( size_t i = 0; i < vert.size(); i++ ) {
            for ( uint8_t d = 0; d < NP; d++ )
                v2[i][d] = vert[i][d];
        }
    }
    std::shared_ptr<TriangleMesh<NG>> mesh(
        new TriangleMesh<NG>( NP, std::move( v2 ), tri, tri_nab, comm, geom, std::move( block ) ) );
    return mesh;
}
template<uint8_t NG>
template<uint8_t NP>
std::shared_ptr<TriangleMesh<NG>>
TriangleMesh<NG>::generate( const std::vector<std::array<std::array<double, NP>, NG + 1>> &tri_list,
                            const AMP_MPI &comm,
                            double tol )
{
    // Get the global list of tri_list
    auto global_list = comm.allGather( tri_list );
    std::vector<std::array<double, NP>> vertices;
    std::vector<TRI> triangles;
    std::vector<TRI> neighbors;
    if ( comm.getRank() == 0 ) {
        // Create triangles from the points
        TriangleHelpers::createTriangles<NG, NP>( global_list, vertices, triangles, tol );
        // Find the number of unique triangles (duplicates may indicate multiple objects)
        size_t N2 = TriangleHelpers::count<NG>( triangles );
        AMP_INSIST( N2 == tri_list.size(),
                    "Duplicate triangles detected, no connectivity information will be stored" );
        // Get the triangle neighbors
        neighbors = DelaunayHelpers::create_tri_neighbors<NG>( triangles );
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
    }
    // Create the mesh
    std::vector<std::array<double, 3>> v2;
    if constexpr ( NP == 3 ) {
        v2 = std::move( vertices );
    } else {
        v2.resize( vertices.size(), { 0, 0, 0 } );
        for ( size_t i = 0; i < vertices.size(); i++ ) {
            for ( uint8_t d = 0; d < NP; d++ )
                v2[i][d] = vertices[i][d];
        }
    }
    std::shared_ptr<TriangleMesh<NG>> mesh( new TriangleMesh<NG>( NP,
                                                                  std::move( v2 ),
                                                                  std::move( triangles ),
                                                                  std::move( neighbors ),
                                                                  comm,
                                                                  nullptr,
                                                                  std::vector<int>() ) );
    return mesh;
}
template<uint8_t NG>
TriangleMesh<NG>::TriangleMesh( int NP,
                                std::vector<Point> vertices,
                                std::vector<TRI> tri,
                                std::vector<TRI> tri_nab,
                                const AMP_MPI &comm,
                                std::shared_ptr<Geometry::Geometry> geom_in,
                                std::vector<int> block,
                                int max_gcw )
    : d_pos_hash( 0 )
{
    // Set basic mesh info
    d_geometry  = nullptr;
    GeomDim     = static_cast<GeomType>( NG );
    PhysicalDim = NP;
    d_max_gcw   = max_gcw;
    d_comm      = comm;
    d_name      = "NULL";
    d_geometry  = std::move( geom_in );
    setMeshID();
    // Run some basic checks
    if ( block.empty() )
        block = std::vector<int>( tri.size(), 0 );
    if ( d_comm.getSize() > 1 ) {
        AMP_ASSERT( tri_nab.size() == tri.size() );
        AMP_ASSERT( block.size() == tri.size() );
        if ( d_comm.anyReduce( tri.empty() ) ) {
            size_t N_vertex = d_comm.bcast( vertices.size(), 0 );
            size_t N_tri    = d_comm.bcast( tri.size(), 0 );
            vertices.resize( N_vertex );
            tri.resize( N_tri );
            tri_nab.resize( N_tri );
            block.resize( N_tri );
            d_comm.bcast( vertices.data(), N_vertex, 0 );
            d_comm.bcast( tri.data(), N_tri, 0 );
            d_comm.bcast( tri_nab.data(), N_tri, 0 );
            d_comm.bcast( block.data(), N_tri, 0 );
        }
    }
    // Check for duplicate entries in the triangle
    for ( const auto &t : tri ) {
        for ( size_t i = 1; i <= NG; i++ )
            for ( size_t j = 0; j < i; j++ )
                AMP_ASSERT( t[i] != t[j] );
    }
    // Remove vertices that are not used
    removeUnusedVerticies<NG>( vertices, tri );
    // Perform the load balancing
    loadBalance( vertices, tri, tri_nab, block );
    // Initialize the iterators and some common data
    initialize();
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
template<uint8_t NG>
void TriangleMesh<NG>::initializeBoundingBox()
{
    // Initialize the bounding box
    d_box.resize( 6, 0 );
    d_box_local.resize( 6, 0 );
    for ( size_t d = 0; d < 3; d++ ) {
        d_box_local[2 * d + 0] = 1e100;
        d_box_local[2 * d + 1] = -1e100;
    }
    for ( int i = d_vertex.start(); i < d_vertex.end(); i++ ) {
        for ( size_t d = 0; d < 3; d++ ) {
            d_box_local[2 * d + 0] = std::min( d_box_local[2 * d + 0], d_vertex[i][d] );
            d_box_local[2 * d + 1] = std::max( d_box_local[2 * d + 1], d_vertex[i][d] );
        }
    }
    for ( int i = 0; i < d_vertex.size(); i++ ) {
        for ( size_t d = 0; d < 3; d++ ) {
            d_box[2 * d + 0] = std::min( d_box[2 * d + 0], d_vertex[i][d] );
            d_box[2 * d + 1] = std::max( d_box[2 * d + 1], d_vertex[i][d] );
        }
    }
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
template<uint8_t NG>
void TriangleMesh<NG>::initialize()
{
    // Create the local triangles
    size_t N_tri = d_globalTri.end() - d_globalTri.start();
    std::vector<std::array<ElementID, NG + 1>> tri2( N_tri );
    for ( size_t i = 0, i2 = d_globalTri.start(); i < N_tri; i++, i2++ ) {
        for ( size_t d = 0; d <= NG; d++ ) {
            int node = d_globalTri[i2][d];
            if ( node == -1 )
                continue;
            tri2[i][d] = d_vertex.getID( node );
        }
    }
    // Create the remote triangles (store only the new values)
    d_remoteTri.resize( d_max_gcw );
    if ( d_max_gcw > 0 && d_comm.getSize() > 1 ) {
        std::set<int> list;
        int start = d_globalTri.start();
        int end   = d_globalTri.end();
        auto add  = [&list, start, end, &nab = d_globalNab]( int i, std::vector<int> &remote ) {
            for ( auto t : nab[i] ) {
                if ( t != -1 && ( t < start || t >= end ) ) {
                    if ( list.find( t ) == list.end() ) {
                        list.insert( t );
                        remote.push_back( t );
                    }
                }
            }
        };
        for ( int i = start; i < end; i++ )
            add( i, d_remoteTri[0] );
        AMP::Utilities::quicksort( d_remoteTri[0] );
        for ( int g = 1; g < d_max_gcw; g++ ) {
            for ( int i : d_remoteTri[g - 1] )
                add( i, d_remoteTri[g] );
            AMP::Utilities::quicksort( d_remoteTri[g] );
        }
    }
    // Get the bounding boxes
    initializeBoundingBox();
    // Initialize child data
    buildChildren();
    // Initialize the iterators
    initializeIterators();
    // Get the global size
    d_N_global[0]  = d_vertex.size();
    d_N_global[NG] = d_globalTri.size();
    for ( int i = 1; i < NG; i++ )
        d_N_global[i] = d_comm.sumReduce( d_iterators[0][i].size() );
    for ( int i = NG + 1; i < 4; i++ )
        d_N_global[i] = 0;
}
template<uint8_t NG>
void TriangleMesh<NG>::initializeIterators()
{
    int size    = d_comm.getSize();
    int max_gcw = size == 1 ? 0 : d_max_gcw;
    d_iterators.resize( max_gcw + 1 );
    for ( int gcw = 0; gcw <= max_gcw; gcw++ ) {
        for ( int type = 0; type <= NG; type++ )
            d_iterators[gcw][type] = createIterator( static_cast<GeomType>( type ), gcw );
    }
    // Compute the parents
    if constexpr ( NG >= 1 ) {
        d_parents[0][1] = computeNodeParents( d_iterators[max_gcw][1] );
    }
    if constexpr ( NG >= 2 ) {
        d_parents[0][2] = computeNodeParents( d_iterators[max_gcw][2] );
        d_parents[1][2] = getParents( 1, d_iterators[max_gcw][2] );
    }
    if constexpr ( NG >= 3 ) {
        d_parents[0][3] = computeNodeParents( d_iterators[max_gcw][3] );
        d_parents[1][3] = getParents( 1, d_iterators[max_gcw][3] );
        d_parents[2][3] = getParents( 2, d_iterators[max_gcw][3] );
    }

    // Create the block iterators
    std::set<int> blockSet( d_blockID.begin(), d_blockID.end() );
    d_block_ids = std::vector<int>( blockSet.begin(), blockSet.end() );
    d_block_it.resize( d_block_ids.size() );
    for ( size_t i = 0; i < d_block_ids.size(); i++ ) {
        auto id0 = d_block_ids[i];
        d_block_it[i].resize( d_iterators.size() );
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
                d_block_it[i][gcw][type] = createIterator( list_ptr );
            }
        }
    }
    // Create the surface iterators
    d_surface_it.resize( d_iterators.size() );
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
            d_surface_it[gcw][type] = createIterator( list_ptr );
        }
    }
    // Create the boundary iterators
    if ( d_geometry ) {
        int Ns = d_geometry->NSurface();
        d_boundary_ids.resize( Ns );
        d_boundary_it.resize( Ns );
        for ( int i = 0; i < Ns; i++ ) {
            d_boundary_ids[i] = i;
            d_boundary_it[i].resize( d_iterators.size() );
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
                    d_boundary_it[i][gcw][type2] = createIterator( ptr );
                }
            }
        }
    } else {
        if ( d_surface_it.back()[0].size() == 0 ) {
            d_boundary_ids.clear();
            d_boundary_it.clear();
        } else {
            d_boundary_ids = std::vector<int>( 1, 0 );
            d_boundary_it.resize( 1 );
            d_boundary_it[0] = d_surface_it;
        }
    }
}
template<uint8_t NG>
MeshIterator TriangleMesh<NG>::createIterator( std::shared_ptr<std::vector<ElementID>> list ) const
{
    if ( list->empty() )
        return MeshIterator();
    auto type = ( *list )[0].type();
    bool test = true;
    for ( const auto &id : *list )
        test = test && id.type() == type;
    AMP_ASSERT( test );
    if ( type == GeomType::Vertex )
        return TriangleMeshIterator<NG, 0>( this, list );
    if constexpr ( NG >= 1 ) {
        if ( type == GeomType::Edge )
            return TriangleMeshIterator<NG, 1>( this, list );
    }
    if constexpr ( NG >= 2 ) {
        if ( type == GeomType::Face )
            return TriangleMeshIterator<NG, 2>( this, list );
    }
    if constexpr ( NG >= 3 ) {
        if ( type == GeomType::Cell )
            return TriangleMeshIterator<NG, 3>( this, list );
    }
    AMP_ERROR( "Internal error" );
}
template<uint8_t NG>
MeshIterator TriangleMesh<NG>::createIterator( GeomType type, int gcw ) const
{
    if ( static_cast<int>( type ) > NG || gcw > d_max_gcw )
        return MeshIterator();
    int myRank = d_comm.getRank();
    std::vector<int> tri_list( d_globalTri.end() - d_globalTri.start() );
    std::iota( tri_list.begin(), tri_list.end(), d_globalTri.start() );
    for ( int g = 0; g < gcw; g++ )
        tri_list.insert( tri_list.end(), d_remoteTri[g].begin(), d_remoteTri[g].end() );
    if ( static_cast<uint8_t>( type ) == NG ) {
        auto elements = std::make_shared<std::vector<ElementID>>();
        elements->reserve( tri_list.size() );
        for ( int tri : tri_list )
            elements->push_back( d_globalTri.getID( tri ) );
        AMP::Utilities::quicksort( *elements );
        return TriangleMeshIterator<NG, NG>( this, elements );
    } else if ( type == GeomType::Vertex ) {
        std::set<int> node_set;
        for ( int i : tri_list ) {
            for ( auto n : d_globalTri[i] )
                node_set.insert( n );
        }
        auto elements = std::make_shared<std::vector<ElementID>>();
        elements->reserve( node_set.size() );
        for ( int node : node_set )
            elements->push_back( d_vertex.getID( node ) );
        AMP::Utilities::quicksort( *elements );
        return TriangleMeshIterator<NG, 0>( this, elements );
    } else if ( type == GeomType::Edge ) {
        if constexpr ( NG > 1 ) {
            auto elements = std::make_shared<std::vector<ElementID>>();
            // Create local edges
            auto &vec = *elements;
            vec.reserve( d_childEdge.size() );
            for ( size_t i = 0; i < d_childEdge.size(); i++ )
                vec.emplace_back( true, GeomType::Edge, i, myRank );
            // Add ghost values
            if ( gcw > 0 ) {
                int start = d_vertex.start();
                int end   = d_vertex.end();
                auto it   = createIterator( static_cast<GeomType>( NG ), gcw );
                auto it2  = dynamic_cast<const TriangleMeshIterator<NG, NG> *>( it.rawIterator() );
                AMP_ASSERT( it2 );
                for ( auto id : *( it2->d_list ) ) {
                    for ( auto &child : getChildren<NG, 1>( d_globalTri[id] ) ) {
                        if ( child[0] < start || child[0] >= end ) {
                            for ( auto [id, t] : d_remoteEdge ) {
                                if ( t == child ) {
                                    vec.push_back( id );
                                }
                            }
                        }
                    }
                }
            }
            AMP::Utilities::quicksort( *elements );
            return TriangleMeshIterator<NG, 1>( this, elements );
        }
    } else if ( type == GeomType::Face ) {
        if constexpr ( NG > 2 ) {
            auto elements = std::make_shared<std::vector<ElementID>>();
            // Create local edges
            auto &vec = *elements;
            vec.reserve( d_childFace.size() );
            for ( size_t i = 0; i < d_childFace.size(); i++ )
                vec.emplace_back( true, GeomType::Face, i, myRank );
            // Add ghost values
            if ( gcw > 0 ) {
                int start = d_vertex.start();
                int end   = d_vertex.end();
                auto it   = createIterator( static_cast<GeomType>( NG ), gcw );
                auto it2  = dynamic_cast<const TriangleMeshIterator<NG, NG> *>( it.rawIterator() );
                AMP_ASSERT( it2 );
                for ( auto id : *( it2->d_list ) ) {
                    for ( auto &child : getChildren<NG, 2>( d_globalTri[id] ) ) {
                        if ( child[0] < start || child[0] >= end ) {
                            for ( auto [id, t] : d_remoteFace ) {
                                if ( t == child ) {
                                    vec.push_back( id );
                                }
                            }
                        }
                    }
                }
            }
            AMP::Utilities::quicksort( *elements );
            return TriangleMeshIterator<NG, 2>( this, elements );
        }
    }
    AMP_ERROR( "Internal error" );
    return MeshIterator();
}


/********************************************************
 * Build the children data                               *
 ********************************************************/
template<uint8_t NG>
void TriangleMesh<NG>::buildChildren()
{
    if constexpr ( NG == 1 )
        return;
    int myRank = d_comm.getRank();
    int N_tri  = d_globalTri.end() - d_globalTri.start();
    int start  = d_vertex.start();
    int end    = d_vertex.end();
    // Build local children
    d_childEdge.clear();
    d_childFace.clear();
    d_childEdge.reserve( N_tri );
    d_childFace.reserve( N_tri );
    std::vector<Edge> remoteEdges;
    std::vector<Face> remoteFaces;
    auto list = getList<NG>( createIterator( static_cast<GeomType>( NG ), d_max_gcw ) );
    for ( size_t i = 0; i < list->size(); i++ ) {
        auto id  = list->operator[]( i );
        auto tri = d_globalTri[id];
        if constexpr ( NG >= 2 ) {
            auto children = getChildren<NG, 1>( tri );
            for ( auto child : children ) {
                if ( child[0] >= start && child[0] < end )
                    d_childEdge.push_back( child );
                else
                    remoteEdges.push_back( child );
            }
        }
        if constexpr ( NG >= 3 ) {
            auto children = getChildren<NG, 2>( tri );
            for ( auto child : children ) {
                if ( child[0] >= start && child[0] < end )
                    d_childFace.push_back( child );
                else
                    remoteFaces.push_back( child );
            }
        }
    }
    d_childEdge = AMP::DelaunayHelpers::uniqueTri<1>( d_childEdge );
    d_childFace = AMP::DelaunayHelpers::uniqueTri<2>( d_childFace );
    if ( d_comm.getSize() == 1 )
        return;
    // Build remote children
    std::map<std::array<int, 2>, ElementID> edgeIDs;
    std::map<std::array<int, 3>, ElementID> faceIDs;
    for ( size_t i = 0; i < d_childEdge.size(); i++ )
        edgeIDs[d_childEdge[i]] = ElementID( true, GeomType::Edge, i, myRank );
    for ( size_t i = 0; i < d_childFace.size(); i++ )
        faceIDs[d_childFace[i]] = ElementID( true, GeomType::Face, i, myRank );
    d_comm.mapGather( edgeIDs );
    d_comm.mapGather( faceIDs );
    if constexpr ( NG >= 2 ) {
        for ( auto edge : remoteEdges ) {
            auto it = edgeIDs.find( edge );
            AMP_ASSERT( it != edgeIDs.end() );
            int rank  = it->second.owner_rank();
            int index = it->second.local_id();
            ElementID id( rank == myRank, GeomType::Edge, index, rank );
            d_remoteEdge[id] = edge;
        }
    }
    if constexpr ( NG >= 3 ) {
        for ( auto face : remoteFaces ) {
            auto it = faceIDs.find( face );
            AMP_ASSERT( it != faceIDs.end() );
            int rank  = it->second.owner_rank();
            int index = it->second.local_id();
            ElementID id( rank == myRank, GeomType::Face, index, rank );
            d_remoteFace[id] = face;
        }
    }
}


/********************************************************
 * Build the parent data                                 *
 ********************************************************/
template<uint8_t NG>
StoreCompressedList<ElementID> TriangleMesh<NG>::computeNodeParents( const MeshIterator &it )
{
    int start = d_vertex.start();
    int end   = d_vertex.end();
    auto list = getList<NG>( it );
    if ( list->empty() )
        return StoreCompressedList<ElementID>( end - start );
    auto type = list->front().type();
    std::vector<std::vector<ElementID>> parents( end - start );
    if ( type == GeomType::Edge ) {
        for ( auto id : *list ) {
            for ( auto node : getElem<1>( id ) ) {
                if ( node >= start && node < end )
                    parents[node - start].push_back( id );
            }
        }
    }
    if constexpr ( NG >= 2 ) {
        if ( type == GeomType::Face ) {
            for ( auto id : *list ) {
                for ( auto node : getElem<2>( id ) ) {
                    if ( node >= start && node < end )
                        parents[node - start].push_back( id );
                }
            }
        }
    }
    if constexpr ( NG >= 3 ) {
        if ( type == GeomType::Cell ) {
            for ( auto id : *list ) {
                for ( auto node : getElem<3>( id ) ) {
                    if ( node >= start && node < end )
                        parents[node - start].push_back( id );
                }
            }
        }
    }
    // Check that every node has a parent
    for ( size_t i = 0; i < parents.size(); i++ )
        AMP_ASSERT( !parents[i].empty() );
    // Return the parents
    return StoreCompressedList<ElementID>( parents );
}
template<uint8_t NG>
StoreCompressedList<ElementID> TriangleMesh<NG>::getParents( int childType, const MeshIterator &it )
{
    auto list   = getList<NG>( it );
    int N_local = numLocalElements( static_cast<GeomType>( childType ) );
    if ( list->empty() )
        return StoreCompressedList<ElementID>( N_local );
    auto parentType = static_cast<int>( list->front().type() );
    AMP_ASSERT( childType < parentType );
    if ( childType == 0 )
        return computeNodeParents( it );
    std::vector<std::vector<ElementID>> parents( N_local );
    if constexpr ( NG >= 2 ) {
        if ( childType == 1 && parentType == 2 ) {
            for ( auto id : *list ) {
                auto elem = getElem<2>( id );
                for ( const auto &child : getChildren<2, 1>( elem ) ) {
                    auto id2 = getID<1>( child );
                    if ( id2.is_local() )
                        parents[id2.local_id()].push_back( id );
                }
            }
        }
    }
    if constexpr ( NG >= 3 ) {
        if ( childType == 1 && parentType == 3 ) {
            for ( auto id : *list ) {
                auto elem = getElem<3>( id );
                for ( const auto &child : getChildren<3, 1>( elem ) ) {
                    auto id2 = getID<1>( child );
                    if ( id2.is_local() )
                        parents[id2.local_id()].push_back( id );
                }
            }
        }
        if ( childType == 2 && parentType == 3 ) {
            for ( auto id : *list ) {
                auto elem = getElem<3>( id );
                for ( const auto &child : getChildren<3, 2>( elem ) ) {
                    auto id2 = getID<2>( child );
                    if ( id2.is_local() )
                        parents[id2.local_id()].push_back( id );
                }
            }
        }
    }
    // Check that every node has a parent
    for ( size_t i = 0; i < parents.size(); i++ )
        AMP_ASSERT( !parents[i].empty() );
    // Return the parents
    return StoreCompressedList<ElementID>( parents );
}


/********************************************************
 * Return the class name                                 *
 ********************************************************/
template<uint8_t NG>
std::string TriangleMesh<NG>::meshClass() const
{
    return Utilities::stringf( "TriangleMesh<%u>", NG );
}


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
template<uint8_t NG>
TriangleMesh<NG>::TriangleMesh( std::shared_ptr<const MeshParameters> params_in )
    : Mesh( params_in )
{
    // Check for valid inputs
    AMP_INSIST( !d_comm.isNull(), "Communicator must be set" );
    AMP_ERROR( "Not finished" );
}
template<uint8_t NG>
TriangleMesh<NG>::TriangleMesh( const TriangleMesh &rhs )
    : Mesh( rhs ),
      d_N_global{ rhs.d_N_global },
      d_vertex( rhs.d_vertex ),
      d_globalTri( rhs.d_globalTri ),
      d_globalNab( rhs.d_globalNab ),
      d_blockID( rhs.d_blockID ),
      d_remoteTri( rhs.d_remoteTri ),
      d_childEdge( rhs.d_childEdge ),
      d_childFace( rhs.d_childFace ),
      d_remoteEdge( rhs.d_remoteEdge ),
      d_remoteFace( rhs.d_remoteFace ),
      d_block_ids( rhs.d_block_ids ),
      d_boundary_ids( rhs.d_boundary_ids ),
      d_pos_hash( 0 )
{
    for ( size_t i = 0; i < NG; i++ ) {
        for ( size_t j = 0; j <= NG; j++ ) {
            d_parents[i][j] = rhs.d_parents[i][j];
        }
    }
    initializeIterators();
}
template<uint8_t NG>
std::unique_ptr<Mesh> TriangleMesh<NG>::clone() const
{
    return std::unique_ptr<TriangleMesh<NG>>( new TriangleMesh<NG>( *this ) );
}


/****************************************************************
 * De-constructor                                                *
 ****************************************************************/
template<uint8_t NG>
TriangleMesh<NG>::~TriangleMesh() = default;


/****************************************************************
 * Function to return the element given an ID                    *
 ****************************************************************/
template<uint8_t NG>
MeshElement *TriangleMesh<NG>::getElement2( const MeshElementID &id ) const
{
    if ( id.type() == AMP::Mesh::GeomType::Vertex )
        return new TriangleMeshElement<NG, 0>( id, this );
    if constexpr ( NG > 0 )
        if ( id.type() == AMP::Mesh::GeomType::Edge )
            return new TriangleMeshElement<NG, 1>( id, this );
    if constexpr ( NG > 1 )
        if ( id.type() == AMP::Mesh::GeomType::Face )
            return new TriangleMeshElement<NG, 2>( id, this );
    if constexpr ( NG > 2 )
        if ( id.type() == AMP::Mesh::GeomType::Cell )
            return new TriangleMeshElement<NG, 3>( id, this );
    return nullptr;
}
template<uint8_t NG>
MeshElement TriangleMesh<NG>::getElement( const MeshElementID &id ) const
{
    return MeshElement( getElement2( id ) );
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
template<uint8_t NG>
std::pair<const ElementID *, const ElementID *>
TriangleMesh<NG>::getElementParents( const ElementID &id, const GeomType type ) const
{
    auto type1 = static_cast<size_t>( id.type() );
    auto type2 = static_cast<size_t>( type );
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
template<uint8_t NG>
std::vector<MeshElement> TriangleMesh<NG>::getElementParents( const MeshElement &elem,
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
template<uint8_t NG>
size_t TriangleMesh<NG>::numLocalElements( const GeomType type ) const
{
    int t = static_cast<int>( type );
    if ( t > NG )
        return 0;
    return d_iterators[0][t].size();
}
template<uint8_t NG>
size_t TriangleMesh<NG>::numGlobalElements( const GeomType type ) const
{
    int t = static_cast<int>( type );
    if ( t <= NG )
        return d_N_global[t];
    return 0;
}
template<uint8_t NG>
size_t TriangleMesh<NG>::numGhostElements( const GeomType type, int gcw ) const
{
    if ( gcw == 0 || d_comm.getSize() == 1 )
        return 0;
    int t = static_cast<int>( type );
    return d_iterators[gcw][t].size() - d_iterators[0][t].size();
}


/****************************************************************
 * Function to get an iterator                                   *
 ****************************************************************/
template<uint8_t NG>
MeshIterator TriangleMesh<NG>::getIterator( const GeomType type, const int gcw ) const
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
template<uint8_t NG>
MeshIterator TriangleMesh<NG>::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    int gcw2  = d_comm.getSize() == 1 ? 0 : gcw;
    int type2 = static_cast<int>( type );
    if ( type2 > NG || gcw > d_max_gcw )
        return MeshIterator();
    return d_surface_it[gcw2][type2];
}


/****************************************************************
 * Functions to get the boundaries                               *
 ****************************************************************/
template<uint8_t NG>
std::vector<int> TriangleMesh<NG>::getBoundaryIDs() const
{
    return d_boundary_ids;
}
template<uint8_t NG>
MeshIterator
TriangleMesh<NG>::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    int gcw2     = d_comm.getSize() == 1 ? 0 : gcw;
    int type2    = static_cast<int>( type );
    size_t index = d_boundary_it.size();
    for ( size_t i = 0; i < d_boundary_it.size(); i++ ) {
        if ( d_boundary_ids[i] == id )
            index = i;
    }
    if ( type2 > NG || gcw > d_max_gcw || index >= d_boundary_it.size() )
        return MeshIterator();
    return d_boundary_it[index][gcw2][type2];
}
template<uint8_t NG>
std::vector<int> TriangleMesh<NG>::getBlockIDs() const
{
    std::vector<int> ids( d_block_it.size() );
    for ( size_t i = 0; i < d_block_it.size(); i++ )
        ids[i] = i;
    return ids;
}
template<uint8_t NG>
MeshIterator
TriangleMesh<NG>::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    int gcw2     = d_comm.getSize() == 1 ? 0 : gcw;
    int type2    = static_cast<int>( type );
    size_t index = d_block_it.size();
    for ( size_t i = 0; i < d_block_it.size(); i++ ) {
        if ( d_block_ids[i] == id )
            index = i;
    }
    if ( type2 > NG || gcw > d_max_gcw || index >= d_block_it.size() )
        return MeshIterator();
    return d_block_it[index][gcw2][type2];
}


/****************************************************************
 * Functions to dispace the mesh                                 *
 ****************************************************************/
template<uint8_t NG>
uint64_t TriangleMesh<NG>::positionHash() const
{
    return d_pos_hash;
}
template<uint8_t NG>
void TriangleMesh<NG>::displaceMesh( const std::vector<double> &x )
{
    AMP_ASSERT( x.size() <= 3 );
    for ( int i = 0; i < d_vertex.size(); i++ ) {
        for ( size_t d = 0; d < x.size(); d++ )
            d_vertex[i][d] += x[d];
    }
    for ( size_t d = 0; d < x.size(); d++ ) {
        d_box[2 * d + 0] += x[d];
        d_box[2 * d + 1] += x[d];
        d_box_local[2 * d + 0] += x[d];
        d_box_local[2 * d + 1] += x[d];
    }
    d_pos_hash++;
}
template<uint8_t NG>
void TriangleMesh<NG>::displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> x )
{
    int rank  = d_comm.getRank();
    int start = d_vertex.start();
    // Get the updated local coordinates
    std::vector<Point> local( d_vertex.end() - start );
    auto DOFs = x->getDOFManager();
    std::vector<size_t> dofs;
    double offset[3] = { 0, 0, 0 };
    for ( size_t i = 0; i < local.size(); i++ ) {
        MeshElementID id( true, AMP::Mesh::GeomType::Vertex, i, rank, d_meshID );
        DOFs->getDOFs( id, dofs );
        AMP_DEBUG_ASSERT( dofs.size() <= 3 );
        x->getValuesByGlobalID( dofs.size(), dofs.data(), offset );
        for ( size_t d = 0; d < 3; d++ )
            local[i][d] = d_vertex[i + start][d] + offset[d];
    }
    // Send the data to all ranks
    auto &disp = const_cast<std::vector<int> &>( d_vertex.offset() );
    std::vector<int> cnt( d_comm.getSize(), 0 );
    for ( size_t i = 0; i < cnt.size(); i++ )
        cnt[i] = disp[i + 1] - disp[i];
    d_comm.allGather( local.data(), local.size(), d_vertex.data(), cnt.data(), disp.data(), true );
    // Update the bounding box
    initializeBoundingBox();
    d_pos_hash++;
}


/****************************************************************
 *  Get the coordinated of the given vertex or the centroid      *
 ****************************************************************/
template<uint8_t NG>
std::array<double, 3> TriangleMesh<NG>::getPos( const ElementID &id ) const
{
    if ( id.type() == GeomType::Vertex ) {
        return d_vertex[id];
    } else {
        AMP_ERROR( "Not finished" );
    }
}


/****************************************************************
 * Return the IDs of the elements composing the current element  *
 ****************************************************************/
template<uint8_t NG>
void TriangleMesh<NG>::getVerticies( const ElementID &id, ElementID *IDs ) const
{
    auto type   = id.type();
    int N_nodes = 0;
    int nodes[NG + 1];
    if ( static_cast<uint8_t>( type ) > NG ) {
        return;
    } else if ( type == GeomType::Vertex ) {
        IDs[0] = id;
        return;
    } else if ( static_cast<uint8_t>( type ) == NG ) {
        N_nodes  = NG + 1;
        auto tri = d_globalTri[id];
        for ( int d = 0; d <= NG; d++ )
            nodes[d] = tri[d];
    } else if ( type == GeomType::Edge ) {
        N_nodes = 2;
        std::array<int, 2> edge;
        if ( id.is_local() )
            edge = d_childEdge[id.local_id()];
        else
            edge = d_remoteEdge.find( id )->second;
        nodes[0] = edge[0];
        nodes[1] = edge[1];
    } else if ( type == GeomType::Face ) {
        N_nodes = 3;
        std::array<int, 3> tri;
        if ( id.is_local() )
            tri = d_childFace[id.local_id()];
        else
            tri = d_remoteFace.find( id )->second;
        nodes[0] = tri[0];
        nodes[1] = tri[1];
        nodes[2] = tri[2];
    } else {
        AMP_ERROR( "Not finished" );
    }
    for ( int d = 0; d < N_nodes; d++ )
        IDs[d] = d_vertex.getID( nodes[d] );
}
template<uint8_t NG>
void TriangleMesh<NG>::getElementsIDs( const ElementID &id,
                                       const GeomType type,
                                       ElementID *IDs ) const
{
    if ( type == id.type() ) {
        IDs[0] = id;
        return;
    }
    if ( type == GeomType::Vertex ) {
        getVerticies( id, IDs );
        return;
    }
    if ( !id.is_local() )
        AMP_ERROR( "Getting children elements is not supported for ghost data" );
    if constexpr ( NG >= 2 ) {
        if ( id.type() == GeomType::Face && type == GeomType::Edge ) {
            auto children = getChildren<2, 1>( getElem<2>( id ) );
            for ( size_t i = 0; i < children.size(); i++ )
                IDs[i] = getID<1>( children[i] );
        }
    }
    if constexpr ( NG >= 3 ) {
        if ( id.type() == GeomType::Cell && type == GeomType::Edge ) {
            auto children = getChildren<3, 1>( getElem<3>( id ) );
            for ( size_t i = 0; i < children.size(); i++ )
                IDs[i] = getID<1>( children[i] );
        } else if ( id.type() == GeomType::Cell && type == GeomType::Face ) {
            auto children = getChildren<3, 2>( getElem<3>( id ) );
            for ( size_t i = 0; i < children.size(); i++ )
                IDs[i] = getID<2>( children[i] );
        }
    }
}


/********************************************************
 *  Convert between the ElementID and the triangle       *
 ********************************************************/
template<uint8_t NG>
template<uint8_t TYPE>
std::array<int, TYPE + 1> TriangleMesh<NG>::getElem( const ElementID &id ) const
{
    static_assert( TYPE <= NG );
    AMP_DEBUG_ASSERT( static_cast<uint8_t>( id.type() ) == TYPE );
    if constexpr ( TYPE == 0 ) {
        return { d_vertex.index( id ) };
    } else if constexpr ( TYPE == NG ) {
        return d_globalTri[id];
    } else if constexpr ( TYPE == 1 ) {
        int myRank = d_comm.getRank();
        int rank   = id.owner_rank();
        if ( rank == myRank )
            return d_childEdge[id.local_id()];
        AMP_ERROR( "Not finished" );
    } else if constexpr ( TYPE == 2 ) {
        int myRank = d_comm.getRank();
        int rank   = id.owner_rank();
        if ( rank == myRank )
            return d_childFace[id.local_id()];
        AMP_ERROR( "Not finished" );
    }
}
template<uint8_t NG>
template<uint8_t TYPE>
ElementID TriangleMesh<NG>::getID( const std::array<int, TYPE + 1> &tri ) const
{
    static_assert( TYPE <= NG );
    int myRank = d_comm.getRank();
    if constexpr ( TYPE == 0 ) {
        return d_vertex.getID( tri[0] );
    } else if constexpr ( TYPE == NG ) {
        int index = AMP::Utilities::findfirst( d_globalTri.size(), d_globalTri.data(), tri );
        AMP_DEBUG_ASSERT( d_globalTri[index] == tri );
        return d_globalTri.getID( index );
    } else if constexpr ( TYPE == 1 ) {
        int index  = AMP::Utilities::findfirst( d_childEdge, tri );
        bool found = index == (int) d_childEdge.size() ? false : d_childEdge[index] == tri;
        if ( found )
            return ElementID( true, GeomType::Edge, index, myRank );
        AMP_ERROR( "Not finished" );
    } else if constexpr ( TYPE == 2 ) {
        int index  = AMP::Utilities::findfirst( d_childFace, tri );
        bool found = index == (int) d_childFace.size() ? false : d_childFace[index] == tri;
        if ( found )
            return ElementID( true, GeomType::Face, index, myRank );
        AMP_ERROR( "Not finished" );
    }
}


/********************************************************
 *  Get the neighboring elements                         *
 ********************************************************/
template<uint8_t NG>
void TriangleMesh<NG>::getNeighborIDs( const ElementID &id, std::vector<ElementID> &IDs ) const
{
    IDs.clear();
    if ( !id.is_local() )
        AMP_ERROR( "Getting neighbors for non-owned elements is not supported" );
    // Check if we are dealing with the largest geometric type
    auto type = id.type();
    if ( static_cast<size_t>( type ) == NG ) {
        IDs.reserve( NG + 1 );
        int index = d_globalTri.index( id );
        for ( size_t i = 0; i <= NG; i++ ) {
            int neighbor = d_globalNab[index][i];
            if ( neighbor != -1 && neighbor != index )
                IDs.push_back( d_globalTri.getID( neighbor ) );
        }
        return;
    }
    // The neighbors are any elements that share a parent
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
template<uint8_t NG>
bool TriangleMesh<NG>::isOnSurface( const ElementID &id ) const
{
    auto type = id.type();
    if ( static_cast<uint8_t>( type ) == NG ) {
        // Triangle is on the surface if any neighbor is null
        int index = d_globalTri.index( id );
        bool test = false;
        for ( int tmp : d_globalNab[index] )
            test = test || tmp == -1;
        return test;
    } else if ( static_cast<uint8_t>( type ) == NG - 1 ) {
        // Face is on the surface if it has one parent
        auto parents = getElementParents( id, static_cast<GeomType>( NG ) );
        size_t N     = parents.second - parents.first;
        return N == 1;
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
template<uint8_t NG>
bool TriangleMesh<NG>::isOnBoundary( const ElementID &elemID, int id ) const
{
    int type     = static_cast<int>( elemID.type() );
    size_t index = d_boundary_it.size();
    for ( size_t i = 0; i < d_boundary_it.size(); i++ ) {
        if ( d_boundary_ids[i] == id )
            index = i;
    }
    if ( type > NG || index >= d_boundary_it.size() )
        return false;
    const auto &it = d_boundary_it[index].back()[type];
    return inIterator( elemID, &it );
}
template<uint8_t NG>
bool TriangleMesh<NG>::isInBlock( const ElementID &elemID, int id ) const
{
    if ( static_cast<uint8_t>( elemID.type() ) == NG ) {
        int index = d_globalTri.index( elemID );
        return id == d_blockID[index];
    } else {
        auto parents = getElementParents( elemID, static_cast<GeomType>( NG ) );
        for ( auto p = parents.first; p != parents.second; ++p ) {
            if ( isInBlock( *p, id ) )
                return true;
        }
    }
    return false;
}
template<uint8_t NG>
bool TriangleMesh<NG>::inIterator( const ElementID &id, const MeshIterator *it )
{
    if ( it->size() == 0 )
        return false;
    auto list = getList<NG>( *it );
#ifdef AMP_DEBUG
    auto &list2 = *list;
    for ( size_t i = 1; i < list->size(); i++ )
        AMP_ASSERT( list2[i] >= list2[i - 1] );
#endif
    size_t i = std::min<size_t>( AMP::Utilities::findfirst( *list, id ), list->size() - 1 );
    return list->operator[]( i ) == id;
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
template<uint8_t NG>
bool TriangleMesh<NG>::operator==( const Mesh &rhs ) const
{
    // Check if &rhs == this
    if ( this == &rhs )
        return true;
    // Check if we can cast to a TriangleMesh
    auto mesh = dynamic_cast<const TriangleMesh<NG> *>( &rhs );
    if ( !mesh )
        return false;
    // Perform comparison
    AMP_ERROR( "Not finished" );
    return false;
}


/****************************************************************
 * Write restart data                                            *
 ****************************************************************/
template<uint8_t NG>
void TriangleMesh<NG>::writeRestart( int64_t ) const
{
    AMP_ERROR( "writeRestart is not implimented for TriangleMesh" );
}


} // namespace AMP::Mesh


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Utilities.hpp"
#define TRI( NG ) std::array<int, NG + 1>
#define POS( NP ) std::array<double, NP>
#define INSTANTIATE_GENERATE( NG, NP )                                                 \
    template std::shared_ptr<AMP::Mesh::TriangleMesh<NG>>                              \
    AMP::Mesh::TriangleMesh<NG>::generate<NP>(                                         \
        const std::vector<std::array<POS( NP ), NG + 1>> &, const AMP_MPI &, double ); \
    template std::shared_ptr<AMP::Mesh::TriangleMesh<NG>>                              \
    AMP::Mesh::TriangleMesh<NG>::generate<NP>( const std::vector<POS( NP )> &,         \
                                               const std::vector<TRI( NG )> &,         \
                                               const std::vector<TRI( NG )> &,         \
                                               const AMP_MPI &,                        \
                                               std::shared_ptr<Geometry::Geometry>,    \
                                               std::vector<int> )
#define INSTANTIATE_TYPE( NG, TYPE )                                                              \
    template TRI( TYPE ) AMP::Mesh::TriangleMesh<NG>::getElem<TYPE>( const ElementID & ) const;   \
    template AMP::Mesh::ElementID AMP::Mesh::TriangleMesh<NG>::getID<TYPE>( const TRI( TYPE ) & ) \
        const
#define INSTANTIATE_FIND( NG )                                                            \
    template size_t AMP::Utilities::findfirst<TRI( NG )>( std::vector<TRI( NG )> const &, \
                                                          TRI( NG ) const & )
template class AMP::Mesh::StoreTriData<double, 3>;
template class AMP::Mesh::StoreTriData<int, 2>;
template class AMP::Mesh::StoreTriData<int, 3>;
template class AMP::Mesh::StoreTriData<int, 4>;
template class AMP::Mesh::TriangleMesh<1>;
template class AMP::Mesh::TriangleMesh<2>;
template class AMP::Mesh::TriangleMesh<3>;
INSTANTIATE_GENERATE( 1, 1 );
INSTANTIATE_GENERATE( 1, 2 );
INSTANTIATE_GENERATE( 1, 3 );
INSTANTIATE_GENERATE( 2, 2 );
INSTANTIATE_GENERATE( 2, 3 );
INSTANTIATE_GENERATE( 3, 3 );
INSTANTIATE_TYPE( 1, 0 );
INSTANTIATE_TYPE( 1, 1 );
INSTANTIATE_TYPE( 2, 0 );
INSTANTIATE_TYPE( 2, 1 );
INSTANTIATE_TYPE( 2, 2 );
INSTANTIATE_TYPE( 3, 0 );
INSTANTIATE_TYPE( 3, 1 );
INSTANTIATE_TYPE( 3, 2 );
INSTANTIATE_TYPE( 3, 3 );
INSTANTIATE_FIND( 0 );
INSTANTIATE_FIND( 1 );
INSTANTIATE_FIND( 2 );
INSTANTIATE_FIND( 3 );
template AMP::AMP_MPI::Request AMP::AMP_MPI::Isend<std::array<double, 1ul>>(
    std::array<double, 1ul> const *, int, int, int ) const;
template void AMP::AMP_MPI::recv<std::array<double, 1ul>>(
    std::array<double, 1ul> *, int &, int, bool, int ) const;
