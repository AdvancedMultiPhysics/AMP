#ifndef included_AMP_DelaunayFaceList_hpp
#define included_AMP_DelaunayFaceList_hpp

#include "AMP/utils/DelaunayFaceList.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/extended_int.h"

#include "ProfilerApp.h"

#include <stdint.h>
#include <stdlib.h>
#include <vector>


#define DEBUG_CHECK 0 // Flag to enable extra checks (1: some additional cost, 2: very expensive)

// Choose our numerical format and precision
// Note: for exact integer arithmetic, we need to support at least O(N^D)
//    where N is the index range and D is the dimension.
//    Note: I have not determined the small constant, but is is likely ~4
//    Note: Some routines use higher precisision internally (test_in_circumsphere)
typedef AMP::extended::int128_t int128_t;
typedef AMP::extended::int256_t int256_t;
typedef AMP::extended::int512_t int512_t;


namespace AMP::DelaunayTessellation {


/********************************************************************
 * Check if we conserve the neighbor triangles for two sets          *
 ********************************************************************/
static bool conserved_neighbors( const int N1, const int list1[], const int N2, const int list2[] )
{
    if ( N1 > 16 || N2 > 16 ) {
        printf( "Need to change internal data structures\n" );
        return false;
    }
    int tmp[16];
    for ( int i = 0; i < N1; i++ )
        tmp[i] = list1[i];
    for ( int j = 0; j < N2; j++ ) {
        if ( list2[j] < 0 )
            continue;
        bool found = false;
        for ( int i = 0; i < N1; i++ ) {
            if ( tmp[i] < 0 )
                continue;
            if ( tmp[i] == list2[j] ) {
                found  = true;
                tmp[i] = -1;
                break;
            }
        }
        if ( !found )
            return false;
    }
    for ( int i = 0; i < N1; i++ ) {
        if ( tmp[i] >= 0 )
            return false;
    }
    return true;
}


/********************************************************************
 * Check if two given triangles share a face                         *
 ********************************************************************/
bool are_tri_neighbors( const int ndim, const int tri1[], const int tri2[], int *f1, int *f2 )
{
    // Lets check each node in tri1 to see if it is in tri2
    *f1 = -1;
    for ( int i = 0; i <= ndim; i++ ) {
        int tmp    = tri1[i];
        bool found = false;
        for ( int j = 0; j <= ndim; j++ ) {
            if ( tmp == tri2[j] ) {
                found = true;
                break;
            }
        }
        if ( !found ) {
            if ( *f1 != -1 ) {
                // There are 2 or more nodes not in common
                return false;
            } else {
                *f1 = i;
            }
        }
    }
    if ( *f1 == -1 ) {
        // All nodes are in common
        return false;
    }
    // Now we need to find the missing node in tri2
    *f2 = -1;
    for ( int i = 0; i <= ndim; i++ ) {
        int tmp    = tri2[i];
        bool found = false;
        for ( int j = 0; j <= ndim; j++ ) {
            if ( tmp == tri1[j] ) {
                found = true;
                break;
            }
        }
        if ( !found ) {
            *f2 = i;
            break;
        }
    }
    return true;
}


/********************************************************************
 * This function performs a set of test taht checks that each        *
 *    triangle is valid and that the triangle neighbors are valid.   *
 * Note: This can be an expensive test and should only be used when  *
 *    debugging.                                                     *
 * Inputs:                                                           *
 *    N         The number of vertices                              *
 *    x         The coordinates of the vertices (NDIMxN)            *
 *    N_tri     The number of triangles                              *
 *    tri       The current list of triangles (NDIM+1xN_tri)         *
 *    tri_nab   The current list of triangle neighbors (NDIM+1xN_tri)*
 *    N_unused  The number of unused coordinates                     *
 ********************************************************************/
template<int NDIM>
bool check_current_triangles( int N,
                              const std::array<int, NDIM> x[],
                              size_t N_tri,
                              const std::array<int, NDIM + 1> tri[],
                              const std::array<int, NDIM + 1> tri_nab[],
                              const std::vector<size_t> &unused )
{
    if ( N_tri == 0 )
        return false;
    PROFILE( "check_current_triangles", 4 );
    bool pass = true;
    for ( size_t i = 0; i < N_tri; i++ ) {
        // First check if we are dealing with an unused triangle
        bool used = true;
        for ( auto &elem : unused ) {
            if ( i == elem )
                used = false;
        }
        if ( !used )
            break;
        // Check if the triangle is valid
        std::array<int, NDIM> x2[NDIM + 1];
        for ( int d = 0; d <= NDIM; d++ ) {
            int k = tri[i][d];
            if ( k < 0 || k >= N ) {
                printf( "Error with internal structures (invalid triangle index)\n" );
                pass = false;
                break;
            }
            x2[d] = x[k];
        }
        if ( !pass ) {
            break;
        }
        double vol = fabs( DelaunayHelpers::calcVolume<NDIM, int>( x2 ) );
        if ( vol < 0.0 ) {
            // The volume of each triangle must be strickly > 0
            printf( "Invalid triangle volume detected\n" );
            pass = false;
        }
        if ( !pass ) {
            break;
        }
        // Check if the triangle neighbors are valid
        for ( int d = 0; d <= NDIM; d++ ) {
            int k = tri_nab[i][d];
            if ( k == -1 )
                continue;
            if ( k < 0 || k >= (int) N_tri )
                pass = false;
            for ( auto &elem : unused ) {
                if ( k == static_cast<int>( elem ) )
                    pass = false;
            }
            for ( int d2 = 0; d2 <= NDIM; d2++ ) {
                if ( tri_nab[k][d2] == k )
                    pass = false;
            }
            bool shared = false;
            for ( int d2 = 0; d2 <= NDIM; d2++ ) {
                if ( tri_nab[k][d2] == (int) i )
                    shared = true;
            }
            if ( !shared )
                pass = false;
        }
        if ( !pass ) {
            printf( "Error with internal structures (invalid triangle neighbor)\n" );
            break;
        }
    }
    return pass;
}


/********************************************************************
 * FaceList constructor                                              *
 ********************************************************************/
static inline uint32_t get_hash_key( uint32_t key )
{
    key *= 0x9E3779B9; // 2^32*0.5*(sqrt(5)-1)
    key >>= 22;        // mod(key,1024)
    return key;
}
template<int NDIM>
DelaunayTessellation::FaceList<NDIM>::FaceList( const int Nx_in,
                                                const Point *x_in,
                                                const int tri0_id,
                                                const std::array<int, NDIM + 1> &tri0 )
    : Nx( Nx_in ), x0( x_in )
{
    for ( auto &elem : hash_table )
        elem = -1; // Initialize the hash table to -1
    data.resize( NDIM + 1 );
    for ( int i = 0; i <= NDIM; i++ ) {
        auto key        = get_hash_key( get_face_index( i, 0 ) );
        int k           = hash_table[key];
        hash_table[key] = i;
        data[i].prev    = -static_cast<int>( key + 1 );
        data[i].next    = k;
        if ( k != -1 )
            data[k].prev = i;
        data[i].tri_id  = tri0_id;
        data[i].face_id = i;
        int j1          = 0;
        for ( int j2 = 0; j2 <= NDIM; j2++ ) {
            if ( j2 == i )
                continue;
            int m             = tri0[j2];
            data[i].index[j1] = m;
            data[i].x[j1]     = x0[m];
            j1++;
        }
    }
    // Initialize the "center" of the volume
    for ( int d = 0; d < NDIM; d++ )
        xc[d] = 0.0;
    for ( size_t i = 0; i < data.size(); i++ ) {
        for ( int j = 0; j < NDIM; j++ ) {
            for ( int d = 0; d < NDIM; d++ )
                xc[d] += data[i].x[j][d];
        }
    }
    for ( int d = 0; d < NDIM; d++ )
        xc[d] /= NDIM * data.size();
    check_data();
}


/********************************************************************
 * Delete a face                                                     *
 ********************************************************************/
template<int NDIM>
void DelaunayTessellation::FaceList<NDIM>::delete_faces( std::vector<int> &ids )
{
    // Sort the ids to delete
    AMP::Utilities::quicksort( ids );
    // We delete in reverse order so that we can safely swap entries
    for ( int i = (int) ids.size() - 1; i >= 0; i-- ) {
        // Delete the entry and remove it from the hash table
        int k = ids[i];
        if ( data[k].prev >= 0 ) {
            data[data[k].prev].next = data[k].next;
            if ( data[k].next != -1 )
                data[data[k].next].prev = data[k].prev;
        } else {
            auto key        = -data[k].prev - 1;
            int j           = data[k].next;
            hash_table[key] = j;
            if ( j != -1 )
                data[j].prev = -static_cast<int>( key + 1 );
        }
        if ( k + 1 < (int) data.size() ) {
            // Move the last entry to the current entry and update the hash table
            std::swap( data[k], data.back() );
            if ( data[k].prev >= 0 ) {
                data[data[k].prev].next = k;
            } else {
                auto key        = -data[k].prev - 1;
                hash_table[key] = k;
            }
            if ( data[k].next != -1 )
                data[data[k].next].prev = k;
        }
        data.pop_back();
    }
}


/********************************************************************
 * This function adds a node to the convex hull                      *
 * Note: currently this algorithum scales as O(N)                    *
 ********************************************************************/
template<int NDIM>
void DelaunayTessellation::FaceList<NDIM>::add_node(
    const int node_id,
    std::vector<size_t> &unused,
    size_t &N_tri,
    std::vector<uint32_t> &new_tri_id,
    std::vector<std::array<int, NDIM + 1>> &new_tri,
    std::vector<std::array<int, NDIM + 1>> &new_tri_nab,
    std::vector<int> &neighbor,
    std::vector<int> &face_id )
{
    PROFILE( "FaceList::add_node", 4 );
#if DEBUG_CHECK == 2
    check_data();
#endif
    // Create the new triangles
    new_tri_id.clear();
    new_tri.clear();
    new_tri_nab.clear();
    neighbor.clear();
    face_id.clear();
    std::vector<int> ids;
    Point x2[NDIM + 1];
    for ( size_t i = 0; i < data.size(); i++ ) {
        // Get the triangle number and face id
        int tri_num  = data[i].tri_id;
        int face_num = data[i].face_id;
        // We will need to for a new triangle with each face
        bool pass = outside_triangle( data[i].x, x0[node_id] );
        if ( !pass )
            continue;
        for ( int j = 0; j < NDIM; j++ )
            x2[j] = data[i].x[j];
        x2[NDIM]      = x0[node_id];
        double volume = fabs( DelaunayHelpers::calcVolume<NDIM, int>( x2 ) );
        if ( volume <= 0 )
            continue;
        // Create a new triangle consisting of the current point and the current face
        Triangle tri, nab;
        for ( int j = 0; j < NDIM; j++ )
            tri[j] = data[i].index[j];
        tri[NDIM] = node_id;
        for ( int j = 0; j < NDIM; j++ )
            nab[j] = -1;
        nab[NDIM] = tri_num;
        int id;
        if ( !unused.empty() ) {
            id = unused.back();
            unused.pop_back();
        } else {
            id = N_tri;
            N_tri++;
        }
        new_tri_id.push_back( id );
        new_tri.push_back( tri );
        new_tri_nab.push_back( nab );
        neighbor.push_back( tri_num );
        face_id.push_back( face_num );
        ids.push_back( i );
    }
    if ( ids.empty() ) {
        // No triangles were created, the point must be inside (or on) the convex hull
        AMP_ERROR( "Error: point is inside or on the convex hull" );
    }
    // Check that the new triangles are valid and have the proper point ordering
    for ( size_t i = 0; i < ids.size(); i++ ) {
        for ( int j1 = 0; j1 <= NDIM; j1++ ) {
            int k  = new_tri[i][j1];
            x2[j1] = x0[k];
        }
        double volume = DelaunayHelpers::calcVolume<NDIM, int>( x2 );
        if ( fabs( volume ) <= 0 ) {
            // The triangle is invalid
            AMP_ERROR( "Invalid triangle created" );
        } else if ( volume < 0 ) {
            // The volume is negitive, swap the order of the last two points
            std::swap( new_tri[i][NDIM - 1], new_tri[i][NDIM] );
            std::swap( new_tri_nab[i][NDIM - 1], new_tri_nab[i][NDIM] );
        }
        for ( int j1 = 0; j1 <= NDIM; j1++ ) {
            int k  = new_tri[i][j1];
            x2[j1] = x0[k];
        }
        volume = DelaunayHelpers::calcVolume<NDIM, int>( x2 );
        if ( volume <= 0.0 )
            printf( "Warning: volume is still negitive\n" );
    }
    // Check if the current triangle shares a face with one of the other new triangles
    for ( size_t i1 = 0; i1 < ids.size(); i1++ ) {
        for ( size_t i2 = i1 + 1; i2 < ids.size(); i2++ ) {
            int f1, f2;
            bool is_neighbor =
                are_tri_neighbors( NDIM, new_tri[i1].data(), new_tri[i2].data(), &f1, &f2 );
            if ( is_neighbor ) {
#if DEBUG_CHECK > 0
                AMP_ASSERT( new_tri_nab[i1][f1] == -1 );
                AMP_ASSERT( new_tri_nab[i2][f2] == -1 );
#endif
                new_tri_nab[i1][f1] = new_tri_id[i2];
                new_tri_nab[i2][f2] = new_tri_id[i1];
            }
        }
    }
    // Remove the old faces that were used to create the triangles
    delete_faces( ids );
    // Add the new faces for each triangle
    int N_add = 0;
    for ( size_t i = 0; i < new_tri_nab.size(); i++ ) {
        for ( int d = 0; d <= NDIM; d++ ) {
            if ( new_tri_nab[i][d] == -1 )
                N_add++;
        }
    }
    if constexpr ( NDIM == 2 ) {
        AMP_ASSERT( N_add == 2 );
    } else if constexpr ( NDIM == 3 ) {
        AMP_ASSERT( N_add >= 3 );
        AMP_ASSERT( N_add <= (int) ( 2 * ids.size() + 1 ) );
    }
    for ( size_t i = 0; i < new_tri_id.size(); i++ ) {
        for ( int j = 0; j <= NDIM; j++ ) {
            if ( new_tri_nab[i][j] == -1 ) {
                // Add the face
                face_data_struct tmp;
                tmp.tri_id  = new_tri_id[i];
                tmp.face_id = j;
                int j1      = 0;
                for ( int j2 = 0; j2 <= NDIM; j2++ ) {
                    if ( j2 == j )
                        continue;
                    int k         = new_tri[i][j2];
                    tmp.index[j1] = k;
                    tmp.x[j1]     = x0[k];
                    j1++;
                }
                // Update the hash table
                auto key        = get_hash_key( get_face_index( j, new_tri_id[i] ) );
                int k           = hash_table[key];
                hash_table[key] = data.size();
                tmp.prev        = -static_cast<int>( key + 1 );
                tmp.next        = k;
                if ( k != -1 )
                    data[k].prev = data.size();
                data.push_back( tmp );
            }
        }
    }
#if DEBUG_CHECK == 2
    check_data();
#endif
// Check the triangle neighbors
#if DEBUG_CHECK == 2
    auto tri_nab_tmp = new_tri_nab;
    for ( size_t i = 0; i < new_tri_nab.size(); i++ ) {
        for ( int d = 0; d <= NDIM; d++ ) {
            int k = -1;
            for ( size_t j = 0; j < new_tri_id.size(); j++ )
                if ( tri_nab_tmp[i][d] == (int) new_tri_id[j] )
                    k = j;
            tri_nab_tmp[i][d] = k;
        }
    }
    bool all_valid = check_current_triangles<NDIM>(
        Nx, x0, new_tri.size(), new_tri, tri_nab_tmp, std::vector<size_t>(), 0 );
    if ( !all_valid )
        printf( "Warning: new triangle neighbors are inconsistent\n" );
#endif
}


/********************************************************************
 * Function to check the face data                                   *
 ********************************************************************/
template<int NDIM>
static inline bool compare_index( const int *i1, const int *i2 );
template<>
inline bool compare_index<2>( const int *i1, const int *i2 )
{
    return ( i1[0] == i2[0] || i1[0] == i2[1] ) && ( i1[1] == i2[0] || i1[1] == i2[1] );
}
template<>
inline bool compare_index<3>( const int *i1, const int *i2 )
{
    return ( i1[0] == i2[0] || i1[0] == i2[1] || i1[0] == i2[2] ) &&
           ( i1[1] == i2[0] || i1[1] == i2[1] || i1[1] == i2[2] ) &&
           ( i1[2] == i2[0] || i1[2] == i2[1] || i1[2] == i2[2] );
}
template<int NDIM>
void DelaunayTessellation::FaceList<NDIM>::check_data()
{
    for ( size_t i = 0; i < data.size(); i++ ) {
        int Nf = 0;
        for ( size_t j = 0; j < data.size(); j++ ) {
            bool c1 = data[i].tri_id == data[j].tri_id && data[i].face_id == data[j].face_id;
            bool c2 = compare_index<NDIM>( data[i].index, data[j].index );
            if ( c1 && c2 )
                Nf++;
            else if ( c1 || c2 )
                AMP_ERROR( "Internal error" );
        }
        AMP_ASSERT( Nf == 1 );
    }
    if constexpr ( NDIM == 2 ) {
        // No node should be on the face list more than twice
        std::vector<uint8_t> tmp( Nx );
        for ( size_t i = 0; i < data.size(); i++ ) {
            tmp[data[i].index[0]]++;
            tmp[data[i].index[1]]++;
        }
        bool pass = true;
        for ( int i = 0; i < Nx; i++ )
            pass = pass && tmp[i] <= 2;
        if ( !pass )
            AMP_ERROR( "Internal error" );
    }
}


/********************************************************************
 * This function updates the faces on the convex hull                *
 * Note:  Special attention must be paid to triangle indicies that   *
 * are in both the old and new lists.                                *
 ********************************************************************/
template<int NDIM>
void DelaunayTessellation::FaceList<NDIM>::update_face( const int N,
                                                        const int old_tid[],
                                                        const int old_fid[],
                                                        const int new_tid[],
                                                        const int new_fid[],
                                                        const Triangle *tri_in )
{
    PROFILE( "update_face", 4 );
    // Check the inputs
    bool valid_inputs = N <= 16;
    for ( int i = 0; i < N; i++ ) {
        for ( int k = 0; k <= NDIM; k++ ) {
            if ( tri_in[new_tid[i]][k] == -1 )
                valid_inputs = false;
        }
    }
    if ( !valid_inputs )
        throw std::logic_error( "Internal error: DelaunayTessellation::FaceList::update_face" );
    // First we want to check for entries that appear in both the old and new lists
    size_t index_old[16];    // List of the unique indicies of the faces in the old list
    size_t index_new[16];    // List of the unique indicies of the faces in the new list
    size_t index_update[16]; // List of the unique indicies of the faces in both lists
    for ( int i = 0; i < N; i++ ) {
        index_old[i] = get_face_index( old_fid[i], old_tid[i] );
        index_new[i] = get_face_index( new_fid[i], new_tid[i] );
    }
    int N2       = N;
    int N_update = 0;
    for ( int i = N2 - 1; i >= 0; i-- ) {
        for ( int j = 0; j < N2; j++ ) {
            if ( index_old[i] == index_new[j] ) {
                // This face occurs in both lists, remove it
                index_update[N_update] = index_old[i];
                index_old[i]           = index_old[N2 - 1];
                index_new[j]           = index_new[N2 - 1];
                N_update++;
                N2--;
                break;
            }
        }
    }
    // Update the faces whose triangle number and face number did not change (the vertices might
    // have changed)
    for ( int i = 0; i < N_update; i++ ) {
        // Find the face
        auto key     = get_hash_key( index_update[i] );
        auto tri_id  = static_cast<int>( index_update[i] / ( NDIM + 1 ) );
        auto face_id = static_cast<int>( index_update[i] % ( NDIM + 1 ) );
        int j        = hash_table[key];
        while ( j != -1 ) {
            if ( data[j].tri_id == tri_id && data[j].face_id == face_id )
                break;
            j = data[j].next;
        }
        AMP_ASSERT( j != -1 ); // Check that the face was found
        // Update the current face (note: we do not need to update the hash table data, since the
        // key did not change)
        int j1 = 0;
        for ( int j2 = 0; j2 <= NDIM; j2++ ) {
            if ( j2 == face_id )
                continue;
            int k             = tri_in[tri_id][j2];
            data[j].index[j1] = k;
            data[j].x[j1]     = x0[k];
            j1++;
        }
    }
    // Replace each face that needs updating
    for ( int i = 0; i < N2; i++ ) {
        // Find the old face
        auto key      = get_hash_key( index_old[i] );
        auto old_tri  = static_cast<int>( index_old[i] / ( NDIM + 1 ) );
        auto old_face = static_cast<int>( index_old[i] % ( NDIM + 1 ) );
        int j         = hash_table[key];
        while ( j != -1 ) {
            if ( data[j].tri_id == old_tri && data[j].face_id == old_face )
                break;
            j = data[j].next;
        }
        AMP_ASSERT( j != -1 ); // Check that the face was found
        // Replace the current face with the new face
        auto new_tri    = static_cast<int>( index_new[i] / ( NDIM + 1 ) );
        auto new_face   = static_cast<int>( index_new[i] % ( NDIM + 1 ) );
        data[j].tri_id  = new_tri;
        data[j].face_id = new_face;
        int j1          = 0;
        for ( int j2 = 0; j2 <= NDIM; j2++ ) {
            if ( j2 == new_face )
                continue;
            int k             = tri_in[new_tri][j2];
            data[j].index[j1] = k;
            data[j].x[j1]     = x0[k];
            j1++;
        }
        // Update the hash table
        if ( data[j].prev >= 0 ) {
            int k = data[j].prev;
            AMP_ASSERT( k >= 0 );
            data[k].next = data[j].next;
            if ( data[j].next != -1 )
                data[data[j].next].prev = k;
        } else {
            int k           = data[j].next;
            hash_table[key] = k;
            if ( k != -1 )
                data[k].prev = -static_cast<int>( key + 1 );
        }
        key             = get_hash_key( index_new[i] );
        int k           = hash_table[key];
        hash_table[key] = j;
        data[j].prev    = -static_cast<int>( key + 1 );
        data[j].next    = k;
        if ( k != -1 )
            data[k].prev = j;
    }
#if DEBUG_CHECK == 2
    check_data();
#endif
}


/********************************************************************
 * This function calculates the normal for the face of a simplex     *
 * Note: the normal is not normalized and may not be in the outward  *
 *       direction, but may be computed using exact math             *
 *                                                                   *
 * Compute the normal using the n-dimensional cross product:         *
 *    |     e1          e2      ...      en     |                    *
 *    | (A1-A2).e1  (A1-A2).e2  ...  (A1-A2).en |                    *
 *    | (A1-A3).e1  (A1-A3).e2  ...  (A1-A3).en |                    *
 *    |     ...         ...     ...      ...    |                    *
 *    | (A1-An).e1  (A1-An).e2  ...  (A1-An).en |                    *
 *                                                                   *
 ********************************************************************/
template<int NDIM>
std::array<int64_t, NDIM>
DelaunayTessellation::FaceList<NDIM>::calc_surface_normal( const std::array<int, NDIM> x[] )
{
    std::array<int64_t, NDIM> norm;
    if constexpr ( NDIM == 1 ) {
        norm[0] = int64_t( 1 );
    } else if constexpr ( NDIM == 2 ) {
        norm[0] = x[0][1] - x[1][1];
        norm[1] = x[1][0] - x[0][0];
    } else if constexpr ( NDIM == 3 ) {
        norm[0] = int64_t( x[0][1] - x[1][1] ) * int64_t( x[0][2] - x[2][2] ) -
                  int64_t( x[0][2] - x[1][2] ) * int64_t( x[0][1] - x[2][1] );
        norm[1] = int64_t( x[0][2] - x[1][2] ) * int64_t( x[0][0] - x[2][0] ) -
                  int64_t( x[0][0] - x[1][0] ) * int64_t( x[0][2] - x[2][2] );
        norm[2] = int64_t( x[0][0] - x[1][0] ) * int64_t( x[0][1] - x[2][1] ) -
                  int64_t( x[0][1] - x[1][1] ) * int64_t( x[0][0] - x[2][0] );
    } else {
        AMP_ERROR( "Invalid dimension" );
    }
    return norm;
}


/********************************************************************
 * This function determines if a point is outside a triangle on the  *
 *    convex hull.  This can be done by computing the distance       *
 *    between a hyperplane and a point.                              *
 * This distance between a hyperplane and a point can be determined  *
 *   from  D_i=dot(n,x_0-x_i), while the sign can be determined by   *
 *   comparing to a known point inside the convex hull (xc).         *
 * Note: this uses a mixture of exact and inexact math, but should   *
 *   be accurate.  The exact math portion requires N^D precision.    *
 ********************************************************************/
template<int NDIM>
bool DelaunayTessellation::FaceList<NDIM>::outside_triangle( const std::array<int, NDIM> x[NDIM],
                                                             const std::array<int, NDIM> &xi ) const
{
    // First compute the normal
    // Note: the normal is not normalized and may point inward
    auto norm = calc_surface_normal( x );
    // Compute the distance from the surface
    using ETYPE = typename AMP::DelaunayHelpers::getETYPE<NDIM, int>::ETYPE;
    ETYPE dist2( 0 );
    double dot = 0.0;
    double tmp = 0.0;
    for ( int i = 0; i < NDIM; i++ ) {
        double norm2 = static_cast<double>( norm[i] );
        dot += norm2 * ( x[0][i] - xc[i] );
        dist2 += ETYPE( norm[i] ) * ETYPE( xi[i] - x[0][i] );
        tmp += norm2 * norm2;
    }
    double sign = ( dot < 0 ) ? -1.0 : 1.0;
    double dist = sign * static_cast<double>( dist2 ) / std::sqrt( tmp );
    return dist > 0;
}


} // namespace AMP::DelaunayTessellation

#endif
