#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/DelaunayInterpolation.h"
#include "AMP/utils/DelaunayTessellation.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include "LapackWrappers.h"

#define NDIM_MAX 3
#define PROFILE_LEVEL 3


static void inv_M( const int n, const double *M, double *M_inv );
static void solve_system( const int N, const double *M, const double *b, double *x );
static void Gauss_Seidel( const unsigned int Nb,
                          const unsigned int N,
                          const unsigned int M,
                          const double D[],
                          const unsigned int N_row[],
                          unsigned int *icol[],
                          const double A[],
                          const double rhs[],
                          double *x,
                          const int N_it );
static double interp_cubic_recursive( const int ndim,
                                      const int N,
                                      const double *x,
                                      const double *f,
                                      const double *g,
                                      const double *xi,
                                      const double *L,
                                      double *gi );
static double interp_line( const int n,
                           const double *x0,
                           const double f0,
                           const double *g0,
                           const double *x1,
                           const double f1,
                           const double *g1,
                           const double *x,
                           double *g );
static void get_interpolant_points( const int ndim,
                                    const int N,
                                    const double *x,
                                    const double *L,
                                    const double *xi,
                                    double *xi1,
                                    double *xi2,
                                    bool check );
static void compute_gradient( const int ndim, const double *x, const double *f, double *g );

template<class type_a>
static void quicksort1( int n, type_a *arr );
template<class type_a, class type_b>
static void quicksort2( int n, type_a *arr, type_b *brr );
static int intersect_sorted( const int N_lists,
                             const int size[],
                             unsigned int *list[],
                             const int N_max,
                             unsigned int *intersection );


static inline size_t log2ceil( size_t x )
{
    size_t ans = 1;
    while ( x >>= 1 )
        ans++;
    return ans;
}


static constexpr double TRI_TOL = 1e-10;


/********************************************************************
 * Primary constructor                                               *
 ********************************************************************/
template<class TYPE>
DelaunayInterpolation<TYPE>::DelaunayInterpolation( const int ndim )
{
    AMP_ASSERT( ndim >= 1 );
    AMP_ASSERT( ndim <= NDIM_MAX );
    // Initialize the class data
    d_ndim       = ndim;
    d_level      = 4;
    d_N          = 0;
    d_N_tri      = 0;
    d_tri        = nullptr;
    d_x          = nullptr;
    d_N_node_sum = 0;
    d_N_node     = nullptr;
    d_node_list  = nullptr;
    d_tri_nab    = nullptr;
    d_node_tri   = nullptr;
    d_tree       = nullptr;
}


/********************************************************************
 * De-constructor                                                    *
 ********************************************************************/
template<class TYPE>
DelaunayInterpolation<TYPE>::~DelaunayInterpolation()
{
    // Delete all data
    d_ndim = 0;
    d_N    = 0;
    if ( d_x != nullptr ) {
        if ( d_level == 2 || d_level == 4 )
            delete[] d_x;
        d_x = nullptr;
    }
    d_N_tri = 0;
    delete[] d_tri;
    d_tri = nullptr;
    delete[] d_N_node;
    d_N_node = nullptr;
    if ( d_node_list != nullptr ) {
        delete[] d_node_list[0];
    }
    delete[] d_node_list;
    d_node_list = nullptr;
    delete[] d_tri_nab;
    d_tri_nab = nullptr;
    delete[] d_node_tri;
    d_node_tri = nullptr;
    delete d_tree;
    d_tree = nullptr;
}


/********************************************************************
 * Function to set the storage level                                 *
 ********************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::set_storage_level( const int level_new )
{
    PROFILE_START( "set_storage_level", PROFILE_LEVEL );
    bool copy_x           = false;
    bool delete_x         = false;
    bool delete_neighbors = false;
    switch ( level_new ) {
    case 1:
        // Store only the tesselation
        delete_x         = true;
        delete_neighbors = true;
        break;
    case 2:
        // Store the tesselation and the coordinates
        if ( d_level == 1 || d_level == 3 )
            copy_x = true;
        delete_neighbors = true;
        break;
    case 3:
        // Store the tesselation and the node and triangle neighbors
        delete_x = true;
        break;
    case 4:
        // Store all data
        if ( d_level == 1 || d_level == 3 )
            copy_x = true;
        break;
    default:
        // Unknown storage
        throw std::logic_error( "Unknown storage" );
    }
    if ( d_level == 1 || d_level == 3 )
        delete_x = false;
    if ( d_x != nullptr && delete_x ) {
        delete[] d_x;
        d_x = nullptr;
    }
    if ( d_x != nullptr && copy_x ) {
        TYPE *tmp = d_x;
        d_x       = new TYPE[d_ndim * d_N];
        for ( size_t i = 0; i < d_ndim * d_N; i++ )
            d_x[i] = tmp[i];
    }
    if ( d_N_node != nullptr && delete_neighbors ) {
        delete[] d_N_node;
        delete[] d_node_list[0];
        delete[] d_node_list;
        d_N_node    = nullptr;
        d_node_list = nullptr;
    }
    if ( d_tri_nab != nullptr && delete_neighbors ) {
        delete[] d_tri_nab;
        d_tri_nab = nullptr;
    }
    if ( d_node_tri != nullptr && delete_neighbors ) {
        delete[] d_node_tri;
        d_node_tri = nullptr;
    }
    if ( d_tree != nullptr && delete_neighbors ) {
        delete[] d_tri_nab;
        d_tree = nullptr;
    }
    d_level = level_new;
    PROFILE_STOP( "set_storage_level", PROFILE_LEVEL );
}


/********************************************************************
 * Function to return the triangles                                  *
 ********************************************************************/
template<class TYPE>
size_t DelaunayInterpolation<TYPE>::get_N_tri() const
{
    return static_cast<size_t>( d_N_tri );
}
template<class TYPE>
int *DelaunayInterpolation<TYPE>::get_tri( int base ) const
{
    PROFILE_START( "get_tri", PROFILE_LEVEL );
    auto tri = new int[d_N_tri * ( d_ndim + 1 )];
    for ( size_t i = 0; i < ( d_ndim + 1 ) * d_N_tri; i++ )
        tri[i] = d_tri[i] + base;
    PROFILE_STOP( "get_tri", PROFILE_LEVEL );
    return tri;
}
template<class TYPE>
int *DelaunayInterpolation<TYPE>::get_tri_nab( int base ) const
{
    PROFILE_START( "get_tri_nab", PROFILE_LEVEL );
    create_tri_neighbors();
    auto tri_nab = new int[d_N_tri * ( d_ndim + 1 )];
    for ( size_t i = 0; i < ( d_ndim + 1 ) * d_N_tri; i++ )
        tri_nab[i] = static_cast<int>( d_tri_nab[i] ) + base;
    memory_usage( d_level );
    PROFILE_STOP( "get_tri_nab", PROFILE_LEVEL );
    return tri_nab;
}
template<class TYPE>
void DelaunayInterpolation<TYPE>::copy_tessellation(
    int *N_out, TYPE *x_out, const int base, int *N_tri_out, int *tri_out ) const
{
    PROFILE_START( "copy_tessellation", PROFILE_LEVEL );
    *N_out = d_N;
    if ( x_out != nullptr ) {
        if ( d_x == nullptr ) {
            for ( size_t i = 0; i < d_ndim * d_N; i++ )
                x_out[i] = 0;
        } else {
            for ( size_t i = 0; i < d_ndim * d_N; i++ )
                x_out[i] = d_x[i];
        }
    }
    *N_tri_out = d_N_tri;
    if ( tri_out != nullptr ) {
        if ( d_tri == nullptr ) {
            for ( size_t i = 0; i < ( d_ndim + 1 ) * d_N_tri; i++ )
                tri_out[i] = 0;
        } else {
            for ( size_t i = 0; i < ( d_ndim + 1 ) * d_N_tri; i++ )
                tri_out[i] = d_tri[i] + base;
        }
    }
    PROFILE_STOP( "copy_tessellation", PROFILE_LEVEL );
}


/********************************************************************
 * Function to return the memory usage                               *
 ********************************************************************/
template<class TYPE>
size_t DelaunayInterpolation<TYPE>::memory_usage( const int mem_level ) const
{
    PROFILE_START( "memory_usage", PROFILE_LEVEL );
    // int level2 = mem_level;
    size_t bytes = sizeof( DelaunayInterpolation );
    if ( mem_level == 0 ) {
        // Return the current memory usage
        bytes += ( d_ndim + 1 ) * d_N_tri * sizeof( unsigned int );
        if ( d_x != nullptr && ( d_level == 2 || d_level == 4 ) )
            bytes += d_ndim * d_N * sizeof( TYPE );
        if ( d_N_node != nullptr ) {
            bytes += d_N * sizeof( int );
            bytes += d_N * sizeof( unsigned int * );
            bytes += d_N_node_sum * sizeof( unsigned int );
        }
        if ( d_tri_nab != nullptr )
            bytes += ( d_ndim + 1 ) * d_N_tri * sizeof( int );
        if ( d_tree != nullptr )
            bytes += d_tree->memory_usage();
    } else {
        bytes += ( d_ndim + 1 ) * d_N_tri *
                 sizeof( unsigned int ); // The storage required for the triangles
        if ( mem_level == 2 || mem_level == 4 ) {
            // Store the coordinates
            bytes += d_ndim * d_N * sizeof( TYPE );
        }
        if ( mem_level == 3 || mem_level == 4 ) {
            // Store the node and triangle neighbors
            bytes += d_N * sizeof( int );            // d_N_node
            bytes += d_N * sizeof( unsigned int * ); // d_node_list
            if ( d_N_node_sum > 0 ) {
                // We know how much storage we will need for the node lists
                bytes += d_N_node_sum * sizeof( unsigned int );
            } else {
                // Estimate the memory needed for the node lists
                bytes += 2 * d_ndim * d_N_tri * sizeof( unsigned int );
            }
            bytes += d_N_tri * ( d_ndim ) * sizeof( int ); // d_tri_nab
            bytes += d_N * sizeof( int );                  // d_node_tri
            if ( d_tree != nullptr ) {
                // We know the memory used for the kdtree
                bytes += d_tree->memory_usage();
            } else {
                // Estimate the memory needed for the kdtree
                bytes += d_ndim * d_N * sizeof( TYPE );  // The tree will store the coordinates
                bytes += d_N * sizeof( int );            // The tree will store the indicies
                size_t N_leaves = 2 * log2ceil( d_N );   // An estimate of the number of leaves
                bytes += N_leaves * ( 48 + 8 * d_ndim ); // An estimate of the memory for the leaves
            }
        }
    }
    PROFILE_STOP( "memory_usage", PROFILE_LEVEL );
    return bytes;
}


/********************************************************************
 * Function to construct the tessellation                            *
 ********************************************************************/
template<class TYPE>
void write_failed_points( int ndim, int N, const TYPE *data, FILE *fid );
template<>
void write_failed_points<double>( int ndim, int N, const double *data, FILE *fid )
{
    fprintf( fid, "%i points in %iD in double precision\n", N, ndim );
    fwrite( data, sizeof( double ), N * ndim, fid );
}
template<>
void write_failed_points<int>( int ndim, int N, const int *data, FILE *fid )
{
    fprintf( fid, "%i points in %iD in int precision\n", N, ndim );
    fwrite( data, sizeof( int ), N * ndim, fid );
}
template<class TYPE>
int DelaunayInterpolation<TYPE>::create_tessellation( const int N,
                                                      const TYPE *x,
                                                      const TYPE *y,
                                                      const TYPE *z )
{
    if ( d_level == 1 )
        d_level = 2;
    if ( d_level == 3 )
        d_level = 4;
    const TYPE *xyz[3] = { x, y, z };
    auto tmp           = new TYPE[N * d_ndim];
    for ( int i = 0; i < N; i++ ) {
        for ( int d = 0; d < d_ndim; d++ )
            tmp[d + i * d_ndim] = xyz[d][i];
    }
    int error = create_tessellation( N, tmp );
    delete[] tmp;
    return error;
}
template<class TYPE>
int DelaunayInterpolation<TYPE>::create_tessellation( const int N_in, const TYPE x_in[] )
{
    PROFILE_START( "create_tessellation", PROFILE_LEVEL );
    AMP_ASSERT( d_ndim > 0 );
    AMP_ASSERT( N_in > d_ndim );
    AMP_ASSERT( x_in != nullptr );
    d_N        = N_in;
    int N_tri2 = -1;
    if ( d_ndim == 1 ) {
        // The triangles are just the sorted points (note the duplicate indicies)
        auto x_tmp = new TYPE[d_N];
        auto i_tmp = new int[d_N];
        for ( size_t i = 0; i < d_N; i++ )
            x_tmp[i] = x_in[i];
        for ( size_t i = 0; i < d_N; i++ )
            i_tmp[i] = (int) i;
        quicksort2( N_in, x_tmp, i_tmp );
        N_tri2 = N_in - 1;
        d_tri  = new int[2 * N_tri2];
        for ( int i = 0; i < N_tri2; i++ ) {
            d_tri[2 * i + 0] = i_tmp[i + 0];
            d_tri[2 * i + 1] = i_tmp[i + 1];
        }
        delete[] x_tmp;
        delete[] i_tmp;
        if ( d_level == 3 || d_level == 4 ) {
            d_N_tri = N_tri2;
            create_tri_neighbors();
        }
    } else if ( d_ndim == 2 || d_ndim == 3 ) {
        if ( d_level == 3 || d_level == 4 )
            N_tri2 =
                DelaunayTessellation::create_tessellation( d_ndim, N_in, x_in, &d_tri, &d_tri_nab );
        else
            N_tri2 =
                DelaunayTessellation::create_tessellation( d_ndim, N_in, x_in, &d_tri, nullptr );
    } else {
        throw std::logic_error( "Unsupported dimension" );
    }
    if ( N_tri2 < N_in - d_ndim ) {
        AMP::perr << "Error creating tessellation (" << N_tri2 << ")\n";
        d_N_tri = 0;
        delete[] d_tri;
        d_tri = nullptr;
        if ( d_tri_nab != nullptr )
            delete[] d_tri_nab;
        d_tri_nab              = nullptr;
        std::string debug_file = "DelaunayTessellation_failed_points";
        if ( !debug_file.empty() ) {
            // Write a debug file with the failed points
            FILE *pFile = fopen( debug_file.c_str(), "wb" );
            write_failed_points<TYPE>( d_ndim, N_in, x_in, pFile );
            fclose( pFile );
            AMP::perr << "  Failed points written to " << debug_file << std::endl;
        }
        PROFILE_STOP2( "create_tessellation", PROFILE_LEVEL );
        return N_tri2;
    }
    d_N_tri = N_tri2;
    // Copy the data
    d_N_node_sum = 0;
    if ( d_level == 2 || d_level == 4 ) {
        d_x = new TYPE[d_ndim * d_N];
        for ( size_t i = 0; i < d_ndim * d_N; i++ )
            d_x[i] = x_in[i];
    } else {
        d_x = const_cast<TYPE *>( x_in );
    }
    PROFILE_STOP( "create_tessellation", PROFILE_LEVEL );
    return 0;
}


/********************************************************************
 * Function to construct the tessellation using a given tessellation *
 ********************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_tessellation(
    const int N_in, const TYPE x_in[], const int base, const int N_tri_in, const int tri_in[] )
{
    // Delete the existing data
    if ( d_x != nullptr ) {
        if ( d_level == 2 || d_level == 4 )
            delete[] d_x;
        d_x = nullptr;
    }
    if ( d_tri != nullptr ) {
        delete[] d_tri;
        d_tri = nullptr;
    }
    if ( d_N_node != nullptr ) {
        delete[] d_N_node;
        delete[] d_node_list[0];
        delete[] d_node_list;
        d_N_node    = nullptr;
        d_node_list = nullptr;
    }
    if ( d_tri_nab != nullptr ) {
        delete[] d_tri_nab;
        d_tri_nab = nullptr;
    }
    // Special case where we do not want to create the tessellation
    if ( N_tri_in == 0 ) {
        d_N          = 0;
        d_N_tri      = 0;
        d_tri        = nullptr;
        d_x          = nullptr;
        d_N_node_sum = 0;
        d_N_node     = nullptr;
        d_node_list  = nullptr;
        d_tri_nab    = nullptr;
        return;
    }
    // Check the inputs
    AMP_ASSERT( d_ndim > 0 );
    AMP_ASSERT( N_in > 0 );
    AMP_ASSERT( x_in != nullptr );
    AMP_ASSERT( N_tri_in > 0 );
    AMP_ASSERT( tri_in != nullptr );
    // Copy the data
    d_N          = N_in;
    d_N_tri      = N_tri_in;
    d_N_node_sum = 0;
    if ( d_level == 2 || d_level == 4 ) {
        d_x = new TYPE[d_ndim * d_N];
        for ( size_t i = 0; i < d_ndim * d_N; i++ )
            d_x[i] = x_in[i];
    } else {
        d_x = const_cast<TYPE *>( x_in );
    }
    d_tri = new int[( d_ndim + 1 ) * d_N_tri];
    for ( size_t i = 0; i < ( d_ndim + 1 ) * d_N_tri; i++ )
        d_tri[i] = tri_in[i] - base;
}


/********************************************************************
 * Function to update the coordinates                                *
 ********************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::update_coordinates( const TYPE x_in[] )
{
    AMP_ASSERT( x_in != nullptr );
    if ( d_level == 2 || d_level == 4 ) {
        d_x = const_cast<TYPE *>( x_in );
    } else {
        d_x = new TYPE[d_ndim * d_N];
        for ( size_t i = 0; i < d_ndim * d_N; i++ )
            d_x[i] = x_in[i];
    }
}


/************************************************************************
 * This function creates the kdtree                                      *
 ************************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_kdtree() const
{
    // Create the kdtree
    if ( d_tree == nullptr ) {
        PROFILE_START( "create_kdtree", PROFILE_LEVEL );
        double *x[NDIM_MAX];
        for ( size_t d = 0; d < d_ndim; d++ ) {
            x[d] = new double[d_N];
            for ( size_t i = 0; i < d_N; i++ )
                x[d][i] = d_x[d + i * d_ndim];
        }
        d_tree = new kdtree( d_ndim, d_N, x );
        for ( size_t d = 0; d < d_ndim; d++ )
            delete[] x[d];
        PROFILE_STOP( "create_kdtree", PROFILE_LEVEL );
    }
}


/************************************************************************
 * This function find the nearest point to the desired point             *
 ************************************************************************/
template<class TYPE>
template<class TYPE2>
void DelaunayInterpolation<TYPE>::find_nearest( const unsigned int Ni,
                                                const TYPE2 xi[],
                                                const int base,
                                                unsigned int *index )
{
    if ( Ni == 0 )
        return;
    PROFILE_START( "find_nearest", PROFILE_LEVEL );
    // Create the kdtree
    create_kdtree();
    // Use the kdtree to perform the nearest neighbor interpolation
    auto index2 = new size_t[Ni];
    auto xi2    = new double[Ni * d_ndim];
    for ( size_t i = 0; i < Ni * d_ndim; i++ )
        xi2[i] = xi[i];
    d_tree->find_nearest( Ni, xi2, index2 );
    delete[] xi2;
    for ( size_t i = 0; i < Ni; i++ )
        index[i] = static_cast<unsigned int>( index2[i] + base );
    delete[] index2;
    PROFILE_STOP( "find_nearest", PROFILE_LEVEL );
}


/************************************************************************
 * This function find the triangle that contains the point               *
 * Note: Most of the time in this function is spent getting the neighbor *
 * triangle lists.                                                       *
 * Ex: in 3d with 1e5 points:                                            *
 *   70% is spent in get_tri_tri, 10% in the loop, 15% in get_node_node  *
 ************************************************************************/
template<class TYPE>
template<class TYPE2>
void DelaunayInterpolation<TYPE>::find_tri(
    const unsigned int Ni, const TYPE2 xi[], const int base, int index[], bool extrap )
{
    if ( Ni == 0 )
        return;
    PROFILE_START( "find_tri", PROFILE_LEVEL );
    // Create the kdtree
    create_kdtree();
    // Create a list of the nodes that link to every other node
    create_node_neighbors();
    // For each triangle, get a list of the triangles that are neighbors
    create_tri_neighbors();
    // For each node, get a starting triangle
    create_node_tri();
    // First choose a starting triangle
    auto index_node = new unsigned int[Ni];
    find_nearest( Ni, xi, 0, index_node );
    for ( unsigned int i = 0; i < Ni; i++ ) {
        index[i] = d_node_tri[index_node[i]] + base;
    }
    delete[] index_node;
    index_node = nullptr;
    // Loop through the query points
    unsigned char Nd = d_ndim + 1;
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )], xi2[NDIM_MAX], L[NDIM_MAX + 1];
    bool failed_search = false;
    size_t N_it_tot    = 0;
    for ( unsigned int i = 0; i < Ni; i++ ) {
        for ( unsigned int j = 0; j < d_ndim; j++ )
            xi2[j] = xi[j + i * d_ndim];
        int current_tri = index[i] - base;
        size_t it       = 0;
        while ( true ) {
            // Get the point in Barycentric coordinates
            for ( int j1 = 0; j1 < Nd; j1++ ) {
                int j = d_tri[j1 + current_tri * Nd];
                for ( int j2 = 0; j2 < d_ndim; j2++ )
                    x2[j2 + j1 * d_ndim] = d_x[j2 + j * d_ndim];
            }
            compute_Barycentric( d_ndim, x2, xi2, L );
            // We are inside the triangle if all coordinates are in the range [0,1]
            bool in_triangle = true;
            for ( int j = 0; j < Nd; j++ ) {
                if ( L[j] < -TRI_TOL || L[j] > 1.0 + TRI_TOL ) {
                    in_triangle = false;
                    break;
                }
            }
            if ( in_triangle ) {
                // Success, save the index and move on
                index[i] = current_tri + base;
                break;
            }
            // We are outside an edge of the triangle iff the coordinate for the edge is < 0
            // Check if we are outside the convex hull (we will be outside an edge which is on the
            // convex hull)
            bool outside_hull = false;
            for ( int j = 0; j < Nd; j++ ) {
                if ( L[j] < -TRI_TOL && d_tri_nab[j + current_tri * Nd] == -1 )
                    outside_hull = true;
            }
            if ( outside_hull ) {
                if ( !extrap ) {
                    // We are outside the convex hull, store the error, and continue
                    index[i] = -1;
                    break;
                }
                // Zero all values of L for the sides on the convex hull
                bool finished = true;
                for ( int j = 0; j < Nd; j++ ) {
                    if ( d_tri_nab[j + current_tri * Nd] == -1 )
                        L[j] = 0.0;
                    else if ( L[j] < -TRI_TOL )
                        finished = false;
                }
                // If no remaining coordiantes are negitive, this is the nearest triangle
                if ( finished ) {
                    index[i] = current_tri + base;
                    break;
                }
            }
            // We want to advance in the direction that is the most negitive
            int k      = 0;
            double val = 0.0;
            for ( int j = 0; j < Nd; j++ ) {
                if ( L[j] < val ) {
                    val = L[j];
                    k   = j;
                }
            }
            int next_tri = d_tri_nab[k + current_tri * Nd] - base;
            if ( next_tri < 0 || next_tri > ( (int) d_N_tri ) || next_tri == current_tri )
                throw std::logic_error( "Internal error" );
            current_tri = next_tri;
            it++;
            if ( it > d_N_tri + 10 ) {
                // We should never revisit a triangle, so we must have some ended in an infinite
                // cycle
                failed_search = true;
                index[i]      = -2;
                break;
            }
        }
        N_it_tot += it;
    }
    // Check the average number of triangles searched,
    // this should be relatively small since we start with a tirangle that contains the nearest
    // point
    if ( N_it_tot / Ni > 25 )
        AMP::perr
            << "Search took longer than it should, there may be a problem with the tessellation\n";
    // Check if the search failed for any values
    if ( failed_search )
        AMP::perr << "Search failed for some points\n";
    // Remove temporary data that we are not storing (based on the storage level)
    if ( d_level != 3 && d_level != 4 ) {
        delete[] d_N_node;
        delete[] d_node_list[0];
        delete[] d_node_list;
        delete[] d_tri_nab;
        d_N_node_sum = 0;
        d_N_node     = nullptr;
        d_node_list  = nullptr;
        d_tri_nab    = nullptr;
    }
    PROFILE_STOP( "find_tri", PROFILE_LEVEL );
}


/****************************************************************
 * Function to get a list of the nodes that connect to each node *
 ****************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::calc_node_gradient( const double *f,
                                                      const int method,
                                                      double *grad,
                                                      const int n_it )
{
    PROFILE_START( "calc_node_gradient", PROFILE_LEVEL );
    // First we need to get a list of the nodes that link to every other node
    create_node_neighbors();
    if ( method == 1 ) {
        // We are performing a local gradient calculation using a least-squares minimization
        // Calculate the derivative for each edge that touches the node, then
        // use a weighted least-squares minimization to get the gradient
        double M[NDIM_MAX * NDIM_MAX], rhs[NDIM_MAX], rh[NDIM_MAX];
        for ( unsigned int i = 0; i < d_N; i++ ) {
            // Initialize M, rhs
            for ( double &j : M )
                j = 0.0;
            for ( int j = 0; j < NDIM_MAX; j++ ) {
                rhs[j] = 0.0;
                rh[j]  = 0.0;
            }
            // Loop through the neighbor nodes, contructing the matrix and rhs
            for ( unsigned int j = 0; j < d_N_node[i]; j++ ) {
                int k = d_node_list[i][j];
                // Comupute the distance beween the neighbors and the direction vector
                double r = 0.0;
                for ( int n = 0; n < d_ndim; n++ ) {
                    rh[n] = ( d_x[n + k * d_ndim] - d_x[n + i * d_ndim] );
                    r += rh[n] * rh[n];
                }
                r = sqrt( r );
                for ( int n = 0; n < d_ndim; n++ )
                    rh[n] /= r;
                // Compute the derivative of f in the direction of the neighbor
                double df_dr = ( f[k] - f[i] ) / r;
                // Get the weights (use the inverse distance)
                double wi = 1.0 / r;
                // Construct the linear system
                for ( int j1 = 0; j1 < d_ndim; j1++ ) {
                    for ( int j2 = 0; j2 < d_ndim; j2++ ) {
                        M[j1 + j2 * d_ndim] += 2.0 * wi * rh[j1] * rh[j2];
                    }
                    rhs[j1] += 2.0 * wi * rh[j1] * df_dr;
                }
            }
            // Solve the linear system to get the local gradient
            solve_system( d_ndim, M, rhs, &grad[i * d_ndim] );
        }
    } else if ( method == 2 || method == 3 ) {
        /* Both methods 2 and 3 use a higher order approximation for the derivative which
         * requires the function value and the gradient at the neighbor points, and then
         * performs a least squares minimization.  This reduces the error of the gradient,
         * but requires solving a system of size N*d_ndim x N*d_ndim.  The resulting system is
         * sparse.
         * Method 2 solves the sparse system directly, while method 3 uses Gauss-Seidel
         * iteration to improve the solution calculated by method 1.  Note: method 2 in
         * only implimented in the MATLAB prototype since it would require linking to a
         * sparse matrix solve, and can have significant memory requirements.
         * Method 3 does not have these limitations, and for most systems only 10-20 iterations are
         * necessary.
         * The construction of the matrix is:
         * Looking at the taylor series:
         *   f(x) = f(x0) + f'(x0)*(x-x0) + 1/2*f''(x0)*(x-x0)^2 + ...
         * We can then approximate the second derivative:
         *   f''(x) = (f'(x)-f'(x0))/(x-x0)
         * Combining we get:
         *   f'(x)+f'(x0) = 2*(f(x)-f(x0))/(x-x0)
         * Using this for the approximation of the derivative at the kth point:
         *   Si = sum(wik*(dot(gi,r^)+dot(gk,r^)-2/r*(fk-fi))^2)
         * Then we can perform the minimization on S to get a linear system for each
         * component of the gradient of the kth point.
         * The resulting system will be a block system with d_ndimxd_ndim blocks.
         */
        if ( method == 2 ) {
            AMP::perr << "This method is not implimented\n";
            return;
        }
        // First, allocate storage to store the matrix components
        auto A   = new double[d_ndim * d_ndim * d_N_node_sum];
        auto D   = new double[d_ndim * d_ndim * d_N];
        auto rhs = new double[d_ndim * d_N];
        // Initialize the matrix elements and rhs to zero
        memset( A, 0, d_ndim * d_ndim * d_N_node_sum * sizeof( double ) );
        memset( D, 0, d_ndim * d_ndim * d_N * sizeof( double ) );
        memset( rhs, 0, d_ndim * d_N * sizeof( double ) );
        // Loop through the nodes, constructing the matrix elements
        int m = 0;
        for ( unsigned int i = 0; i < d_N; i++ ) {
            // Loop through the neighbor nodes, contructing the matrix and rhs
            for ( unsigned int j = 0; j < d_N_node[i]; j++ ) {
                int k = d_node_list[i][j];
                // Comupute the distance beween the neighbors and the direction vector
                double r = 0.0;
                double rh[NDIM_MAX];
                for ( int n = 0; n < d_ndim; n++ ) {
                    rh[n] = ( d_x[n + k * d_ndim] - d_x[n + i * d_ndim] );
                    r += rh[n] * rh[n];
                }
                r = sqrt( r );
                for ( int n = 0; n < d_ndim; n++ )
                    rh[n] /= r;
                // Compute the derivative of f in the direction of the neighbor
                double df_dr = ( f[k] - f[i] ) / r;
                // Get the weights (use the inverse distance)
                double wi = 1.0 / r;
                // Construct the linear system
                for ( int j1 = 0; j1 < d_ndim; j1++ ) {
                    for ( int j2 = 0; j2 < d_ndim; j2++ ) {
                        A[j1 + j2 * d_ndim + m * d_ndim * d_ndim] = 2.0 * wi * rh[j1] * rh[j2];
                        D[j1 + j2 * d_ndim + i * d_ndim * d_ndim] +=
                            A[j1 + j2 * d_ndim + m * d_ndim * d_ndim];
                    }
                    rhs[j1 + i * d_ndim] += 2.0 * wi * rh[j1] * 2.0 * df_dr;
                }
                m++;
            }
        }
        if ( method == 2 ) {
            // Construct the sparse system and solve it directly
            // NOT implimented
            AMP::perr << "This method is not implemented\n";
        } else if ( method == 3 ) {
            // Use a block Gauss-Seidel method to improve the solution computed by method 1
            // First we need to compute the solution using method 1.  We can do this by noting
            // That D(:,:,i) is the same as method 1, and the rhs is a factor of 2 larger
            // than in method 1
            double rhs2[NDIM_MAX];
            for ( unsigned int i = 0; i < d_N; i++ ) {
                for ( int j = 0; j < d_ndim; j++ )
                    rhs2[j] = 0.5 * rhs[j + i * d_ndim];
                solve_system( d_ndim, &D[i * d_ndim * d_ndim], rhs2, &grad[i * d_ndim] );
            }
            // Now we can perform a block Gauss-Seidel iteration to improve the solution
            Gauss_Seidel( d_ndim,
                          (unsigned int) d_N,
                          (unsigned int) d_N_node_sum,
                          D,
                          d_N_node,
                          d_node_list,
                          A,
                          rhs,
                          grad,
                          n_it );
        } else {
            AMP::perr << "Unknown method\n";
        }
        // Free the temporary memory
        delete[] A;
        delete[] D;
        delete[] rhs;
    } else if ( method == 4 ) {
        // This is the same as method 3 (Gauss-Seidel) but does not store the matrix entries,
        //    instead they are re-created every time
        // First, lets get the initial solution using method 1
        PROFILE_STOP2( "calc_node_gradient", PROFILE_LEVEL );
        calc_node_gradient( f, 1, grad );
        PROFILE_START2( "calc_node_gradient", PROFILE_LEVEL );
        // Loop through the Gauss-Seidel iterations
        double D[NDIM_MAX * NDIM_MAX] = { 0 }, rhs[NDIM_MAX] = { 0 }, Ax[NDIM_MAX] = { 0 },
                            rh[NDIM_MAX] = { 0 };
        for ( int it = 0; it < n_it; it++ ) {
            // Loop through the nodes updating x
            //    x(k+1) = aii^-1*(bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i))
            for ( unsigned int i = 0; i < d_N; i++ ) {
                for ( int j = 0; j < d_ndim * d_ndim; j++ )
                    D[j] = 0.0;
                for ( int j = 0; j < d_ndim; j++ ) {
                    rhs[j] = 0.0;
                    Ax[j]  = 0.0;
                }
                // Loop through the neighbor nodes, contructing D, A*x and rhs
                for ( unsigned int j = 0; j < d_N_node[i]; j++ ) {
                    int k = d_node_list[i][j];
                    // Comupute the distance beween the neighbors and the direction vector
                    double r = 0.0;
                    for ( int n = 0; n < d_ndim; n++ ) {
                        rh[n] = ( d_x[n + k * d_ndim] - d_x[n + i * d_ndim] );
                        r += rh[n] * rh[n];
                    }
                    r = sqrt( r );
                    for ( int n = 0; n < d_ndim; n++ )
                        rh[n] /= r;
                    // Compute the derivative of f in the direction of the neighbor
                    double df_dr = ( f[k] - f[i] ) / r;
                    // Get the weights (use the inverse distance)
                    double wi = 1.0 / r;
                    // Construct the linear system
                    for ( int j1 = 0; j1 < d_ndim; j1++ ) {
                        for ( int j2 = 0; j2 < d_ndim; j2++ ) {
                            double tmp = 2.0 * wi * rh[j1] * rh[j2];
                            Ax[j1] += tmp * grad[j2 + k * d_ndim];
                            D[j1 + j2 * d_ndim] += tmp;
                        }
                        rhs[j1] += 2.0 * wi * rh[j1] * 2.0 * df_dr;
                    }
                }
                // Update x
                for ( int j = 0; j < d_ndim; j++ )
                    rhs[j] -= Ax[j];
                solve_system( d_ndim, D, rhs, &grad[i * d_ndim] );
            }
        }
    } else {
        // Unkown method
        AMP::perr << "Unknown method\n";
    }
    // Update the internal storage using set_storage_level
    set_storage_level( d_level );
    PROFILE_STOP( "calc_node_gradient", PROFILE_LEVEL );
}


/********************************************************************
 * Function to perform nearest-neighbor interpolation                *
 ********************************************************************/
template<class TYPE>
template<class TYPE2>
void DelaunayInterpolation<TYPE>::interp_nearest( const double f[],
                                                  const unsigned int Ni,
                                                  const TYPE2 xi[],
                                                  const unsigned int nearest[],
                                                  double *fi )
{
    PROFILE_START( "interp_nearest", PROFILE_LEVEL );
    AMP_ASSERT( f != nullptr );
    AMP_ASSERT( xi != nullptr );
    AMP_ASSERT( fi != nullptr );
    for ( unsigned int i = 0; i < Ni; i++ )
        fi[i] = f[nearest[i]];
    PROFILE_STOP( "interp_nearest", PROFILE_LEVEL );
}


/********************************************************************
 * Function to perform linear interpolation                          *
 ********************************************************************/
template<class TYPE>
template<class TYPE2>
void DelaunayInterpolation<TYPE>::interp_linear( const double f[],
                                                 const unsigned int Ni,
                                                 const TYPE2 xi[],
                                                 const int index[],
                                                 double *fi,
                                                 double *gi,
                                                 bool extrap )
{
    PROFILE_START( "interp_linear", PROFILE_LEVEL );
    AMP_ASSERT( d_x != nullptr );
    AMP_ASSERT( f != nullptr );
    AMP_ASSERT( xi != nullptr );
    AMP_ASSERT( fi != nullptr );
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )], f2[NDIM_MAX + 1], L[NDIM_MAX + 1];
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    for ( unsigned int i = 0; i < Ni; i++ ) {
        // Check if the triange index is valid
        if ( index[i] == -1 && !extrap ) {
            fi[i] = NaN;
            continue;
        } else if ( index[i] < 0 ) {
            throw std::logic_error( "Invalid triangle specified" );
        }
        // Compute the Barycentric coordinates
        for ( int j = 0; j < d_ndim + 1; j++ ) {
            unsigned int k = d_tri[j + index[i] * ( d_ndim + 1 )];
            for ( int j2 = 0; j2 < d_ndim; j2++ )
                x2[j2 + j * d_ndim] = d_x[j2 + k * d_ndim];
            f2[j] = f[k];
        }
        double xi2[NDIM_MAX];
        for ( int j = 0; j < d_ndim; j++ )
            xi2[j] = xi[i * d_ndim + j];
        compute_Barycentric( d_ndim, x2, xi2, L );
        if ( !extrap ) {
            bool outside = false;
            for ( int d = 0; d < d_ndim + 1; d++ ) {
                if ( L[d] < -1e-8 )
                    outside = true;
            }
            if ( outside ) {
                fi[i] = NaN;
                continue;
            }
        }
        // Perform the linear interpolation
        fi[i] = 0.0;
        for ( int j = 0; j < d_ndim + 1; j++ )
            fi[i] += L[j] * f2[j];
        // Compute the gradient
        if ( gi != nullptr )
            compute_gradient( d_ndim, x2, f2, &gi[i * d_ndim] );
    }
    PROFILE_STOP( "interp_linear", PROFILE_LEVEL );
}


/****************************************************************
 * Function to perform cubic interpolation                       *
 ****************************************************************/
template<class TYPE>
template<class TYPE2>
void DelaunayInterpolation<TYPE>::interp_cubic( const double f[],
                                                const double g[],
                                                const unsigned int Ni,
                                                const TYPE2 xi[],
                                                const int index[],
                                                double *fi,
                                                double *gi_out,
                                                int extrap )
{
    PROFILE_START( "interp_cubic", PROFILE_LEVEL );
    AMP_ASSERT( d_x != nullptr );
    AMP_ASSERT( f != nullptr );
    AMP_ASSERT( g != nullptr );
    AMP_ASSERT( xi != nullptr );
    AMP_ASSERT( fi != nullptr );
    double xi0[NDIM_MAX], gi[NDIM_MAX];
    for ( unsigned int i = 0; i < Ni; i++ ) {
        for ( int j = 0; j < d_ndim; j++ )
            xi0[j] = xi[i * d_ndim + j];
        interp_cubic_single( f, g, xi0, index[i], fi[i], gi, extrap );
        if ( gi_out != nullptr ) {
            for ( int j = 0; j < d_ndim; j++ )
                gi_out[j + i * d_ndim] = gi[j];
        }
    }
    PROFILE_STOP( "interp_cubic", PROFILE_LEVEL );
}
template<class TYPE>
void DelaunayInterpolation<TYPE>::interp_cubic_single( const double f[],
                                                       const double g[],
                                                       const double xi[],
                                                       const int index,
                                                       double &fi,
                                                       double *gi,
                                                       int extrap )
{
    const bool check_collinear = true; // Do we want to perform checks that points are collinear
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )], f2[NDIM_MAX + 1];
    double g2[NDIM_MAX * ( NDIM_MAX + 1 )], L[NDIM_MAX + 1];
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Check if the triange index is valid
    if ( index == -1 && extrap == 0 ) {
        fi = std::numeric_limits<double>::quiet_NaN();
        for ( int j = 0; j < d_ndim + 1; j++ )
            gi[j] = nan;
        return;
    } else if ( index < 0 ) {
        PROFILE_STOP2( "interp_cubic", PROFILE_LEVEL );
        throw std::logic_error( "Invalid triangle specified" );
    }
    // Compute the Barycentric coordinates
    for ( int j = 0; j < d_ndim + 1; j++ ) {
        unsigned int k = d_tri[j + index * ( d_ndim + 1 )];
        for ( int j2 = 0; j2 < d_ndim; j2++ ) {
            x2[j2 + j * d_ndim] = d_x[j2 + k * d_ndim];
            g2[j2 + j * d_ndim] = g[j2 + k * d_ndim];
        }
        f2[j] = f[k];
    }
    compute_Barycentric( d_ndim, x2, xi, L );
    for ( int j = 0; j < d_ndim + 1; j++ ) {
        if ( fabs( L[j] ) < TRI_TOL )
            L[j] = 0.0;
    }
    // Count the number of zero-valued and negitive dimensions
    int N_L_zero = 0;
    int N_L_neg  = 0;
    for ( int j = 0; j < d_ndim + 1; j++ ) {
        N_L_zero += ( L[j] == 0.0 ) ? 1 : 0;
        N_L_neg += ( L[j] < 0.0 ) ? 1 : 0;
    }
    if ( N_L_zero == d_ndim ) {
        // We are at a vertex
        for ( int j = 0; j < d_ndim + 1; j++ ) {
            if ( L[j] != 0.0 ) {
                fi = f2[j];
                for ( int j2 = 0; j2 < d_ndim; j2++ )
                    gi[j2] = g2[j2 + j * d_ndim];
                break;
            }
        }
    } else if ( N_L_zero == 0 && N_L_neg == 0 ) {
        // No zero-valued or negivie dimensions, begin the interpolation
        fi = interp_cubic_recursive( d_ndim, d_ndim + 1, x2, f2, g2, xi, L, gi );
    } else {
        // Remove any directions that are 0 (edge, face, etc.)
        int N = d_ndim + 1 - N_L_zero;
        double x3[NDIM_MAX * ( NDIM_MAX + 1 )], f3[NDIM_MAX + 1], g3[NDIM_MAX * ( NDIM_MAX + 1 )],
            L3[NDIM_MAX + 1];
        int k = 0;
        for ( int j1 = 0; j1 < d_ndim + 1; j1++ ) {
            int k2 = -1;
            if ( L[j1] != 0.0 ) {
                k2 = k;
                k++;
            } else {
                k2 = d_ndim - ( j1 - k );
            }
            f3[k2] = f2[j1];
            L3[k2] = L[j1];
            for ( int j2 = 0; j2 < d_ndim; j2++ ) {
                x3[j2 + k2 * d_ndim] = x2[j2 + j1 * d_ndim];
                g3[j2 + k2 * d_ndim] = g2[j2 + j1 * d_ndim];
            }
        }
        if ( N_L_neg == 0 ) {
            // No negitive directions, we are ready to perform the interpolation
            fi = interp_cubic_recursive( d_ndim, N, x3, f3, g3, xi, L3, gi );
        } else {
            // We need to deal with the negitive coordinates
            if ( extrap == 0 ) {
                fi = std::numeric_limits<double>::quiet_NaN();
                for ( int d = 0; d < d_ndim; d++ )
                    gi[d] = std::numeric_limits<double>::quiet_NaN();
            } else if ( extrap == 1 ) {
                // Use linear interpolation based on the nearest node and it's gradient
                double dist = 1e100;
                int index   = 0;
                for ( int j = 0; j < d_ndim + 1; j++ ) {
                    double dist2 = 0.0;
                    for ( int d = 0; d < d_ndim; d++ )
                        dist2 += ( xi[d] - x2[d + j * d_ndim] ) * ( xi[d] - x2[d + j * d_ndim] );
                    if ( dist2 < dist ) {
                        index = j;
                        dist  = dist2;
                    }
                }
                fi = f2[index];
                for ( int d = 0; d < d_ndim; d++ ) {
                    fi += g2[d + index * d_ndim] * ( xi[d] - x2[d + index * d_ndim] );
                    gi[d] = g2[d + index * d_ndim];
                }
            } else if ( extrap == 2 ) {
                // Use quadratic interpolation
                if ( N == 2 ) {
                    // We can perform interpolation along a line
                    fi = interp_cubic_recursive( d_ndim, N, x3, f3, g3, xi, L3, gi );
                } else {
                    // Choose two points within (or on) the triangle
                    double xi1[NDIM_MAX], xi2[NDIM_MAX], fi1, fi2, gi1[NDIM_MAX], gi2[NDIM_MAX];
                    get_interpolant_points( d_ndim, N, x3, L3, xi, xi1, xi2, check_collinear );
                    // Use cubic interpolation to get f and g for the two points
                    PROFILE_STOP2( "interp_cubic", PROFILE_LEVEL );
                    interp_cubic_single( f, g, xi1, index, fi1, gi1, 0 );
                    interp_cubic_single( f, g, xi2, index, fi2, gi2, 0 );
                    PROFILE_START2( "interp_cubic", PROFILE_LEVEL );
                    // Perform quadratic interpolation using a linear approximation to the gradient
                    fi = interp_line( d_ndim, xi1, fi1, gi1, xi2, fi2, gi2, xi, gi );
                }
            } else {
                PROFILE_STOP2( "interp_cubic", PROFILE_LEVEL );
                throw std::logic_error( "Invalid value for extrap" );
            }
        }
    }
}


/****************************************************************
 * This function performs cubic interpolation recursively.       *
 ****************************************************************/
double interp_cubic_recursive( const int d_ndim,
                               const int N,
                               const double *x,
                               const double *f,
                               const double *g,
                               const double *xi,
                               const double *L,
                               double *gi )
{
    double fi = 0.0;
    if ( N == 2 ) {
        // We are at an edge, perform interpolation along a line
        fi = interp_line( d_ndim, &x[0], f[0], &g[0], &x[d_ndim], f[1], &g[d_ndim], xi, gi );
        return fi;
    }
    // Check that we have no negitive coordinates
    for ( int i = 0; i < N; i++ ) {
        if ( L[i] <= 0.0 ) {
            AMP::perr << "Internal error: negitive Barycentric coordinates\n";
            return std::numeric_limits<double>::quiet_NaN();
        }
    }
    // Step 1: Find the point of intersection between the line
    //    through each vertex and the opposite edge, face, etc.
    // We can easily compute this using the Barycentric coordinates we computed earlier
    double P[NDIM_MAX + 1][NDIM_MAX];
    for ( int i = 0; i < N; i++ ) {
        double L2[NDIM_MAX + 1];
        double tmp = 0.0;
        L2[i]      = 0.0;
        for ( int j = 0; j < N; j++ ) {
            if ( i == j )
                continue;
            L2[j] = L[j];
            tmp += L[j];
        }
        tmp = 1.0 / tmp;
        for ( int j = 0; j < N; j++ )
            L2[j] *= tmp;
        for ( int j = 0; j < d_ndim; j++ ) {
            P[i][j] = 0.0;
            for ( int k = 0; k < N; k++ )
                P[i][j] += x[j + k * d_ndim] * L2[k];
        }
    }
    // Step 2: For each point in P, interpolate f and the gradient
    double Pf[NDIM_MAX + 1];
    double Pg[NDIM_MAX + 1][NDIM_MAX];
    for ( int i = 0; i < N; i++ ) {
        double x2[( NDIM_MAX + 1 ) * NDIM_MAX], f2[NDIM_MAX + 1], g2[( NDIM_MAX + 1 ) * NDIM_MAX],
            L2[NDIM_MAX + 1];
        int k = 0;
        for ( int j = 0; j < N; j++ ) {
            if ( i == j )
                continue;
            f2[k] = f[j];
            L2[k] = L[j];
            for ( int n = 0; n < d_ndim; n++ ) {
                x2[n + k * d_ndim] = x[n + j * d_ndim];
                g2[n + k * d_ndim] = g[n + j * d_ndim];
            }
            k++;
        }
        Pf[i] = interp_cubic_recursive( d_ndim, N - 1, x2, f2, g2, P[i], L2, Pg[i] );
    }
    // Step 3: For each vertex/point pair, perform interpolation along the line
    // to get the solution at the desired point (there wil be N approximations)
    double f1[NDIM_MAX + 1], g1[NDIM_MAX + 1][NDIM_MAX];
    for ( int i = 0; i < N; i++ ) {
        f1[i] = interp_line(
            d_ndim, &x[i * d_ndim], f[i], &g[i * d_ndim], P[i], Pf[i], Pg[i], xi, g1[i] );
    }
    // Step 4: Perform a weighted average of the solutions.
    double w[NDIM_MAX + 1];
    for ( int i = 0; i < N; i++ )
        w[i] = 1.0 / ( (double) N ); // Use an average weight for now
    fi = 0.0;
    for ( int i = 0; i < N; i++ )
        fi += w[i] * f1[i];
    if ( gi != nullptr ) {
        for ( int i = 0; i < d_ndim; i++ ) {
            gi[i] = 0.0;
            for ( int j = 0; j < N; j++ )
                gi[i] += w[i] * g1[j][i];
        }
    }
    return fi;
}


/****************************************************************
 * Function to get two points within a triangle to use for       *
 * interpolation when the desired point is outside the triangle  *
 ****************************************************************/
static void get_interpolant_points( const int d_ndim,
                                    const int N,
                                    const double *x,
                                    const double *L,
                                    const double *xi,
                                    double *xi1,
                                    double *xi2,
                                    bool check )
{
    int N_neg = 0;
    for ( int i = 0; i < N; i++ ) {
        if ( L[i] < 0.0 )
            N_neg++;
    }
    double L1[NDIM_MAX + 1], L2[NDIM_MAX + 1];
    memset( L1, 0, ( NDIM_MAX + 1 ) * sizeof( double ) );
    memset( L2, 0, ( NDIM_MAX + 1 ) * sizeof( double ) );
    if ( N_neg == 1 || N_neg == N - 1 ) {
        // We have one point that is the opposite sign
        // Choose that point and the intersection with the opposite face
        double sign = ( N_neg == 1 ) ? 1.0 : -1.0;
        for ( int i = 0; i < N; i++ ) {
            L1[i] = 0.0;
            if ( sign * L[i] < 0.0 ) {
                L1[i] = 1.0;
                L2[i] = 0.0;
                for ( int j = 0; j < d_ndim; j++ )
                    xi1[j] = x[i * d_ndim + j];
            } else {
                L2[i] = fabs( L[i] );
            }
        }
        double tmp = 0;
        for ( int i = 0; i < N; i++ )
            tmp += L2[i];
        tmp = 1.0 / tmp;
        for ( int i = 0; i < N; i++ )
            L2[i] *= tmp;
    } else if ( N_neg == 2 && N == 4 ) {
        // Choose the points on the two edges connecting the two posisitve (or negitive) points to
        // each other
        double tmp1 = 0, tmp2 = 0;
        for ( int i = 0; i < N; i++ ) {
            L1[i] = 0.0;
            L2[i] = 0.0;
            if ( L[i] > 0.0 ) {
                L1[i] = L[i];
                tmp1 += L1[i];
            } else {
                L2[i] = -L[i];
                tmp2 += L2[i];
            }
        }
        tmp1 = 1.0 / tmp1;
        tmp2 = 1.0 / tmp2;
        for ( int i = 0; i < N; i++ ) {
            L1[i] *= tmp1;
            L2[i] *= tmp2;
        }
    } else {
        throw std::logic_error( "Error: Unhandled case" );
    }
    for ( int j = 0; j < d_ndim; j++ ) {
        xi1[j] = 0.0;
        xi2[j] = 0.0;
        for ( int k = 0; k < N; k++ ) {
            xi1[j] += x[j + k * d_ndim] * L1[k];
            xi2[j] += x[j + k * d_ndim] * L2[k];
        }
    }
    // Check that the three points are collinear
    if ( check ) {
        bool collinear = false;
        double a[NDIM_MAX], b[NDIM_MAX];
        double d2[2] = { 0, 0 };
        for ( int i = 0; i < d_ndim; i++ ) {
            a[i] = xi[i] - xi1[i];
            b[i] = xi[i] - xi2[i];
            d2[0] += a[i] * a[i];
            d2[1] += b[i] * b[i];
        }
        const double tol = 1e-8 * std::max( d2[0], d2[1] );
        if ( d_ndim == 2 ) {
            double c  = a[0] * b[1] - a[1] * b[0];
            collinear = fabs( c ) < tol;
        } else if ( d_ndim == 3 ) {
            double c[NDIM_MAX];
            c[0]      = a[1] * b[2] - a[2] * b[1];
            c[1]      = a[2] * b[0] - a[0] * b[2];
            c[2]      = a[0] * b[1] - a[1] * b[0];
            collinear = fabs( c[0] ) < tol && fabs( c[1] ) < tol && fabs( c[2] ) < tol;
        } else {
            throw std::logic_error( "Not programmed for this dimension yet" );
        }
        if ( !collinear ) {
            char tmp[100];
            sprintf( tmp, "get_interpolant_points failed: collinear (%i,%i)", N_neg, N );
            throw std::logic_error( tmp );
        }
    }
}


/****************************************************************
 * Function to interpolate along a line                          *
 * Note: if x is outside the line between x1 and x2, then we     *
 *    will perform quadratic interpolation using a linear        *
 *    approximation for the gradient.                            *
 ****************************************************************/
double interp_line( const int n,
                    const double *x0,
                    const double f0,
                    const double *g0,
                    const double *x1,
                    const double f1,
                    const double *g1,
                    const double *x,
                    double *g )
{
    // Get the length of the line and the position of x on the line
    double r   = 0.0;
    double rx  = 0.0;
    double dot = 0.0;
    for ( int i = 0; i < n; i++ ) {
        r += ( x1[i] - x0[i] ) * ( x1[i] - x0[i] );
        rx += ( x[i] - x0[i] ) * ( x[i] - x0[i] );
        dot += ( x[i] - x0[i] ) * ( x1[i] - x0[i] );
    }
    r  = sqrt( r );
    rx = sqrt( rx );
    if ( dot < 0.0 )
        rx = -rx;
    // double rh[n];
    double rh[NDIM_MAX];
    for ( int i = 0; i < n; i++ )
        rh[i] = ( x1[i] - x0[i] ) / r;
    double f = 0.0;
    if ( rx <= r && rx >= 0.0 ) {
        // Get the gradient along the line at the endpoints
        double df0 = 0.0;
        double df1 = 0.0;
        for ( int i = 0; i < n; i++ ) {
            df0 += rh[i] * g0[i];
            df1 += rh[i] * g1[i];
        }
        // Get the equation of the line( f(x) = a0+a1*x+a2*x^2+a3*x^3 )
        double a[4];
        a[0] = f0;
        a[1] = df0;
        a[2] = 1.0 / ( r * r ) * ( 3.0 * ( f1 - f0 ) - r * ( 2.0 * df0 + df1 ) );
        a[3] = 1.0 / ( r * r * r ) * ( 2.0 * ( f0 - f1 ) + r * ( df0 + df1 ) );
        // Compute f(x) along the line
        f = a[0] + a[1] * rx + a[2] * rx * rx + a[3] * rx * rx * rx;
        // Compute the gradient
        if ( g != nullptr ) {
#if 1
            // Use linear interpolation for the component perpendicular to the line,
            // and the previously computed component for the direction parallel to the line
            double b = rx / r;
            double df_dr =
                a[1] + 2.0 * a[2] * rx + 3.0 * a[3] * rx * rx; // derivative of f at x along r
            for ( int i = 0; i < n; i++ ) {
                double gp0 = g0[i] - rh[i] * df0;
                double gp1 = g1[i] - rh[i] * df1;
                double dg_drp =
                    gp0 +
                    b * ( gp1 - gp0 ); // derivative of f at x perpendicular to r (ith component)
                g[i] = dg_drp + df_dr * rh[i]; // ith component of the gradient at x
            }
#else
            // Use linear interpolation for the gradient
            double b = rx / r;
            for ( int i = 0; i < n; i++ )
                g[i] = ( 1 - b ) * g0[i] + b * g1[i];
#endif
        }
    } else {
        // Perform quadratic interpolation from the closer point using
        //    a linear approximation for the gradient
        double g2[NDIM_MAX];
        // Compute the gradient
        double b = rx / r;
        for ( int i = 0; i < n; i++ )
            g2[i] = g0[i] + b * ( g1[i] - g0[i] );
        if ( g != nullptr ) {
            for ( int i = 0; i < n; i++ )
                g[i] = g2[i];
        }
        // Perform the interpolation
        if ( rx > 0.0 ) {
            f = f1;
            for ( int i = 0; i < n; i++ )
                f += 0.5 * ( g1[i] + g2[i] ) * ( x[i] - x1[i] );
        } else {
            f = f0;
            for ( int i = 0; i < n; i++ )
                f += 0.5 * ( g0[i] + g2[i] ) * ( x[i] - x0[i] );
        }
    }
    return f;
}


/********************************************************************
 * Function to get a list of the nodes that connect to each node     *
 ********************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_node_neighbors() const
{
    // Check to see if we already created the structure
    if ( d_N_node != nullptr )
        return;
    PROFILE_START( "create_node_neighbors", PROFILE_LEVEL );
    // Allocate the data
    d_N_node           = new unsigned int[d_N];
    auto node_list_tmp = new unsigned int *[d_N];
    node_list_tmp[0]   = new unsigned int[2 * d_ndim * ( d_ndim + 1 ) * d_N_tri];
    // Count the number of nodes that are connected to any other node
    const int Nd = d_ndim + 1;
    for ( unsigned int i = 0; i < d_N; i++ )
        d_N_node[i] = 0;
    for ( size_t i = 0; i < d_N_tri * ( d_ndim + 1 ); i++ ) {
        d_N_node[d_tri[i]] += d_ndim;
    }
    // Break the node list array into sub arrays to store the neighbor nodes
    for ( unsigned int i = 1; i < d_N; i++ )
        node_list_tmp[i] = &node_list_tmp[i - 1][d_N_node[i - 1]];
    // For each triangle, add the neighbor nodes
    for ( unsigned int i = 0; i < d_N; i++ )
        d_N_node[i] = 0;
    for ( unsigned int i = 0; i < d_N_tri; i++ ) {
        for ( int j = 0; j <= d_ndim; j++ ) {
            int j1 = d_tri[j + i * Nd];
            int j2 = d_N_node[j1];
            for ( int k = 0; k <= d_ndim; k++ ) {
                if ( j == k )
                    continue;
                node_list_tmp[j1][j2] = d_tri[k + i * Nd];
                j2++;
            }
            d_N_node[j1] += d_ndim;
        }
    }
    // Eliminate duplicate entries in the node list and sort the list
    for ( unsigned int i = 0; i < d_N; i++ ) {
        quicksort1( d_N_node[i], node_list_tmp[i] );
        int k = 0;
        for ( unsigned int j = 1; j < d_N_node[i]; j++ ) {
            if ( node_list_tmp[i][j] != node_list_tmp[i][k] ) {
                node_list_tmp[i][k + 1] = node_list_tmp[i][j];
                k++;
            }
        }
        d_N_node[i] = k + 1;
    }
    // Create the final list that contains storage only for the needed values
    d_N_node_sum = 0;
    for ( unsigned int i = 0; i < d_N; i++ )
        d_N_node_sum += d_N_node[i];
    d_node_list    = new unsigned int *[d_N];
    d_node_list[0] = new unsigned int[d_N_node_sum];
    for ( unsigned int i = 0; i < d_N_node_sum; i++ )
        d_node_list[0][i] = static_cast<unsigned int>( -1 );
    for ( unsigned int i = 1; i < d_N; i++ )
        d_node_list[i] = &d_node_list[i - 1][d_N_node[i - 1]];
    for ( unsigned int i = 0; i < d_N; i++ ) {
        for ( unsigned int j = 0; j < d_N_node[i]; j++ )
            d_node_list[i][j] = node_list_tmp[i][j];
    }
    // Delete the temporary memory
    delete[] node_list_tmp[0];
    delete[] node_list_tmp;
    PROFILE_STOP( "create_node_neighbors", PROFILE_LEVEL );
}


/**************************************************************************
 * Function to get a list of the triangles that neighbors to each triangle *
 * Note:  This function relies on tri_list being in sorted order for       *
 * proper operation.
 **************************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_tri_neighbors() const
{
    // Check to see if we already created the structure
    if ( d_tri_nab != nullptr )
        return;
    // 1D is a special easy case
    if ( d_ndim == 1 ) {
        d_tri_nab = new int[2 * d_N_tri];
        for ( size_t i = 0; i < d_N_tri; i++ ) {
            d_tri_nab[2 * i + 0] = static_cast<int>( i + 1 );
            d_tri_nab[2 * i + 1] = static_cast<int>( i - 1 );
        }
        d_tri_nab[1]                   = -1;
        d_tri_nab[2 * ( d_N_tri - 1 )] = -1;
        return;
    }
    // Allocate memory
    const unsigned char Nd = d_ndim + 1;
    auto N_tri_nab         = new unsigned int[d_N];   // Number of triangles connected each node (N)
    auto tri_list          = new unsigned int *[d_N]; // List of triangles connected each node (N)
    tri_list[0]            = new unsigned int[( d_ndim + 1 ) * d_N_tri];
    d_tri_nab              = new int[( d_ndim + 1 ) * d_N_tri];
    if ( d_N_tri == 1 ) {
        for ( int i = 0; i <= d_ndim; i++ )
            d_tri_nab[i] = -1;
        delete[] N_tri_nab;
        delete[] tri_list[0];
        delete[] tri_list;
        return;
    }
    PROFILE_START( "create_tri_neighbors", PROFILE_LEVEL );
    // For each node, get a list of the triangles that connect to that node
    // Count the number of triangles connected to each vertex
    for ( size_t i = 0; i < d_N; i++ )
        N_tri_nab[i] = 0;
    for ( size_t i = 0; i < Nd * d_N_tri; i++ )
        N_tri_nab[d_tri[i]]++;
    for ( size_t i = 1; i < d_N; i++ )
        tri_list[i] = &tri_list[i - 1][N_tri_nab[i - 1]];
    for ( size_t i = 0; i < Nd * d_N_tri; i++ )
        tri_list[0][i] = static_cast<unsigned int>( -1 );
    // Create a sorted list of all triangles that have each node as a vertex
    for ( size_t i = 0; i < d_N; i++ )
        N_tri_nab[i] = 0;
    for ( size_t i = 0; i < d_N_tri; i++ ) {
        for ( size_t j = 0; j < Nd; j++ ) {
            int k                     = d_tri[j + i * Nd];
            tri_list[k][N_tri_nab[k]] = static_cast<unsigned int>( i );
            N_tri_nab[k]++;
        }
    }
    for ( size_t i = 0; i < d_N; i++ ) {
        quicksort1( N_tri_nab[i], tri_list[i] );
    }
    unsigned int N_tri_max = 0;
    for ( size_t i = 0; i < d_N; i++ ) {
        if ( N_tri_nab[i] > N_tri_max )
            N_tri_max = N_tri_nab[i];
    }
    // Initialize tri_neighbor
    for ( size_t i = 0; i < Nd * d_N_tri; i++ )
        d_tri_nab[i] = -1;
    // Note, if a triangle is a neighbor, it will share all but the current node
    int size[NDIM_MAX];
    int error = 0;
    for ( unsigned int i = 0; i < d_N_tri; i++ ) {
        // Loop through the different faces of the triangle
        for ( int j = 0; j < Nd; j++ ) {
            unsigned int *list[NDIM_MAX] = { nullptr };
            int k1                       = 0;
            for ( int k2 = 0; k2 < Nd; k2++ ) {
                if ( k2 == j )
                    continue;
                int k    = d_tri[k2 + i * Nd];
                list[k1] = tri_list[k];
                size[k1] = N_tri_nab[k];
                k1++;
            }
            // Find the intersection of all triangle lists except the current node
            const auto neg_1             = static_cast<unsigned int>( -1 );
            unsigned int intersection[5] = { neg_1, neg_1, neg_1, neg_1, neg_1 };
            int N_int                    = intersect_sorted( d_ndim, size, list, 5, intersection );
            unsigned int m               = 0;
            if ( N_int == 0 || N_int > 2 ) {
                // We cannot have less than 1 triangle or more than 2 triangles sharing d_ndim nodes
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
            d_tri_nab[j + i * Nd] = m;
        }
        if ( error != 0 )
            break;
    }
    // Check tri_nab
    for ( size_t i = 0; i < d_N_tri; i++ ) {
        for ( int d = 0; d < Nd; d++ ) {
            if ( d_tri_nab[d + i * Nd] < -1 || d_tri_nab[d + i * Nd] >= ( (int) d_N_tri ) ||
                 d_tri_nab[d + i * Nd] == ( (int) i ) )
                error = 2;
        }
    }
    delete[] N_tri_nab;
    delete[] tri_list[0];
    delete[] tri_list;
    if ( error == 1 ) {
        throw std::logic_error( "Error in create_tri_neighbors detected" );
    } else if ( error == 2 ) {
        throw std::logic_error( "Internal error" );
    }
    PROFILE_STOP( "create_tri_neighbors", PROFILE_LEVEL );
}


/****************************************************************
 * Function to compute the starting triangle for each node       *
 * Note: since each node is likely a member of many triangles    *
 *   it doesn't matter which one we use                          *
 ****************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_node_tri() const
{
    // Check to see if we already created the structure
    if ( d_node_tri != nullptr )
        return;
    d_node_tri = new int[d_N];
    memset( d_node_tri, 0, d_N * sizeof( int ) );
    for ( size_t i = 0; i < d_N_tri; i++ ) {
        for ( size_t j = 0; j <= d_ndim; j++ )
            d_node_tri[d_tri[j + i * ( d_ndim + 1 )]] = static_cast<int>( i );
    }
}


/****************************************************************
 * Function to compute the Barycentric coordinates               *
 ****************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::compute_Barycentric( const int ndim,
                                                       const double *x,
                                                       const double *xi,
                                                       double *L )
{
    // Compute the barycentric coordinates T*L=r-r0
    // http://en.wikipedia.org/wiki/Barycentric_coordinate_system_(mathematics)
    double T[NDIM_MAX * NDIM_MAX];
    for ( int i = 0; i < ndim * ndim; i++ )
        T[i] = 0.0;
    for ( int i = 0; i < ndim; i++ ) {
        for ( int j = 0; j < ndim; j++ ) {
            T[j + i * ndim] = x[j + i * ndim] - x[j + ndim * ndim];
        }
    }
    // double r[ndim];
    double r[NDIM_MAX];
    for ( int i = 0; i < ndim; i++ )
        r[i] = xi[i] - x[i + ndim * ndim];
    solve_system( ndim, T, r, L );
    L[ndim] = 1.0;
    for ( int i = 0; i < ndim; i++ )
        L[ndim] -= L[i];
}


/****************************************************************
 * Function to compute the gradient from 3/4 points in 2D/3D     *
 ****************************************************************/
inline void compute_gradient_1d( const double *x, const double *f, double *g )
{
    g[0] = ( f[1] - f[0] ) / ( x[1] - x[0] );
}
inline void compute_gradient_2d( const double *x, const double *f, double *g )
{
    double M[9], b[3], y[3], det;
    M[0] = 1;
    M[3] = x[0];
    M[6] = x[1];
    b[0] = f[0];
    M[1] = 1;
    M[4] = x[2];
    M[7] = x[3];
    b[1] = f[1];
    M[2] = 1;
    M[5] = x[4];
    M[8] = x[5];
    b[2] = f[2];
    DelaunayHelpers<3>::solve_system( M, b, y, det );
    g[0] = y[1] / det;
    g[1] = y[2] / det;
}
inline void compute_gradient_3d( const double *x, const double *f, double *g )
{
    double M[16], b[4], y[4], det;
    M[0]  = 1;
    M[4]  = x[0];
    M[8]  = x[1];
    M[12] = x[2];
    b[0]  = f[0];
    M[1]  = 1;
    M[5]  = x[3];
    M[9]  = x[4];
    M[13] = x[5];
    b[1]  = f[1];
    M[2]  = 1;
    M[6]  = x[6];
    M[10] = x[7];
    M[14] = x[8];
    b[2]  = f[2];
    M[3]  = 1;
    M[7]  = x[9];
    M[11] = x[10];
    M[15] = x[11];
    b[3]  = f[3];
    DelaunayHelpers<4>::solve_system( M, b, y, det );
    g[0] = y[1] / det;
    g[1] = y[2] / det;
    g[2] = y[3] / det;
}
static void compute_gradient( const int ndim, const double *x, const double *f, double *g )
{
    if ( ndim == 1 )
        compute_gradient_1d( x, f, g );
    else if ( ndim == 2 )
        compute_gradient_2d( x, f, g );
    else if ( ndim == 3 )
        compute_gradient_3d( x, f, g );
}


/**************************************************************************
 * Subroutine to perform a quicksort                                       *
 **************************************************************************/
template<class type_a>
static void quicksort1( int n, type_a *arr )
{
    bool test;
    int i, ir, j, jstack, k, l, istack[100];
    type_a a, tmp_a;
    jstack = 0;
    l      = 0;
    ir     = n - 1;
    while ( true ) {
        if ( ir - l < 7 ) { // Insertion sort when subarray small enough.
            for ( j = l + 1; j <= ir; j++ ) {
                a    = arr[j];
                test = true;
                for ( i = j - 1; i >= 0; i-- ) {
                    if ( arr[i] < a ) {
                        arr[i + 1] = a;
                        test       = false;
                        break;
                    }
                    arr[i + 1] = arr[i];
                }
                if ( test ) {
                    i          = l - 1;
                    arr[i + 1] = a;
                }
            }
            if ( jstack == 0 )
                return;
            ir = istack[jstack]; // Pop stack and begin a new round of partitioning.
            l  = istack[jstack - 1];
            jstack -= 2;

        } else {
            k = ( l + ir ) / 2; // Choose median of left, center and right elements as partitioning
                                // element a. Also rearrange so that a(l) ? a(l+1) ? a(ir).
            tmp_a      = arr[k];
            arr[k]     = arr[l + 1];
            arr[l + 1] = tmp_a;
            if ( arr[l] > arr[ir] ) {
                tmp_a   = arr[l];
                arr[l]  = arr[ir];
                arr[ir] = tmp_a;
            }
            if ( arr[l + 1] > arr[ir] ) {
                tmp_a      = arr[l + 1];
                arr[l + 1] = arr[ir];
                arr[ir]    = tmp_a;
            }
            if ( arr[l] > arr[l + 1] ) {
                tmp_a      = arr[l];
                arr[l]     = arr[l + 1];
                arr[l + 1] = tmp_a;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l + 1]; // Partitioning element.
            for ( i = l + 2; i <= ir; i++ ) {
                if ( arr[i] < a )
                    continue;
                while ( arr[j] > a ) // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;       // Pointers crossed. Exit with partitioning complete.
                tmp_a  = arr[i]; // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
            }
            arr[l + 1] = arr[j]; // Insert partitioning element in both arrays.
            arr[j]     = a;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir - i + 1 >= j - l ) {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            } else {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}
template<class type_a, class type_b>
void quicksort2( int n, type_a *arr, type_b *brr )
{
    bool test;
    int i, ir, j, jstack, k, l, istack[100];
    type_a a, tmp_a;
    type_b b, tmp_b;
    jstack = 0;
    l      = 0;
    ir     = n - 1;
    while ( true ) {
        if ( ir - l < 9 ) { // Insertion sort when subarray small enough.
            for ( j = l + 1; j <= ir; j++ ) {
                a    = arr[j];
                b    = brr[j];
                test = true;
                for ( i = j - 1; i >= 0; i-- ) {
                    if ( arr[i] < a ) {
                        arr[i + 1] = a;
                        brr[i + 1] = b;
                        test       = false;
                        break;
                    }
                    arr[i + 1] = arr[i];
                    brr[i + 1] = brr[i];
                }
                if ( test ) {
                    i          = l - 1;
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                }
            }
            if ( jstack == 0 )
                return;
            ir = istack[jstack]; // Pop stack and begin a new round of partitioning.
            l  = istack[jstack - 1];
            jstack -= 2;
        } else {
            k = ( l + ir ) / 2; // Choose median of left, center and right elements as partitioning
                                // element a. Also rearrange so that a(l) ? a(l+1) ? a(ir).
            tmp_a      = arr[k];
            arr[k]     = arr[l + 1];
            arr[l + 1] = tmp_a;
            tmp_b      = brr[k];
            brr[k]     = brr[l + 1];
            brr[l + 1] = tmp_b;
            if ( arr[l] > arr[ir] ) {
                tmp_a   = arr[l];
                arr[l]  = arr[ir];
                arr[ir] = tmp_a;
                tmp_b   = brr[l];
                brr[l]  = brr[ir];
                brr[ir] = tmp_b;
            }
            if ( arr[l + 1] > arr[ir] ) {
                tmp_a      = arr[l + 1];
                arr[l + 1] = arr[ir];
                arr[ir]    = tmp_a;
                tmp_b      = brr[l + 1];
                brr[l + 1] = brr[ir];
                brr[ir]    = tmp_b;
            }
            if ( arr[l] > arr[l + 1] ) {
                tmp_a      = arr[l];
                arr[l]     = arr[l + 1];
                arr[l + 1] = tmp_a;
                tmp_b      = brr[l];
                brr[l]     = brr[l + 1];
                brr[l + 1] = tmp_b;
            }
            // Scan up to find element > a
            j = ir;
            a = arr[l + 1]; // Partitioning element.
            b = brr[l + 1];
            for ( i = l + 2; i <= ir; i++ ) {
                if ( arr[i] < a )
                    continue;
                while ( arr[j] > a ) // Scan down to find element < a.
                    j--;
                if ( j < i )
                    break;       // Pointers crossed. Exit with partitioning complete.
                tmp_a  = arr[i]; // Exchange elements of both arrays.
                arr[i] = arr[j];
                arr[j] = tmp_a;
                tmp_b  = brr[i];
                brr[i] = brr[j];
                brr[j] = tmp_b;
            }
            arr[l + 1] = arr[j]; // Insert partitioning element in both arrays.
            arr[j]     = a;
            brr[l + 1] = brr[j];
            brr[j]     = b;
            jstack += 2;
            // Push pointers to larger subarray on stack, process smaller subarray immediately.
            if ( ir - i + 1 >= j - l ) {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            } else {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }
}


/****************************************************************
 * Function to solve the system Mx=b                             *
 ****************************************************************/
static void solve_system( const int N, const double *M, const double *b, double *x )
{
    if ( N == 1 ) {
        // 1x1 matrix is trivial
        x[0] = b[0] / M[0];
    } else if ( N == 2 ) {
        // 2x2 matrix has a simple inverse
        double inv_det = 1.0 / ( M[0] * M[3] - M[1] * M[2] );
        x[0]           = ( M[3] * b[0] - M[2] * b[1] ) * inv_det;
        x[1]           = ( M[0] * b[1] - M[1] * b[0] ) * inv_det;
    } else if ( N == 3 ) {
        // 3x3 matrix
        double M_inv[9];
        M_inv[0]       = M[4] * M[8] - M[7] * M[5];
        M_inv[1]       = M[7] * M[2] - M[1] * M[8];
        M_inv[2]       = M[1] * M[5] - M[4] * M[2];
        M_inv[3]       = M[6] * M[5] - M[3] * M[8];
        M_inv[4]       = M[0] * M[8] - M[6] * M[2];
        M_inv[5]       = M[3] * M[2] - M[0] * M[5];
        M_inv[6]       = M[3] * M[7] - M[6] * M[4];
        M_inv[7]       = M[6] * M[1] - M[0] * M[7];
        M_inv[8]       = M[0] * M[4] - M[3] * M[1];
        double inv_det = 1.0 / ( M[0] * M_inv[0] + M[3] * M_inv[1] + M[6] * M_inv[2] );
        x[0]           = M_inv[0] * b[0] + M_inv[3] * b[1] + M_inv[6] * b[2];
        x[1]           = M_inv[1] * b[0] + M_inv[4] * b[1] + M_inv[7] * b[2];
        x[2]           = M_inv[2] * b[0] + M_inv[5] * b[1] + M_inv[8] * b[2];
        x[0] *= inv_det;
        x[1] *= inv_det;
        x[2] *= inv_det;
    } else {
#if USE_LAPACK == 0
        // No method to solve the system
        throw std::logic_error( "Need to link LAPACK" );
#else
        // Call Lapack to compute the inverse
        int error;
        int *IPIV;
        double *M2;
        double tmp1[64]; // Use the stack for small matricies (n<=8)
        int tmp2[8];     // Use the stack for small matricies (n<=8)
        if ( N <= 20 ) {
            M2 = tmp1;
            IPIV = tmp2;
        } else {
            M2 = new double[N * N];
            IPIV = new int[N];
        }
        for ( int i = 0; i < N * N; i++ )
            M2[i] = M[i];
        for ( int i = 0; i < N; i++ )
            x[i] = b[i];
        Lapack<double>::gesv( N, 1, M2, N, IPIV, x, N, error );
        if ( M2 != tmp1 ) {
            delete[] M2;
            delete[] IPIV;
        }
#endif
    }
}


/****************************************************************
 * Function to calculate the inverse of a matrix                 *
 ****************************************************************/
static void inv_M( const int N, const double *M, double *M_inv )
{
    if ( N == 1 ) {
        // 1x1 matrix is trivial
        M_inv[0] = 1.0 / M[0];
    } else if ( N == 2 ) {
        // 2x2 matrix has a simple inverse
        double inv_det = 1.0 / ( M[0] * M[3] - M[1] * M[2] );
        M_inv[0]       = M[3] * inv_det;
        M_inv[1]       = -M[1] * inv_det;
        M_inv[2]       = -M[2] * inv_det;
        M_inv[3]       = M[0] * inv_det;
    } else if ( N == 3 ) {
        // 3x3 matrix
        M_inv[0]       = M[4] * M[8] - M[7] * M[5];
        M_inv[1]       = M[7] * M[2] - M[1] * M[8];
        M_inv[2]       = M[1] * M[5] - M[4] * M[2];
        M_inv[3]       = M[6] * M[5] - M[3] * M[8];
        M_inv[4]       = M[0] * M[8] - M[6] * M[2];
        M_inv[5]       = M[3] * M[2] - M[0] * M[5];
        M_inv[6]       = M[3] * M[7] - M[6] * M[4];
        M_inv[7]       = M[6] * M[1] - M[0] * M[7];
        M_inv[8]       = M[0] * M[4] - M[3] * M[1];
        double inv_det = 1.0 / ( M[0] * M_inv[0] + M[3] * M_inv[1] + M[6] * M_inv[2] );
        for ( int i = 0; i < 9; i++ )
            M_inv[i] *= inv_det;
    } else {
#if USE_LAPACK == 0
        // No method to compute the inverse, return all zeros
        AMP::perr << "Need to link LAPACK\n";
        for ( int i = 0; i < n * n; i++ )
            M_inv[i] = M[i];
#else
        // Call Lapack to compute the inverse
        int error;
        int LWORK;
        int *IPIV;
        double *WORK;
        double tmp1[64 * 8]; // Use the stack for small matricies (N<=8)
        int tmp2[8];         // Use the stack for small matricies (N<=8)
        if ( N <= 8 ) {
            LWORK = 64 * 8;
            WORK = tmp1;
            IPIV = tmp2;
        } else {
            LWORK = 64 * N;
            WORK = new double[LWORK];
            IPIV = new int[N];
        }
        for ( int i = 0; i < N * N; i++ )
            M_inv[i] = M[i];
        Lapack<double>::getrf( N, N, M_inv, N, IPIV, error );
        Lapack<double>::getri( N, M_inv, N, IPIV, WORK, LWORK, error );
        if ( WORK != tmp1 ) {
            delete[] IPIV;
            delete[] WORK;
        }
#endif
    }
}


/********************************************************************
 * Function to perform block Gauss-Seidel iteration                  *
 * Note: the performance of this algorithum is strongly dependent on *
 * the memory storage.  For example even passing the row and column  *
 * index instead of N_row and icol increased runtime by ~100x.       *
 ********************************************************************/
static void Gauss_Seidel( const unsigned int Nb,
                          const unsigned int N,
                          const unsigned int,
                          const double D[],
                          const unsigned int N_row[],
                          unsigned int *icol[],
                          const double A[],
                          const double rhs[],
                          double *x,
                          const int N_it )
{
    // First we will compute the inverse of each block
    auto D_inv = new double[Nb * Nb * N];
    for ( unsigned int i = 0; i < N; i++ )
        inv_M( Nb, &D[i * Nb * Nb], &D_inv[i * Nb * Nb] );
    // Next perform the Gauss-Seidel iterations:
    //    x(k+1) = aii^-1*(bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i))
    const double rel_tol = 1e-8;
    const double abs_tol = 1e-12;
    if ( Nb == 2 ) {
        double tmp[2], x_new[2], x_old[2];
        for ( int it = 0; it < N_it; it++ ) {
            int m           = 0;
            double L2_norm  = 0.0;
            double L2_error = 0.0;
            for ( unsigned int i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                tmp[0] = rhs[0 + i * 2];
                tmp[1] = rhs[1 + i * 2];
                for ( unsigned int j = 0; j < N_row[i]; j++ ) {
                    unsigned int k = icol[i][j];
                    tmp[0] -= ( A[0 + m * 4] * x[0 + k * 2] + A[2 + m * 4] * x[1 + k * 2] );
                    tmp[1] -= ( A[1 + m * 4] * x[0 + k * 2] + A[3 + m * 4] * x[1 + k * 2] );
                    m++;
                }
                // Update x(:,i)
                x_new[0]     = D_inv[0 + i * 4] * tmp[0] + D_inv[2 + i * 4] * tmp[1];
                x_new[1]     = D_inv[1 + i * 4] * tmp[0] + D_inv[3 + i * 4] * tmp[1];
                x_old[0]     = x[0 + i * 2];
                x_old[1]     = x[1 + i * 2];
                x[0 + i * 2] = x_new[0];
                x[1 + i * 2] = x_new[1];
                L2_norm += x_new[0] * x_new[0] + x_new[1] * x_new[1];
                L2_error += ( x_old[0] - x_new[0] ) * ( x_old[0] - x_new[0] ) +
                            ( x_old[1] - x_new[1] ) * ( x_old[1] - x_new[1] );
            }
            // Check the quality of the new solution
            L2_norm  = sqrt( L2_norm );
            L2_error = sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
    } else if ( Nb == 3 ) {
        double tmp[3], x_new[3], x_old[3];
        for ( int it = 0; it < N_it; it++ ) {
            int m           = 0;
            double L2_norm  = 0.0;
            double L2_error = 0.0;
            for ( unsigned int i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                tmp[0] = rhs[0 + i * 3];
                tmp[1] = rhs[1 + i * 3];
                tmp[2] = rhs[2 + i * 3];
                for ( unsigned int j = 0; j < N_row[i]; j++ ) {
                    unsigned int k   = icol[i][j];
                    const double *A2 = &A[m * 9];
                    double *x2       = &x[k * 3];
                    tmp[0] -= ( A2[0] * x2[0] + A2[3] * x2[1] + A2[6] * x2[2] );
                    tmp[1] -= ( A2[1] * x2[0] + A2[4] * x2[1] + A2[7] * x2[2] );
                    tmp[2] -= ( A2[2] * x2[0] + A2[5] * x2[1] + A2[8] * x2[2] );
                    m++;
                }
                // Update x(:,i)
                const double *D_inv2 = &D_inv[i * 9];
                x_new[0]             = D_inv2[0] * tmp[0] + D_inv2[3] * tmp[1] + D_inv2[6] * tmp[2];
                x_new[1]             = D_inv2[1] * tmp[0] + D_inv2[4] * tmp[1] + D_inv2[7] * tmp[2];
                x_new[2]             = D_inv2[2] * tmp[0] + D_inv2[5] * tmp[1] + D_inv2[8] * tmp[2];
                x_old[0]             = x[0 + i * 3];
                x_old[1]             = x[1 + i * 3];
                x_old[2]             = x[2 + i * 3];
                x[0 + i * 3]         = x_new[0];
                x[1 + i * 3]         = x_new[1];
                x[2 + i * 3]         = x_new[2];
                L2_norm += x_new[0] * x_new[0] + x_new[1] * x_new[1] + x_new[2] * x_new[2];
                L2_error += ( x_old[0] - x_new[0] ) * ( x_old[0] - x_new[0] ) +
                            ( x_old[1] - x_new[1] ) * ( x_old[1] - x_new[1] ) +
                            ( x_old[2] - x_new[2] ) * ( x_old[2] - x_new[2] );
            }
            // Check the quality of the new solution
            L2_norm  = sqrt( L2_norm );
            L2_error = sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
    } else {
        auto tmp = new double[Nb];
        for ( int it = 0; it < N_it; it++ ) {
            int m           = 0;
            double L2_norm  = 0.0;
            double L2_error = 0.0;
            for ( unsigned int i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                for ( unsigned int j = 0; j < Nb; j++ )
                    tmp[j] = rhs[j + i * Nb];
                for ( unsigned int j = 0; j < N_row[i]; j++ ) {
                    unsigned int k = icol[i][j];
                    for ( unsigned int j1 = 0; j1 < Nb; j1++ ) {
                        for ( unsigned int j2 = 0; j2 < Nb; j2++ ) {
                            tmp[j1] -= A[j1 + j2 * Nb + m * Nb * Nb] * x[j2 + k * Nb];
                        }
                    }
                    m++;
                }
                // Update x(:,i)
                for ( unsigned int j1 = 0; j1 < Nb; j1++ ) {
                    double x_new = 0.0;
                    for ( unsigned int j2 = 0; j2 < Nb; j2++ ) {
                        x_new += D_inv[j1 + j2 * Nb + i * Nb * Nb] * tmp[j2];
                    }
                    double x_old   = x[j1 + i * Nb];
                    x[j1 + i * Nb] = x_new;
                    L2_norm += x_new * x_new;
                    L2_error += ( x_old - x_new ) * ( x_old - x_new );
                }
            }
            // Check the quality of the new solution
            L2_norm  = sqrt( L2_norm );
            L2_error = sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
        delete[] tmp;
    }
    // Delete temporary memory
    delete[] D_inv;
}


// Subroutine to find the first n intersections in multiple lists
// This function assumes the lists are in sorted order
static int intersect_sorted( const int N_lists,
                             const int size[],
                             unsigned int *list[],
                             const int N_max,
                             unsigned int *intersection )
{
    if ( N_max <= 0 )
        return ~( (unsigned int) 0 );
    int N_int  = 0;
    auto index = new int[N_lists];
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
    delete[] index;
    return N_int;
}


// Explicit instantiations
template class DelaunayInterpolation<int>;
template class DelaunayInterpolation<double>;

template void DelaunayInterpolation<int>::find_nearest<int>( const unsigned int,
                                                             const int[],
                                                             const int,
                                                             unsigned int * );
template void DelaunayInterpolation<int>::find_tri<int>(
    const unsigned int, const int[], const int, int *, bool );
template void DelaunayInterpolation<int>::interp_nearest<int>(
    const double[], const unsigned int, const int[], const unsigned int[], double * );
template void DelaunayInterpolation<int>::interp_linear<int>(
    const double[], const unsigned int, const int[], const int[], double *, double *, bool );
template void DelaunayInterpolation<int>::interp_cubic<int>( const double[],
                                                             const double[],
                                                             const unsigned int,
                                                             const int[],
                                                             const int[],
                                                             double *,
                                                             double *,
                                                             int );

template void DelaunayInterpolation<int>::find_nearest<double>( const unsigned int,
                                                                const double[],
                                                                const int,
                                                                unsigned int * );
template void DelaunayInterpolation<int>::find_tri<double>(
    const unsigned int, const double[], const int, int *, bool );
template void DelaunayInterpolation<int>::interp_nearest<double>(
    const double[], const unsigned int, const double[], const unsigned int[], double * );
template void DelaunayInterpolation<int>::interp_linear<double>(
    const double[], const unsigned int, const double[], const int[], double *, double *, bool );
template void DelaunayInterpolation<int>::interp_cubic<double>( const double[],
                                                                const double[],
                                                                const unsigned int,
                                                                const double[],
                                                                const int[],
                                                                double *,
                                                                double *,
                                                                int );

template void DelaunayInterpolation<double>::find_nearest<double>( const unsigned int,
                                                                   const double[],
                                                                   const int,
                                                                   unsigned int * );
template void DelaunayInterpolation<double>::find_tri<double>(
    const unsigned int, const double[], const int, int *, bool );
template void DelaunayInterpolation<double>::interp_nearest<double>(
    const double[], const unsigned int, const double[], const unsigned int[], double * );
template void DelaunayInterpolation<double>::interp_linear<double>(
    const double[], const unsigned int, const double[], const int[], double *, double *, bool );
template void DelaunayInterpolation<double>::interp_cubic<double>( const double[],
                                                                   const double[],
                                                                   const unsigned int,
                                                                   const double[],
                                                                   const int[],
                                                                   double *,
                                                                   double *,
                                                                   int );