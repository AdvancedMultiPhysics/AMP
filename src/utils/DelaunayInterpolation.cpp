#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "AMP/IO/PIO.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/DelaunayInterpolation.h"
#include "AMP/utils/DelaunayTessellation.h"
#include "AMP/utils/Utilities.hpp"

#include "ProfilerApp.h"

#define NDIM_MAX 3
#define ProfileLevel 3


namespace AMP {


static void Gauss_Seidel( const uint32_t Nb,
                          const uint32_t N,
                          const uint32_t M,
                          const double D[],
                          const uint32_t N_row[],
                          uint32_t *icol[],
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


static constexpr double TRI_TOL = 1e-10;


/********************************************************************
 * Primary constructor                                               *
 ********************************************************************/
template<class TYPE>
DelaunayInterpolation<TYPE>::DelaunayInterpolation()
{
    d_N_node_sum = 0;
    d_N_node     = nullptr;
    d_node_list  = nullptr;
    d_node_tri   = nullptr;
    d_tree       = nullptr;
}


/********************************************************************
 * De-constructor                                                    *
 ********************************************************************/
template<class TYPE>
DelaunayInterpolation<TYPE>::~DelaunayInterpolation()
{
    clear();
}
template<class TYPE>
void DelaunayInterpolation<TYPE>::clear()
{
    d_x.clear();
    d_tri.clear();
    d_tri_nab.clear();
    if ( d_N_node != nullptr ) {
        delete[] d_N_node;
        delete[] d_node_list[0];
        delete[] d_node_list;
        d_N_node_sum = 0;
        d_N_node     = nullptr;
        d_node_list  = nullptr;
    }
    delete[] d_node_tri;
    d_node_tri = nullptr;
    delete d_tree;
    d_tree = nullptr;
}


/********************************************************************
 * Function to return the triangles                                  *
 ********************************************************************/
template<class TYPE>
size_t DelaunayInterpolation<TYPE>::get_N_tri() const
{
    if ( d_tri.empty() )
        return 0;
    return d_tri.size( 1 );
}
template<class TYPE>
AMP::Array<int> DelaunayInterpolation<TYPE>::get_tri() const
{
    return d_tri;
}
template<class TYPE>
AMP::Array<int> DelaunayInterpolation<TYPE>::get_tri_nab() const
{
    create_tri_neighbors();
    return d_tri_nab;
}
template<class TYPE>
std::tuple<AMP::Array<TYPE>, AMP::Array<int>> DelaunayInterpolation<TYPE>::copy_tessellation() const
{
    return std::tie( d_x, d_tri );
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
void DelaunayInterpolation<TYPE>::create_tessellation( size_t N,
                                                       const TYPE *x,
                                                       const TYPE *y,
                                                       const TYPE *z )
{
    AMP::Array<TYPE> xyz( 3, N );
    for ( size_t i = 0; i < N; i++ ) {
        xyz( 0, i ) = x[i];
        xyz( 1, i ) = y[i];
        xyz( 2, i ) = z[i];
    }
    create_tessellation( xyz );
}
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_tessellation( const AMP::Array<TYPE> &x )
{
    PROFILE( "create_tessellation", ProfileLevel );
    // Delete the existing data
    clear();
    if ( x.empty() )
        return;
    // Copy the points
    d_x = x;
    // Create the tessellation
    size_t ndim = x.size( 0 );
    size_t N    = x.size( 1 );
    if ( x.size( 0 ) == 1 ) {
        // The triangles are just the sorted points (note the duplicate indices)
        std::vector<TYPE> x_tmp( N );
        std::vector<int> i_tmp( N );
        for ( size_t i = 0; i < N; i++ )
            x_tmp[i] = d_x( i );
        for ( size_t i = 0; i < N; i++ )
            i_tmp[i] = (int) i;
        AMP::Utilities::quicksort( x_tmp, i_tmp );
        int N_tri = N - 1;
        d_tri.resize( 2, N_tri );
        for ( int i = 0; i < N_tri; i++ ) {
            d_tri( 0, i ) = i_tmp[i + 0];
            d_tri( 1, i ) = i_tmp[i + 1];
        }
        create_tri_neighbors();
    } else if ( ndim == 2 || ndim == 3 ) {
        std::tie( d_tri, d_tri_nab ) = DelaunayTessellation::create_tessellation( x );
    } else {
        throw std::logic_error( "Unsupported dimension" );
    }
    AMP_ASSERT( !d_tri.empty() );
    AMP_ASSERT( d_tri.min() >= 0 );
    d_N_node_sum = 0;
}


/********************************************************************
 * Function to construct the tessellation using a given tessellation *
 ********************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_tessellation( const Array<TYPE> &x, const Array<int> &tri )
{
    // Delete the existing data
    clear();
    if ( tri.empty() )
        return;
    // Check the inputs
    AMP_ASSERT( x.size( 0 ) <= 3 );
    AMP_ASSERT( tri.size( 0 ) == x.size( 0 ) + 1 );
    // Copy the data
    d_x   = x;
    d_tri = tri;
}


/************************************************************************
 * This function creates the kdtree                                      *
 ************************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_kdtree() const
{
    // Create the kdtree
    if ( d_tree == nullptr ) {
        PROFILE( "create_kdtree", ProfileLevel );
        double *x[NDIM_MAX] = { nullptr };
        size_t N            = d_x.size( 1 );
        int ndim            = d_x.size( 0 );
        for ( int d = 0; d < ndim; d++ ) {
            x[d] = new double[N];
            for ( size_t i = 0; i < N; i++ )
                x[d][i] = d_x( d, i );
        }
        d_tree = new kdtree( ndim, N, x );
        for ( int d = 0; d < ndim; d++ )
            delete[] x[d];
    }
}


/************************************************************************
 * This function find the nearest point to the desired point             *
 ************************************************************************/
template<class TYPE>
template<class TYPE2>
Array<size_t> DelaunayInterpolation<TYPE>::find_nearest( const Array<TYPE2> &xi ) const
{
    if ( xi.empty() )
        return Array<size_t>();
    AMP_ASSERT( xi.size( 0 ) == d_x.size( 0 ) );
    PROFILE( "find_nearest", ProfileLevel );
    // Create the kdtree
    create_kdtree();
    // Use the kdtree to perform the nearest neighbor interpolation
    Array<size_t> index( xi.size( 1 ) );
    if constexpr ( std::is_same_v<TYPE2, double> ) {
        d_tree->find_nearest( xi.size( 1 ), xi.data(), index.data() );
    } else {
        Array<double> xi2;
        xi2.copy( xi );
        d_tree->find_nearest( xi2.size( 1 ), xi2.data(), index.data() );
    }
    return index;
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
Array<int> DelaunayInterpolation<TYPE>::find_tri( const Array<TYPE2> &xi, bool extrap ) const
{
    if ( xi.empty() )
        return Array<int>();
    AMP_ASSERT( xi.size( 0 ) == d_x.size( 0 ) );
    PROFILE( "find_tri", ProfileLevel );
    Array<int> index( xi.size( 1 ) );
    // Create the kdtree
    create_kdtree();
    // Create a list of the nodes that link to every other node
    create_node_neighbors();
    // For each triangle, get a list of the triangles that are neighbors
    create_tri_neighbors();
    // For each node, get a starting triangle
    create_node_tri();
    // First choose a starting triangle
    auto index_node = find_nearest( xi );
    for ( uint32_t i = 0; i < xi.size( 1 ); i++ )
        index( i ) = d_node_tri[index_node( i )];
    // Loop through the query points
    int ndim         = d_x.size( 0 );
    unsigned char Nd = ndim + 1;
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )], xi2[NDIM_MAX], L[NDIM_MAX + 1];
    bool failed_search = false;
    size_t N_it_tot    = 0;
    for ( uint32_t i = 0; i < xi.size( 1 ); i++ ) {
        for ( int j = 0; j < ndim; j++ )
            xi2[j] = xi( j, i );
        int current_tri = index( i );
        size_t it       = 0;
        while ( true ) {
            // Get the point in Barycentric coordinates
            for ( int j1 = 0; j1 < Nd; j1++ ) {
                int j = d_tri( j1, current_tri );
                for ( int j2 = 0; j2 < ndim; j2++ )
                    x2[j2 + j1 * ndim] = d_x( j2, j );
            }
            compute_Barycentric( ndim, x2, xi2, L );
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
                index( i ) = current_tri;
                break;
            }
            // We are outside an edge of the triangle iff the coordinate for the edge is < 0
            // Check if we are outside the convex hull (we will be outside an edge which is on the
            // convex hull)
            bool outside_hull = false;
            for ( int j = 0; j < Nd; j++ ) {
                if ( L[j] < -TRI_TOL && d_tri_nab( j, current_tri ) == -1 )
                    outside_hull = true;
            }
            if ( outside_hull ) {
                if ( !extrap ) {
                    // We are outside the convex hull, store the error, and continue
                    index( i ) = -1;
                    break;
                }
                // Zero all values of L for the sides on the convex hull
                bool finished = true;
                for ( int j = 0; j < Nd; j++ ) {
                    if ( d_tri_nab( j, current_tri ) == -1 )
                        L[j] = 0.0;
                    else if ( L[j] < -TRI_TOL )
                        finished = false;
                }
                // If no remaining coordiantes are negitive, this is the nearest triangle
                if ( finished ) {
                    index( i ) = current_tri;
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
            int next_tri = d_tri_nab( k, current_tri );
            int N_tri    = d_tri.size( 1 );
            if ( next_tri < 0 || next_tri > N_tri || next_tri == current_tri )
                throw std::logic_error( "Internal error" );
            current_tri = next_tri;
            it++;
            if ( it > (size_t) N_tri + 10 ) {
                // We should never revisit a triangle, so we must have some ended in an infinite
                // cycle
                failed_search = true;
                index( i )    = -2;
                break;
            }
        }
        N_it_tot += it;
    }
    // Check the average number of triangles searched,
    // this should be relatively small since we start with a tirangle that contains the nearest
    // point
    if ( N_it_tot / xi.size( 1 ) > 25 )
        AMP::perr
            << "Search took longer than it should, there may be a problem with the tessellation\n";
    // Check if the search failed for any values
    if ( failed_search )
        AMP::perr << "Search failed for some points\n";
    return index;
}


/****************************************************************
 * Function to get a list of the nodes that connect to each node *
 ****************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::calc_node_gradient( const double *f,
                                                      const int method,
                                                      double *grad,
                                                      const int n_it ) const
{
    PROFILE( "calc_node_gradient", ProfileLevel );
    // First we need to get a list of the nodes that link to every other node
    create_node_neighbors();
    size_t N = d_x.size( 1 );
    int ndim = d_x.size( 0 );
    if ( method == 1 ) {
        // We are performing a local gradient calculation using a least-squares minimization
        // Calculate the derivative for each edge that touches the node, then
        // use a weighted least-squares minimization to get the gradient
        double M[NDIM_MAX * NDIM_MAX], rhs[NDIM_MAX], rh[NDIM_MAX];
        for ( uint32_t i = 0; i < N; i++ ) {
            // Initialize M, rhs
            for ( double &j : M )
                j = 0.0;
            for ( int j = 0; j < NDIM_MAX; j++ ) {
                rhs[j] = 0.0;
                rh[j]  = 0.0;
            }
            // Loop through the neighbor nodes, contructing the matrix and rhs
            for ( uint32_t j = 0; j < d_N_node[i]; j++ ) {
                int k = d_node_list[i][j];
                // Comupute the distance beween the neighbors and the direction vector
                double r = 0.0;
                for ( int n = 0; n < ndim; n++ ) {
                    rh[n] = d_x( n, k ) - d_x( n, i );
                    r += rh[n] * rh[n];
                }
                r = std::sqrt( r );
                for ( int n = 0; n < ndim; n++ )
                    rh[n] /= r;
                // Compute the derivative of f in the direction of the neighbor
                double df_dr = ( f[k] - f[i] ) / r;
                // Get the weights (use the inverse distance)
                double wi = 1.0 / r;
                // Construct the linear system
                for ( int j1 = 0; j1 < ndim; j1++ ) {
                    for ( int j2 = 0; j2 < ndim; j2++ ) {
                        M[j1 + j2 * ndim] += 2.0 * wi * rh[j1] * rh[j2];
                    }
                    rhs[j1] += 2.0 * wi * rh[j1] * df_dr;
                }
            }
            // Solve the linear system to get the local gradient
            DelaunayHelpers::solve( ndim, M, rhs, &grad[i * ndim] );
        }
    } else if ( method == 2 || method == 3 ) {
        /* Both methods 2 and 3 use a higher order approximation for the derivative which
         * requires the function value and the gradient at the neighbor points, and then
         * performs a least squares minimization.  This reduces the error of the gradient,
         * but requires solving a system of size N*d_ndim x N*d_ndim.  The resulting system is
         * sparse.
         * Method 2 solves the sparse system directly, while method 3 uses Gauss-Seidel
         * iteration to improve the solution calculated by method 1.  Note: method 2 in
         * only implemented in the MATLAB prototype since it would require linking to a
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
            AMP::perr << "This method is not implemented\n";
            return;
        }
        // First, allocate storage to store the matrix components
        AMP::Array<double> A( ndim, ndim, d_N_node_sum );
        AMP::Array<double> D( ndim, ndim, N );
        AMP::Array<double> rhs( ndim, N );
        A.fill( 0 );
        D.fill( 0 );
        rhs.fill( 0 );
        // Loop through the nodes, constructing the matrix elements
        int m = 0;
        for ( uint32_t i = 0; i < N; i++ ) {
            // Loop through the neighbor nodes, contructing the matrix and rhs
            for ( uint32_t j = 0; j < d_N_node[i]; j++ ) {
                int k = d_node_list[i][j];
                // Comupute the distance beween the neighbors and the direction vector
                double r = 0.0;
                double rh[NDIM_MAX];
                for ( int n = 0; n < ndim; n++ ) {
                    rh[n] = ( d_x( n, k ) - d_x( n, i ) );
                    r += rh[n] * rh[n];
                }
                r = std::sqrt( r );
                for ( int n = 0; n < ndim; n++ )
                    rh[n] /= r;
                // Compute the derivative of f in the direction of the neighbor
                double df_dr = ( f[k] - f[i] ) / r;
                // Get the weights (use the inverse distance)
                double wi = 1.0 / r;
                // Construct the linear system
                for ( int j1 = 0; j1 < ndim; j1++ ) {
                    for ( int j2 = 0; j2 < ndim; j2++ ) {
                        A( j1, j2, m ) = 2.0 * wi * rh[j1] * rh[j2];
                        D( j1, j2, i ) += A( j1, j2, m );
                    }
                    rhs( j1, i ) += 2.0 * wi * rh[j1] * 2.0 * df_dr;
                }
                m++;
            }
        }
        if ( method == 2 ) {
            // Construct the sparse system and solve it directly
            // NOT implemented
            AMP::perr << "This method is not implemented\n";
        } else if ( method == 3 ) {
            // Use a block Gauss-Seidel method to improve the solution computed by method 1
            // First we need to compute the solution using method 1.  We can do this by noting
            // That D(:,:,i) is the same as method 1, and the rhs is a factor of 2 larger
            // than in method 1
            double rhs2[NDIM_MAX];
            for ( uint32_t i = 0; i < N; i++ ) {
                for ( int j = 0; j < ndim; j++ )
                    rhs2[j] = 0.5 * rhs( j, i );
                DelaunayHelpers::solve( ndim, &D( 0, 0, i ), rhs2, &grad[i * ndim] );
            }
            // Now we can perform a block Gauss-Seidel iteration to improve the solution
            Gauss_Seidel( ndim,
                          (unsigned int) N,
                          (unsigned int) d_N_node_sum,
                          D.data(),
                          d_N_node,
                          d_node_list,
                          A.data(),
                          rhs.data(),
                          grad,
                          n_it );
        } else {
            AMP::perr << "Unknown method\n";
        }
    } else if ( method == 4 ) {
        // This is the same as method 3 (Gauss-Seidel) but does not store the matrix entries,
        //    instead they are re-created every time
        // First, lets get the initial solution using method 1
        calc_node_gradient( f, 1, grad );
        // Loop through the Gauss-Seidel iterations
        double D[NDIM_MAX * NDIM_MAX] = { 0 }, rhs[NDIM_MAX] = { 0 }, Ax[NDIM_MAX] = { 0 },
                            rh[NDIM_MAX] = { 0 };
        for ( int it = 0; it < n_it; it++ ) {
            // Loop through the nodes updating x
            //    x(k+1) = aii^-1*(bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i))
            for ( uint32_t i = 0; i < N; i++ ) {
                for ( int j = 0; j < ndim * ndim; j++ )
                    D[j] = 0.0;
                for ( int j = 0; j < ndim; j++ ) {
                    rhs[j] = 0.0;
                    Ax[j]  = 0.0;
                }
                // Loop through the neighbor nodes, contructing D, A*x and rhs
                for ( uint32_t j = 0; j < d_N_node[i]; j++ ) {
                    int k = d_node_list[i][j];
                    // Comupute the distance beween the neighbors and the direction vector
                    double r = 0.0;
                    for ( int n = 0; n < ndim; n++ ) {
                        rh[n] = d_x( n, k ) - d_x( n, i );
                        r += rh[n] * rh[n];
                    }
                    r = std::sqrt( r );
                    for ( int n = 0; n < ndim; n++ )
                        rh[n] /= r;
                    // Compute the derivative of f in the direction of the neighbor
                    double df_dr = ( f[k] - f[i] ) / r;
                    // Get the weights (use the inverse distance)
                    double wi = 1.0 / r;
                    // Construct the linear system
                    for ( int j1 = 0; j1 < ndim; j1++ ) {
                        for ( int j2 = 0; j2 < ndim; j2++ ) {
                            double tmp = 2.0 * wi * rh[j1] * rh[j2];
                            Ax[j1] += tmp * grad[j2 + k * ndim];
                            D[j1 + j2 * ndim] += tmp;
                        }
                        rhs[j1] += 2.0 * wi * rh[j1] * 2.0 * df_dr;
                    }
                }
                // Update x
                for ( int j = 0; j < ndim; j++ )
                    rhs[j] -= Ax[j];
                DelaunayHelpers::solve( ndim, D, rhs, &grad[i * ndim] );
            }
        }
    } else {
        // Unkown method
        AMP::perr << "Unknown method\n";
    }
}


/********************************************************************
 * Function to perform nearest-neighbor interpolation                *
 ********************************************************************/
template<class TYPE>
template<class TYPE2>
Array<double> DelaunayInterpolation<TYPE>::interp_nearest( const Array<double> &f,
                                                           const Array<TYPE2> &xi,
                                                           const Array<size_t> &nearest ) const
{
    AMP_ASSERT( !xi.empty() );
    AMP_ASSERT( f.size() == ArraySize( d_x.size( 1 ) ) );
    PROFILE( "interp_nearest", ProfileLevel );
    Array<double> fi( xi.size( 1 ) );
    for ( size_t i = 0; i < fi.length(); i++ )
        fi( i ) = f( nearest( i ) );
    return f;
}


/********************************************************************
 * Function to perform linear interpolation                          *
 ********************************************************************/
template<class TYPE>
template<class TYPE2>
std::tuple<AMP::Array<double>, AMP::Array<double>>
DelaunayInterpolation<TYPE>::interp_linear( const AMP::Array<double> &f,
                                            const AMP::Array<TYPE2> &xi,
                                            const AMP::Array<int> &index,
                                            bool extrap ) const
{
    int ndim  = d_x.size( 0 );
    size_t Ni = xi.size( 1 );
    // Check inputs
    AMP_ASSERT( !d_x.empty() );
    AMP_ASSERT( f.size() == ArraySize( d_x.size( 1 ) ) );
    AMP_ASSERT( !xi.empty() );
    AMP_ASSERT( xi.size() == ArraySize( ndim, Ni ) );
    // Allocate data
    PROFILE( "interp_linear", ProfileLevel );
    AMP::Array<double> fi( Ni );
    AMP::Array<double> gi( ndim, Ni );
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )], f2[NDIM_MAX + 1], L[NDIM_MAX + 1];
    constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    fi.fill( 0 );
    gi.fill( 0 );
    for ( size_t i = 0; i < Ni; i++ ) {
        // Check if the triange index is valid
        if ( index( i ) == -1 && !extrap ) {
            fi( i ) = NaN;
            continue;
        } else if ( index( i ) < 0 ) {
            throw std::logic_error( "Invalid triangle specified" );
        }
        // Compute the Barycentric coordinates
        for ( int j = 0; j < ndim + 1; j++ ) {
            uint32_t k = d_tri( j, index( i ) );
            for ( int j2 = 0; j2 < ndim; j2++ )
                x2[j2 + j * ndim] = d_x( j2, k );
            f2[j] = f( k );
        }
        double xi2[NDIM_MAX];
        for ( int d = 0; d < ndim; d++ )
            xi2[d] = xi( d, i );
        compute_Barycentric( ndim, x2, xi2, L );
        if ( !extrap ) {
            bool outside = false;
            for ( int d = 0; d < ndim + 1; d++ ) {
                if ( L[d] < -1e-8 )
                    outside = true;
            }
            if ( outside ) {
                fi( i ) = NaN;
                continue;
            }
        }
        // Perform the linear interpolation
        fi( i ) = 0.0;
        for ( int j = 0; j < ndim + 1; j++ )
            fi( i ) += L[j] * f2[j];
        // Compute the gradient
        compute_gradient( ndim, x2, f2, &gi( 0, i ) );
    }
    return std::tie( fi, gi );
}


/****************************************************************
 * Function to perform cubic interpolation                       *
 ****************************************************************/
template<class TYPE>
template<class TYPE2>
std::tuple<AMP::Array<double>, AMP::Array<double>>
DelaunayInterpolation<TYPE>::interp_cubic( const AMP::Array<double> &f,
                                           const AMP::Array<double> &g,
                                           const AMP::Array<TYPE2> &xi,
                                           const AMP::Array<int> &index,
                                           int extrap ) const
{
    PROFILE( "interp_cubic", ProfileLevel );
    int ndim  = d_x.size( 0 );
    size_t Ni = xi.size( 1 );
    // Check inputs
    AMP_ASSERT( !d_x.empty() );
    AMP_ASSERT( f.size() == ArraySize( d_x.size( 1 ) ) );
    AMP_ASSERT( g.size() == ArraySize( ndim, d_x.size( 1 ) ) );
    AMP_ASSERT( !xi.empty() );
    AMP_ASSERT( xi.size() == ArraySize( ndim, Ni ) );
    // Allocate data
    AMP::Array<double> fi( Ni );
    AMP::Array<double> gi( ndim, Ni );
    fi.fill( 0 );
    gi.fill( 0 );
    double xi0[NDIM_MAX];
    for ( uint32_t i = 0; i < Ni; i++ ) {
        for ( int d = 0; d < ndim; d++ )
            xi0[d] = xi( d, i );
        interp_cubic_single( f.data(), g.data(), xi0, index( i ), fi( i ), &gi( 0, i ), extrap );
    }
    return std::tie( fi, gi );
}
template<class TYPE>
void DelaunayInterpolation<TYPE>::interp_cubic_single( const double f[],
                                                       const double g[],
                                                       const double xi[],
                                                       const int index,
                                                       double &fi,
                                                       double *gi,
                                                       int extrap ) const
{
    PROFILE( "interp_cubic_single", ProfileLevel );
    const bool check_collinear = true; // Do we want to perform checks that points are collinear
    double x2[NDIM_MAX * ( NDIM_MAX + 1 )] = { 0 };
    double g2[NDIM_MAX * ( NDIM_MAX + 1 )] = { 0 };
    double f2[NDIM_MAX + 1]                = { 0 };
    double L[NDIM_MAX + 1]                 = { 0 };
    const double nan                       = std::numeric_limits<double>::quiet_NaN();
    // Check if the triange index is valid
    int ndim = d_x.size( 0 );
    if ( index == -1 && extrap == 0 ) {
        fi = std::numeric_limits<double>::quiet_NaN();
        for ( int j = 0; j < ndim + 1; j++ )
            gi[j] = nan;
        return;
    } else if ( index < 0 ) {
        throw std::logic_error( "Invalid triangle specified" );
    }
    // Compute the Barycentric coordinates
    for ( int j = 0; j < ndim + 1; j++ ) {
        uint32_t k = d_tri( j, index );
        for ( int j2 = 0; j2 < ndim; j2++ ) {
            x2[j2 + j * ndim] = d_x( j2, k );
            g2[j2 + j * ndim] = g[j2 + k * ndim];
        }
        f2[j] = f[k];
    }
    compute_Barycentric( ndim, x2, xi, L );
    for ( int j = 0; j < ndim + 1; j++ ) {
        if ( fabs( L[j] ) < TRI_TOL )
            L[j] = 0.0;
    }
    // Count the number of zero-valued and negitive dimensions
    int N_L_zero = 0;
    int N_L_neg  = 0;
    for ( int j = 0; j < ndim + 1; j++ ) {
        N_L_zero += ( L[j] == 0.0 ) ? 1 : 0;
        N_L_neg += ( L[j] < 0.0 ) ? 1 : 0;
    }
    if ( N_L_zero == ndim ) {
        // We are at a vertex
        for ( int j = 0; j < ndim + 1; j++ ) {
            if ( L[j] != 0.0 ) {
                fi = f2[j];
                for ( int j2 = 0; j2 < ndim; j2++ )
                    gi[j2] = g2[j2 + j * ndim];
                break;
            }
        }
    } else if ( N_L_zero == 0 && N_L_neg == 0 ) {
        // No zero-valued or negivie dimensions, begin the interpolation
        fi = interp_cubic_recursive( ndim, ndim + 1, x2, f2, g2, xi, L, gi );
    } else {
        // Remove any directions that are 0 (edge, face, etc.)
        int N = ndim + 1 - N_L_zero;
        double x3[NDIM_MAX * ( NDIM_MAX + 1 )], f3[NDIM_MAX + 1], g3[NDIM_MAX * ( NDIM_MAX + 1 )],
            L3[NDIM_MAX + 1];
        int k = 0;
        for ( int j1 = 0; j1 < ndim + 1; j1++ ) {
            int k2 = -1;
            if ( L[j1] != 0.0 ) {
                k2 = k;
                k++;
            } else {
                k2 = ndim - ( j1 - k );
            }
            f3[k2] = f2[j1];
            L3[k2] = L[j1];
            for ( int j2 = 0; j2 < ndim; j2++ ) {
                x3[j2 + k2 * ndim] = x2[j2 + j1 * ndim];
                g3[j2 + k2 * ndim] = g2[j2 + j1 * ndim];
            }
        }
        if ( N_L_neg == 0 ) {
            // No negitive directions, we are ready to perform the interpolation
            fi = interp_cubic_recursive( ndim, N, x3, f3, g3, xi, L3, gi );
        } else {
            // We need to deal with the negitive coordinates
            if ( extrap == 0 ) {
                fi = std::numeric_limits<double>::quiet_NaN();
                for ( int d = 0; d < ndim; d++ )
                    gi[d] = std::numeric_limits<double>::quiet_NaN();
            } else if ( extrap == 1 ) {
                // Use linear interpolation based on the nearest node and it's gradient
                double dist = 1e100;
                int index2  = 0;
                for ( int j = 0; j < ndim + 1; j++ ) {
                    double dist2 = 0.0;
                    for ( int d = 0; d < ndim; d++ )
                        dist2 += ( xi[d] - x2[d + j * ndim] ) * ( xi[d] - x2[d + j * ndim] );
                    if ( dist2 < dist ) {
                        index2 = j;
                        dist   = dist2;
                    }
                }
                fi = f2[index2];
                for ( int d = 0; d < ndim; d++ ) {
                    fi += g2[d + index2 * ndim] * ( xi[d] - x2[d + index2 * ndim] );
                    gi[d] = g2[d + index2 * ndim];
                }
            } else if ( extrap == 2 ) {
                // Use quadratic interpolation
                if ( N == 2 ) {
                    // We can perform interpolation along a line
                    fi = interp_cubic_recursive( ndim, N, x3, f3, g3, xi, L3, gi );
                } else {
                    // Choose two points within (or on) the triangle
                    double xi1[NDIM_MAX], xi2[NDIM_MAX], fi1, fi2, gi1[NDIM_MAX], gi2[NDIM_MAX];
                    get_interpolant_points( ndim, N, x3, L3, xi, xi1, xi2, check_collinear );
                    // Use cubic interpolation to get f and g for the two points
                    interp_cubic_single( f, g, xi1, index, fi1, gi1, 0 );
                    interp_cubic_single( f, g, xi2, index, fi2, gi2, 0 );
                    // Perform quadratic interpolation using a linear approximation to the gradient
                    fi = interp_line( ndim, xi1, fi1, gi1, xi2, fi2, gi2, xi, gi );
                }
            } else {
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
            auto msg = AMP::Utilities::stringf(
                "get_interpolant_points failed: collinear (%i,%i)", N_neg, N );
            throw std::logic_error( msg );
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
    r  = std::sqrt( r );
    rx = std::sqrt( rx );
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
    PROFILE( "create_node_neighbors", ProfileLevel );
    // Allocate the data
    size_t N           = d_x.size( 1 );
    int ndim           = d_x.size( 0 );
    size_t N_tri       = d_tri.size( 1 );
    d_N_node           = new unsigned int[N];
    auto node_list_tmp = new uint32_t *[N];
    node_list_tmp[0]   = new unsigned int[2 * ndim * ( ndim + 1 ) * N_tri];
    // Count the number of nodes that are connected to any other node
    for ( uint32_t i = 0; i < N; i++ )
        d_N_node[i] = 0;
    for ( size_t i = 0; i < N_tri * ( ndim + 1 ); i++ ) {
        d_N_node[d_tri( i )] += ndim;
    }
    // Break the node list array into sub arrays to store the neighbor nodes
    for ( uint32_t i = 1; i < N; i++ )
        node_list_tmp[i] = &node_list_tmp[i - 1][d_N_node[i - 1]];
    // For each triangle, add the neighbor nodes
    for ( uint32_t i = 0; i < N; i++ )
        d_N_node[i] = 0;
    for ( uint32_t i = 0; i < N_tri; i++ ) {
        for ( int j = 0; j <= ndim; j++ ) {
            int j1 = d_tri( j, i );
            int j2 = d_N_node[j1];
            for ( int k = 0; k <= ndim; k++ ) {
                if ( j == k )
                    continue;
                node_list_tmp[j1][j2] = d_tri( k, i );
                j2++;
            }
            d_N_node[j1] += ndim;
        }
    }
    // Eliminate duplicate entries in the node list and sort the list
    for ( uint32_t i = 0; i < N; i++ ) {
        AMP::Utilities::quicksort( d_N_node[i], node_list_tmp[i] );
        int k = 0;
        for ( uint32_t j = 1; j < d_N_node[i]; j++ ) {
            if ( node_list_tmp[i][j] != node_list_tmp[i][k] ) {
                node_list_tmp[i][k + 1] = node_list_tmp[i][j];
                k++;
            }
        }
        d_N_node[i] = k + 1;
    }
    // Create the final list that contains storage only for the needed values
    d_N_node_sum = 0;
    for ( uint32_t i = 0; i < N; i++ )
        d_N_node_sum += d_N_node[i];
    d_node_list    = new uint32_t *[N];
    d_node_list[0] = new unsigned int[d_N_node_sum];
    for ( uint32_t i = 0; i < d_N_node_sum; i++ )
        d_node_list[0][i] = static_cast<unsigned int>( -1 );
    for ( uint32_t i = 1; i < N; i++ )
        d_node_list[i] = &d_node_list[i - 1][d_N_node[i - 1]];
    for ( uint32_t i = 0; i < N; i++ ) {
        for ( uint32_t j = 0; j < d_N_node[i]; j++ )
            d_node_list[i][j] = node_list_tmp[i][j];
    }
    // Delete the temporary memory
    delete[] node_list_tmp[0];
    delete[] node_list_tmp;
}


/**************************************************************************
 * Function to get a list of the triangles that neighbors to each triangle *
 **************************************************************************/
template<class TYPE>
void DelaunayInterpolation<TYPE>::create_tri_neighbors() const
{
    size_t N = d_tri.size( 1 );
    int ndim = d_x.size( 0 );
    AMP_ASSERT( d_tri.size() == ArraySize( ndim + 1, N ) );
    d_tri_nab.resize( ndim + 1, N );
    if ( ndim == 1 ) {
        std::vector<std::array<int, 2>> tri( N );
        memcpy( tri.data()->data(), d_tri.data(), 2 * N * sizeof( int ) );
        auto nab = DelaunayHelpers::create_tri_neighbors<1>( tri );
        memcpy( d_tri_nab.data(), nab.data()->data(), 2 * N * sizeof( int ) );
    } else if ( ndim == 2 ) {
        std::vector<std::array<int, 3>> tri( N );
        memcpy( tri.data()->data(), d_tri.data(), 3 * N * sizeof( int ) );
        auto nab = DelaunayHelpers::create_tri_neighbors<2>( tri );
        memcpy( d_tri_nab.data(), nab.data()->data(), 3 * N * sizeof( int ) );
    } else if ( ndim == 3 ) {
        std::vector<std::array<int, 4>> tri( N );
        memcpy( tri.data()->data(), d_tri.data(), 4 * N * sizeof( int ) );
        auto nab = DelaunayHelpers::create_tri_neighbors<3>( tri );
        memcpy( d_tri_nab.data(), nab.data()->data(), 4 * N * sizeof( int ) );
    } else {
        AMP_ERROR( "Unsupported number of dimensions" );
    }
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
    size_t N     = d_x.size( 1 );
    int ndim     = d_x.size( 0 );
    size_t N_tri = d_tri.size( 1 );
    d_node_tri   = new int[N];
    memset( d_node_tri, 0, N * sizeof( int ) );
    for ( size_t i = 0; i < N_tri; i++ ) {
        for ( int j = 0; j <= ndim; j++ )
            d_node_tri[d_tri( j, i )] = static_cast<int>( i );
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
    double r[NDIM_MAX] = { 0 };
    for ( int i = 0; i < ndim; i++ )
        r[i] = xi[i] - x[i + ndim * ndim];
    DelaunayHelpers::solve( ndim, T, r, L );
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
    DelaunayHelpers::solve<double, 3>( M, b, y, det );
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
    DelaunayHelpers::solve<double, 4>( M, b, y, det );
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


/********************************************************************
 * Function to perform block Gauss-Seidel iteration                  *
 * Note: the performance of this algorithum is strongly dependent on *
 * the memory storage.  For example even passing the row and column  *
 * index instead of N_row and icol increased runtime by ~100x.       *
 ********************************************************************/
static void Gauss_Seidel( const uint32_t Nb,
                          const uint32_t N,
                          const unsigned int,
                          const double D[],
                          const uint32_t N_row[],
                          uint32_t *icol[],
                          const double A[],
                          const double rhs[],
                          double *x,
                          const int N_it )
{
    // First we will compute the inverse of each block
    auto D_inv = new double[Nb * Nb * N];
    for ( uint32_t i = 0; i < N; i++ )
        DelaunayHelpers::inverse( Nb, &D[i * Nb * Nb], &D_inv[i * Nb * Nb] );
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
            for ( uint32_t i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                tmp[0] = rhs[0 + i * 2];
                tmp[1] = rhs[1 + i * 2];
                for ( uint32_t j = 0; j < N_row[i]; j++ ) {
                    uint32_t k = icol[i][j];
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
            L2_norm  = std::sqrt( L2_norm );
            L2_error = std::sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
    } else if ( Nb == 3 ) {
        double tmp[3], x_new[3], x_old[3];
        for ( int it = 0; it < N_it; it++ ) {
            int m           = 0;
            double L2_norm  = 0.0;
            double L2_error = 0.0;
            for ( uint32_t i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                tmp[0] = rhs[0 + i * 3];
                tmp[1] = rhs[1 + i * 3];
                tmp[2] = rhs[2 + i * 3];
                for ( uint32_t j = 0; j < N_row[i]; j++ ) {
                    uint32_t k       = icol[i][j];
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
            L2_norm  = std::sqrt( L2_norm );
            L2_error = std::sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
    } else {
        auto tmp = new double[Nb];
        for ( int it = 0; it < N_it; it++ ) {
            int m           = 0;
            double L2_norm  = 0.0;
            double L2_error = 0.0;
            for ( uint32_t i = 0; i < N; i++ ) {
                // Compute bi-sum(aij*x(j,k),j>i)-sum(aij*x(j+1,k),j<i)
                for ( uint32_t j = 0; j < Nb; j++ )
                    tmp[j] = rhs[j + i * Nb];
                for ( uint32_t j = 0; j < N_row[i]; j++ ) {
                    uint32_t k = icol[i][j];
                    for ( uint32_t j1 = 0; j1 < Nb; j1++ ) {
                        for ( uint32_t j2 = 0; j2 < Nb; j2++ ) {
                            tmp[j1] -= A[j1 + j2 * Nb + m * Nb * Nb] * x[j2 + k * Nb];
                        }
                    }
                    m++;
                }
                // Update x(:,i)
                for ( uint32_t j1 = 0; j1 < Nb; j1++ ) {
                    double x_new = 0.0;
                    for ( uint32_t j2 = 0; j2 < Nb; j2++ ) {
                        x_new += D_inv[j1 + j2 * Nb + i * Nb * Nb] * tmp[j2];
                    }
                    double x_old   = x[j1 + i * Nb];
                    x[j1 + i * Nb] = x_new;
                    L2_norm += x_new * x_new;
                    L2_error += ( x_old - x_new ) * ( x_old - x_new );
                }
            }
            // Check the quality of the new solution
            L2_norm  = std::sqrt( L2_norm );
            L2_error = std::sqrt( L2_error );
            if ( ( L2_error / L2_norm ) < rel_tol || L2_error < abs_tol )
                break;
        }
        delete[] tmp;
    }
    // Delete temporary memory
    delete[] D_inv;
}


// Explicit instantiations
// clang-format off
using FG = std::tuple<AMP::Array<double>, AMP::Array<double>>;

template class DelaunayInterpolation<int>;
template class DelaunayInterpolation<double>;

template Array<size_t> DelaunayInterpolation<int>::find_nearest<int>( const Array<int>& ) const;
template Array<int> DelaunayInterpolation<int>::find_tri<int>( const Array<int>&, bool ) const;
template Array<double> DelaunayInterpolation<int>::interp_nearest<int>( const Array<double>&, const Array<int>&, const Array<size_t>& ) const;
template FG DelaunayInterpolation<int>::interp_linear<int>( const AMP::Array<double>&, const AMP::Array<int>&, const AMP::Array<int>&, bool ) const;
template FG DelaunayInterpolation<int>::interp_cubic<int>( const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<int>&, const AMP::Array<int>&, int ) const;

template Array<size_t> DelaunayInterpolation<int>::find_nearest<double>( const Array<double>& ) const;
template Array<int> DelaunayInterpolation<int>::find_tri<double>( const Array<double>&, bool ) const;
template Array<double> DelaunayInterpolation<int>::interp_nearest<double>( const Array<double>&, const Array<double>&, const Array<size_t>& ) const;
template FG DelaunayInterpolation<int>::interp_linear<double>( const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<int>&, bool ) const;
template FG DelaunayInterpolation<int>::interp_cubic<double>( const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<int>&, int ) const;

template Array<size_t> DelaunayInterpolation<double>::find_nearest<double>( const Array<double>& ) const;
template Array<int> DelaunayInterpolation<double>::find_tri<double>( const Array<double>&, bool ) const;
template Array<double> DelaunayInterpolation<double>::interp_nearest<double>( const Array<double>&, const Array<double>&, const Array<size_t>& ) const;
template FG DelaunayInterpolation<double>::interp_linear<double>( const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<int>&, bool ) const;
template FG DelaunayInterpolation<double>::interp_cubic<double>( const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<double>&, const AMP::Array<int>&, int ) const;

} // namespace AMP
