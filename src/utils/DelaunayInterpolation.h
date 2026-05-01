#ifndef included_AMP_DelaunayInterpolation
#define included_AMP_DelaunayInterpolation

#include <stdlib.h>
#include <tuple>

#include "AMP/utils/Array.h"
#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/kdtree.h"


namespace AMP {


/** \class DelaunayInterpolation
 *
 * This class provides Delaunay based N-dimensional simplex interpolation.
 */
class DelaunayInterpolation
{
public:
    //! Empty constructor
    DelaunayInterpolation();

    // Deleted constructors
    DelaunayInterpolation( const DelaunayInterpolation & )            = delete;
    DelaunayInterpolation &operator=( const DelaunayInterpolation & ) = delete;


    //! Function to construct the tessellation
    /*!
     * This function creates the tessellation using the given points.
     * @param x         The coordinates of the vertices ( ndim x N )
     */
    template<class TYPE>
    DelaunayInterpolation( const AMP::Array<TYPE> &x ) : DelaunayInterpolation()
    {
        d_x.copy( x );
        int ndim = x.size( 0 );
        AMP_ASSERT( ndim >= 1 && ndim <= 3 && x.ndim() == 2 );
        auto y = AMP::DelaunayHelpers::convert( x );
        if ( x.size( 0 ) == 1 ) {
            std::tie( d_tri, d_tri_nab ) = DelaunayTessellation::create_tessellation<1>( y );
        } else if ( ndim == 2 ) {
            std::tie( d_tri, d_tri_nab ) = DelaunayTessellation::create_tessellation<2>( y );
        } else if ( ndim == 3 ) {
            std::tie( d_tri, d_tri_nab ) = DelaunayTessellation::create_tessellation<3>( y );
        }
    }


    //! Function to construct the tessellation using a given tessellation
    /*!
     * This function sets the internal tessellation to match a provided tessellation.
     * It does not check if the provided tessellation is valid.  It allows the user to
     * provide their own tessellation if desired.
     * If sucessful, this routine returns 0.
     * @param x         The coordinates of the vertices ( ndim x N )
     *                  or to update the coordinate pointers before each call.
     *                  See update_coordinates for more information.
     * @param tri       The tesselation ( ndim+1 x N_tri )
     */
    template<class TYPE>
    DelaunayInterpolation( const Array<TYPE> &x, const Array<int> &tri ) : DelaunayInterpolation()
    {
        d_x.copy( x );
        d_tri     = tri;
        d_tri_nab = DelaunayHelpers::create_tri_neighbors( d_tri );
    }


    //! Empty destructor
    ~DelaunayInterpolation();


    //! Function to return the number of triangles in the tessellation
    /*!
     * This function returns the number of triangles in the tessellation.
     */
    size_t get_N_tri() const;


    //! Function to return the triangles in the tessellation
    const AMP::Array<int> &get_tri() const;


    //! Function to return the triangles neignbors
    const AMP::Array<int> &get_tri_nab() const;


    //! Function to return the verticies
    const AMP::Array<double> &get_x() const;


    //! Subroutine to find the nearest neighbor to a point
    /*!
     * This function finds the nearest neighbor to each point.
     * It is able to do perform the search in O(N^(1/ndim)) on average.
     * Note: this function requires the calculate of the node lists if they are not stored (see
     * set_storage_level)
     * @param xi        Coordinates of the query points ( ndim x Ni )
     * @return          Return the index of the nearest neighbor (N)
     */
    template<class TYPE>
    inline Array<size_t> find_nearest( const Array<TYPE> &xi ) const
    {
        return find_nearest2( xi.template cloneTo<double>() );
    }


    //! Subroutine to find the triangle that contains the point
    /*!
     * This function finds the triangle that contains the given points.
     * This uses a simple search method, which may not be the most efficient.  It is O(N*Ni).
     * Note: this function requires the calculate of the node lists and triangle neighbor
     * lists if they have not been calculated
     * @param xi        Coordinates of the query points ( ndim x Ni )
     * @param extrap    If the point is outside the convex hull,
     *                  return the nearest triangle instead of -1
     * @return          Ouput index of triangle containing the point
     *                  ( -1: Point is outside convex hull, -2: Search failed )
     */
    template<class TYPE>
    inline Array<int> find_tri( const Array<TYPE> &xi, bool extrap = false ) const
    {
        return find_tri2( xi.template cloneTo<double>(), extrap );
    }


    //! Subroutine to calculate the gradient at each node
    /*!
     * This function gets a list of the nodes that connect to each node
     * @param f         Function values at the vertices (ndim)
     * @param method    Gradient method to use
     *                  1 - Use a simple least squares method using only the local nodes.
     *                      This method is relatively fast, but only first order in the gradient,
     *                      causing the truncation error of the interpolate to be O(x^3).
     *                      Note that it will still give a better cubic interpolation than MATLAB's
     *                      cubic in griddata.
     *                  2 - Least squares method that solves a sparse system.
     *                      This method is second order in the gradient yielding an interpolant that
     *                      has truncation error O(x^4), but requires soving a sparse 2nx2n system.
     *                      This can be reasonably fast for most systems, but is memory limited
     *                      for large systems (it can grow as O(n^2*ndim^2)).
     *                      Currently this method is NOT implemented in the C++ version.
     *                 3 - Least squares method that uses the matrix computed by method 2 and block
     *                      Gauss-Seidel iteration to improve the gradient calculated by method 1.
     *                      Usually only a finite number of iterations are needed to significantly
     *                      reduce the error.  Technically, this method has a trunction error that
     *                      is still O(x^3), but for most purposes, this term is reduced so it is
     *                      less than the O(x^4) term.  For most systems, this method is not
     *                      significantly faster than method 2, but it does not have the same memory
     *                      limitations and is O(n*ndim^2) in memory.
     *                      Typically 10-20 iterations are all that is necessary.
     *                 4 - This is the same as method 3, but does not store any internal data.
     *                      This saves us from creating a large temporary structure ~10*ndim^2*N
     *                      at a cost of ~2x in performance.
     * @param grad      (Output) Calculated gradient at the nodes ( ndim x N )
     * @param n_it      Optional argument specifying the number of Gauss-Seidel iterations.
     *                  Only used if method = 3 or 4.
     */
    void calc_node_gradient( const double *f,
                             const int method,
                             double *grad,
                             const int n_it = 20 ) const;


    //! Subroutine to perform nearest-neighbor interpoaltion
    /*!
     * This function performs nearest-neighbor interpoaltion.
     * @param f         Function values at the triangle vertices (N)
     * @param nearest   The nearest-neighbor points (see find_nearest) ( Ni)
     * @return          Return the interpolated function values at xi (Ni)
     */
    Array<double> interp_nearest( const Array<double> &f, const Array<size_t> &nearest ) const;


    //! Subroutine to perform linear interpoaltion
    /*!
     * This function performs linear interpoaltion.
     * If a valid triangle index is not given, NaN will be returned.
     * If extrap is false and the point is not within the triangle, NaN will be returned.
     * @param f         Function values at the triangle vertices ( N )
     * @param xi        Coordinates of the query points ( ndim x Ni )
     * @param index     The index of the triangle containing the point (see find_tri)
     * @param extrap    Do we want to extrapolate from the current triangle
     *                  Note: extrapolating can incure large error if sliver triangles
     *                  on the boundary are present
     * @return          Return the interpolated function values and gradient at xi <fi,gi>
     *                  fi - The interpolated function values ( Ni )
     *                  gi - The interpolated gradient ( ndim x Ni )
     */
    template<class TYPE>
    inline std::tuple<AMP::Array<double>, AMP::Array<double>>
    interp_linear( const AMP::Array<double> &f,
                   const AMP::Array<TYPE> &xi,
                   const AMP::Array<int> &index,
                   bool extrap = false ) const
    {
        return interp_linear2( f, xi.template cloneTo<double>(), index, extrap );
    }


    //! Subroutine to perform cubic interpoaltion
    /*!
     * This function performs cubic interpoaltion.
     * Note: If the point is not contained within a triangle NaN will be returned.
     * @param f         Function values at the triangle vertices (N)
     * @param g         Gradient of f(x) at the triangle vertices ( ndim x N )
                        (see calc_node_gradient if unknown)
     * @param xi        Coordinates of the query points ( ndim x Ni )
     * @param index     The index of the triangle containing the point (see find_tri)
     * @param extrap    Do we want to extrapolate from the current triangle
     *                  0: Do not extrapolate (NaNs will be used for points outside the domain)
     *                  1: Extrapolate using linear interpolation
     *                     (using the nearest point and it's gradient)
     *                  2: Extrapolate using quadratic interpolation
     *                     (using linear extrapolation for the gradient)
     * @return          Return the interpolated function values and gradient at xi <fi,gi>
     *                  fi - The interpolated function values ( Ni )
     *                  gi - The interpolated gradient ( ndim x Ni )
     */
    template<class TYPE>
    inline std::tuple<AMP::Array<double>, AMP::Array<double>>
    interp_cubic( const AMP::Array<double> &f,
                  const AMP::Array<double> &g,
                  const AMP::Array<TYPE> &xi,
                  const AMP::Array<int> &index,
                  int extrap = 0 ) const
    {
        return interp_cubic2( f, g, xi.template cloneTo<double>(), index, extrap );
    }


private:
    using FG  = std::tuple<AMP::Array<double>, AMP::Array<double>>;
    using VEC = AMP::Array<double>;
    Array<size_t> find_nearest2( const VEC & ) const;
    Array<int> find_tri2( const VEC &, bool ) const;
    FG interp_linear2( const VEC &, const VEC &, const AMP::Array<int> &, bool extrap ) const;
    FG interp_cubic2( const VEC &, const VEC &, const VEC &, const AMP::Array<int> &, int ) const;
    void create_node_neighbors() const;
    void create_node_tri() const;
    void create_kdtree() const;
    void interp_cubic_single(
        const double[], const double[], const double[], const int, double &, double *, int ) const;


private:                            // Internal Data
    Array<double> d_x;              // Pointer to the coordinates (ndim x N)
    Array<int> d_tri;               // Pointer to the coordinates (ndim+1 x N_tri)
    mutable Array<int> d_tri_nab;   // List of neighbor triangles ( ndim+1 x N_tri )
    mutable size_t d_N_node_sum;    // The sum of the number of node neighbors
    mutable unsigned *d_N_node;     // The number of neighbor nodes for each node (1xN)
    mutable unsigned **d_node_list; // The list of neighbor nodes for each node (1xN)
                                    // Note: The first element points to an array of size N_node_sum
    mutable int *d_node_tri;        // For each node, a triangle that contains that node
    mutable kdtree *d_tree;         // Nearest neighbor search tree
};


} // namespace AMP

#endif
