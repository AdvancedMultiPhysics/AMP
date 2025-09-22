#ifndef included_AMP_DelaunayTessellation
#define included_AMP_DelaunayTessellation

#include <array>
#include <stdint.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#include "AMP/utils/Array.h"


namespace AMP::DelaunayTessellation {


//! Check if
/*!
 * @brief  Check if the points are collinear
 * @details  This function will check if all the points in a set are collinear
 * @param x         The coordinates of the vertices (ndim x N)
 * @return          Returns true if the points are collinear
 */
template<class TYPE>
bool collinear( const Array<TYPE> &x );


//! Function that creates the Delaunay Tessellation
/*!
 * This function will create a valid Delaunay Tessellation in multiple dimensions.
 * Currently only 2D and 3D are supported.  If successful, it will return the number of
 * triangles, if unsuccessful it will throw a std::exception.  Additionally, there are
 * several optional stuctures.
 * @param x         The coordinates of the vertices (ndim x N)
 * @return          Returns the triangles and triangle neighbors <tri,tri_nab>
 *                  tri - The returned pointer where the triangles are stored (ndim+1,N)
 *                  tri_nab - The returned pointer where the triangle neighbors are stored
 *                      (ndim+1,N)
 */
template<class TYPE>
std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation( const Array<TYPE> &x );


//! Function to calculate the volume of a simplex
/*!
 * This function calculates the volume of a N-dimensional simplex
 * Note: the sign of the volume depends on the order of the points.
 *   It will be positive for points stored in a clockwise mannor.
 * Note:  If the volume is zero, then the simplex is invalid.
 *   Eg. a line in 2D or a plane in 3D.
 * @param ndim      The number of dimensions (currently only 2D and 3D are supported)
 * @param x         The coordinates of the vertices of the simplex ( NDIM x NDIM+1 )
 */
double calc_volume( int ndim, const double x[] );


//! Function to check if a point is inside the circumsphere of a simplex.
/*!
 * This function checks if a point is inside the circumsphere of a simplex.
 * It returns -1 if the point is outside the circumsphere, 1 if it is inside the sphere,
 * and 0 if it is within the tolerance of the sphere.
 * Note:  For this function to work properly, the volume of the simplex
 *    (as computed by calc_volume) must be positive.
 * Note:  If we are checking the surface between 2 simplicies and they are both valid
 *    (have a positive, non-zero volume), it is suffcient to check the vertix of 1 volume
 *    against the circumcircle of the other.  We do not need to perform both checks.
 * @param x         The coordinates of the vertices of the simplex
 * @param xi        The coordinates of the vertex to check
 * @param TOL_VOL   A tolerance on the volume to use
 */
template<int NDIM, class TYPE>
int test_in_circumsphere( const std::array<TYPE, NDIM> x[],
                          const std::array<TYPE, NDIM> &xi,
                          double TOL_VOL );


//! Function to return the circumsphere containing a simplex
/*!
 * This function computes the circumsphere that contains a simplex
 * @param[in]  x0       The coordinates of the vertices of the simplex
 * @param[out] R        The radius of the circumsphere
 * @param[out] center   The center of the circumsphere
 */
template<int NDIM, class TYPE>
void get_circumsphere( const std::array<TYPE, NDIM> x0[], double &R, double *center );


//! Subroutine to compute the Barycentric coordinates
/**
 * This function computes the Barycentric coordinates.
 * @param[in]  ndim     The number of dimensions
 * @param x     Coordinates of the triangle vertices ( NDIM x NDIM+1 )
 * @param xi    Coordinates of the desired point ( NDIM )
 * @param L     (output) The Barycentric coordinates of the point ( NDIM+1 )
 */
void compute_Barycentric( const int ndim, const double *x, const double *xi, double *L );


} // namespace AMP::DelaunayTessellation

#endif
