#ifndef included_AMP_DelaunayTessellation
#define included_AMP_DelaunayTessellation

#include <array>
#include <stdint.h>
#include <stdlib.h>
#include <tuple>
#include <vector>

#include "AMP/utils/Array.h"


namespace AMP::DelaunayTessellation {


/*!
 * @brief  Check if all the points are collinear
 * @details  This function will check if all the points in a set are collinear
 *    Note all values for the coordinates are int whos value must be strictly |x| < 2^30.
 *    Use AMP::DelaunayHelpers::convert to convert coordinates.
 * @param x         The coordinates of the vertices (ndim x N)
 * @return          Returns true if the points are collinear
 */
bool collinear( const AMP::Array<int> &x );


/*!
 * @brief  Creates a Delaunay Tessellation
 * @details  This function will create a valid Delaunay Tessellation in multiple dimensions.
 *    Currently only 2D and 3D are supported.  If successful, it will return the number of
 *    triangles, if unsuccessful it will throw a std::exception.
 *    Note all values for the coordinates are int whos value must be strictly |x| < 2^30.
 *    Use AMP::DelaunayHelpers::convert to convert coordinates.
 * @param x         The coordinates of the vertices (ndim x N)
 * @return          Returns the triangles and triangle neighbors <tri,tri_nab>
 *                  tri - The returned pointer where the triangles are stored (ndim+1,N)
 *                  tri_nab - The returned pointer where the triangle neighbors are stored
 *                      (ndim+1,N)
 */
template<int NDIM>
std::tuple<AMP::Array<int>, AMP::Array<int>> create_tessellation( const AMP::Array<int> &x );


/*!
 * @brief  Check if a point is inside the circumsphere
 * @details  This function tests if a point is inside the circumsphere of an nd-simplex.
 *    For performance, I assume the points are ordered properly such that
 *    the volume of the simplex (as calculated by calc_volume) is positive.
 *    The point is inside the circumsphere if the determinant is positive
 *    for points stored in a clockwise manner.  If the order is not known,
 *    we can compare to a point we know is inside the cicumsphere.
 *       \f[
 *       \begin{vmatrix}
 *           x_1-x_i & y_1-y_i & z_1-z_i & (x_1-x_i)^2+(y_1-y_i)^2+(z_1-z_i)^2 \\
 *           x_2-x_i & y_2-y_i & z_2-z_i & (x_2-x_i)^2+(y_2-y_i)^2+(z_2-z_i)^2 \\
 *           x_3-x_i & y_3-y_i & z_3-z_i & (x_3-x_i)^2+(y_3-y_i)^2+(z_3-z_i)^2 \\
 *           x_4-x_i & y_4-y_i & z_4-z_i & (x_4-x_i)^2+(y_4-y_i)^2+(z_4-z_i)^2
 *       \end{vmatrix}
 *       \f]
 *    det(A) == 0:  We are on the circumsphere
 *    det(A) > 0:   We are inside the circumsphere
 *    det(A) < 0:   We are outside the circumsphere
 *    Note: this implementation requires N^(D+2) precision
 * @param x         The coordinates of the vertices (ndim x N)
 * @param xi        The coordinates of the point (ndim)
 * @return          Returns +/- 1 if we are inside/outside the circumsphere,
 *                  0 if we are on the surface
 */
template<int NDIM>
int test_in_circumsphere( const std::array<int, NDIM> x[], const std::array<int, NDIM> &xi );

} // namespace AMP::DelaunayTessellation

#endif
