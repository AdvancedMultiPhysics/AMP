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


} // namespace AMP::DelaunayTessellation

#endif
