#ifndef included_AMP_NearestPairSearch
#define included_AMP_NearestPairSearch

#include <iostream>
#include <stdlib.h>
#include <utility>
#include <vector>


namespace AMP::Mesh {
class MeshPoint; // Forward declare MeshPoint
} // namespace AMP::Mesh


namespace AMP {

//! Function to compute the closest pair of points
/*!
 * This function will calculate the closest pair of points in a list
 * @param N         The number of points in the list
 * @param x         The coordinates of the vertices (NDIM x N)
 */
template<int NDIM, class TYPE>
inline std::pair<int, int> find_min_dist( const int N, const TYPE *x );


//! Function to compute the closest pair of points
/*!
 * This function will calculate the closest pair of points in a list
 * @param x         The coordinates of the vertices
 */
std::pair<int, int> find_min_dist( const std::vector<AMP::Mesh::MeshPoint> &x );


} // namespace AMP

#include "AMP/utils/NearestPairSearch.hpp"

#endif
