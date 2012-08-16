#include <ampmesh/euclidean_geometry_tools.h>

#include <numeric>
#include <cassert>
#include <cmath>
#include <algorithm>

void scale_points(unsigned int direction, double scaling_factor, unsigned int n_points, double * points) {
  assert(direction < 3);
  for (unsigned int i = 0; i < n_points; ++i) { 
    points[3*i+direction] *= scaling_factor; 
  } // end for i
}

void scale_points(double const * scaling_factors, unsigned int n_points, double * points) {
  for (unsigned int i = 0; i < 3; ++i) { 
    scale_points(i, scaling_factors[i], n_points, points);
  } // end for i
}

void translate_points(unsigned int direction, double distance, unsigned int n_points, double * points) {
  assert(direction < 3);
  for (unsigned int i = 0; i < n_points; ++i) { 
    points[3*i+direction] += distance; 
  } // end for i
}

void translate_points(double const * translation_vector, unsigned int n_points, double * points) {
  for (unsigned int i = 0; i < 3; ++i) { 
   translate_points(i, translation_vector[i], n_points, points); 
  } // end for i
}

void rotate_points(unsigned int rotation_axis, double rotation_angle, unsigned int n_points, double * points) {
  assert(rotation_axis < 3);
  unsigned int non_fixed_directions[2];
  unsigned int i = 0;
  for (unsigned int j = 0; j < 3; ++j) { 
    if (j != rotation_axis) {
      non_fixed_directions[i++] = j;
    } // end if
  } // end for j
  double tmp[3];
  for (unsigned int j = 0; j < n_points; ++j) {
    tmp[non_fixed_directions[0]] = cos(rotation_angle)*points[3*j+non_fixed_directions[0]]-sin(rotation_angle)*points[3*j+non_fixed_directions[1]];
    tmp[non_fixed_directions[1]] = sin(rotation_angle)*points[3*j+non_fixed_directions[0]]+cos(rotation_angle)*points[3*j+non_fixed_directions[1]];
    tmp[rotation_axis] = points[3*j+rotation_axis];
    std::copy(tmp, tmp+3, points+3*j);
  } // end for j
}

void compute_cross_product(double const * u, double const * v, double * w) {
  w[0] = u[1]*v[2]-u[2]*v[1];
  w[1] = u[2]*v[0]-u[0]*v[2];
  w[2] = u[0]*v[1]-u[1]*v[0];
}

double compute_scalar_product(double const * u, double const * v) {
  return std::inner_product(&(u[0]), &(u[0])+3, &(v[0]), 0.0);
}

void make_vector_from_two_points(double const * start_point, double const * end_point, double * vector) {
  for (unsigned int i = 0; i < 3; ++i) { vector[i] = end_point[i] - start_point[i]; }
}

double compute_vector_norm(double const * vector) {
  return sqrt(std::inner_product(&(vector[0]), &(vector[0])+3, &(vector[0]), 0.0));
}

void normalize_vector(double * vector) {
  double vector_norm = compute_vector_norm(vector);
  double normalizing_factor = 1.0 / vector_norm;
  assert(normalizing_factor < 1.0e12);
  for (unsigned int i = 0; i < 3; ++i) { vector[i] *= normalizing_factor; }
}

double compute_distance_between_two_points(double const * start_point, double const * end_point) {
  double tmp[3];
  make_vector_from_two_points(start_point, end_point, tmp);
  return compute_vector_norm(tmp);
}
