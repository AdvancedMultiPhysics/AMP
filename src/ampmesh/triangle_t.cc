#include <ampmesh/triangle_t.h> 

triangle_t::triangle_t(double const * A, double const * B, double const * C) {
  set_support_points(A, B, C);
}

void triangle_t::set_support_points(double const * A, double const * B, double const * C) {
  if (support_points_ptr.size() == 0) { support_points_ptr.resize(3); }
  support_points_ptr[0] = A; 
  support_points_ptr[1] = B; 
  support_points_ptr[2] = C; 
  normal_updated = false;
  centroid_updated = false;
}

void triangle_t::set_support_points(double const * * ptr) {
  set_support_points(ptr[0], ptr[1], ptr[2]);
}

double const * triangle_t::get_support_point_ptr(unsigned int i) const {
  assert(i < 3);
  return support_points_ptr[i];
}

/*double const * * triangle_t::get_support_points_ptr() const {
  return &(support_points_ptr[0]);
}*/

double const * triangle_t::get_normal() {
  if (!normal_updated) { compute_normal(); }
  return &(normal[0]);
}

double const * triangle_t::get_centroid() {
  if (!centroid_updated) { compute_centroid(); }
  return &(centroid[0]);
}

void triangle_t::compute_centroid() {
  if (centroid.size() == 0) { centroid.resize(3); }
  assert(!centroid_updated);
  for (unsigned int i = 0; i < 3; ++i) {
    centroid[i] = 0.0;
    for (unsigned int j = 0; j < 3; ++j) {
      centroid[i] += support_points_ptr[j][i];
    } // end for j
    centroid[i] /= 3.0;
  } // end for i
  centroid_updated = true;
}

void triangle_t::compute_normal() {
  if (normal.size() == 0) { normal.resize(3); tmp.resize(6); }
  assert(!normal_updated);
  make_vector_from_two_points(support_points_ptr[0], support_points_ptr[1], &(tmp[0])+0);
  make_vector_from_two_points(support_points_ptr[0], support_points_ptr[2], &(tmp[0])+3);
  compute_cross_product(&(tmp[0])+0, &(tmp[0])+3, &(normal[0]));
  double normalizing_factor = 1.0 / sqrt(std::inner_product(&(normal[0]), &(normal[0])+3, &(normal[0]), 0.0));
  assert(normalizing_factor < 1.0e12);
  for (unsigned int i = 0; i < 3; ++i) { normal[i] *= normalizing_factor; }
  normal_updated= true;
}

// check whether plane supported by the triangle ABC is above the point with respect to the normal
bool triangle_t::above_point(double const * p, double tolerance) {
  if (!normal_updated) { 
    compute_normal();
  } // end if
  if (!centroid_updated) { 
    compute_centroid();
  } // end if
  make_vector_from_two_points(&(centroid[0]), p, &(tmp[0]));
  return (compute_scalar_product(&(tmp[0]), &(normal[0])) < tolerance);
}

void scale_points(unsigned int direction, double scaling_factor, unsigned int n_points, double* points) {
  assert(direction < 3);
  for (unsigned int i = 0; i < n_points; ++i) { 
    points[3*i+direction] *= scaling_factor; 
  } // end for i
}

void scale_points(const std::vector<double> &scaling_factors, unsigned int n_points, double* points) {
  assert(scaling_factors.size() == 3);
  for (unsigned int i = 0; i < 3; ++i) { 
    scale_points(i, scaling_factors[i], n_points, points);
  } // end for i
}

void translate_points(unsigned int direction, double distance, unsigned int n_points, double* points) {
  assert(direction < 3);
  for (unsigned int i = 0; i < n_points; ++i) { 
    points[3*i+direction] += distance; 
  } // end for i
}

void translate_points(std::vector<double> translation_vector, unsigned int n_points, double* points) {
  assert(translation_vector.size() == 3);
  for (unsigned int i = 0; i < 3; ++i) { 
   translate_points(i, translation_vector[i], n_points, points); 
  } // end for i
}

void rotate_points(unsigned int rotation_axis, double rotation_angle, unsigned int n_points, double* points) {
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

std::vector<double> compute_cross_product(const std::vector<double> &u, const std::vector<double> &v) {
  assert(u.size() == 3);
  assert(v.size() == 3);
  std::vector<double> w(3);
  compute_cross_product(&(u[0]), &(v[0]), &(w[0]));
  return w;
}

void compute_cross_product(double const * u, double const * v, double * w) {
  w[0] = u[1]*v[2]-u[2]*v[1];
  w[1] = u[2]*v[0]-u[0]*v[2];
  w[2] = u[0]*v[1]-u[1]*v[0];
}

double compute_scalar_product(double const * u, double const * v) {
  return std::inner_product(&(u[0]), &(u[0])+3, &(v[0]), 0.0);
}

double compute_scalar_product(const std::vector<double> &u, const std::vector<double> &v) {
  assert(u.size() == 3);
  assert(v.size() == 3);
  return compute_scalar_product(&(u[0]), &(v[0]));
}

std::vector<double> make_vector_from_two_points(const std::vector<double> &start_point, const std::vector<double> &end_point) {
  assert(start_point.size() == 3);
  assert(end_point.size() == 3);
  std::vector<double> vector(3);
  make_vector_from_two_points(&(start_point[0]), &(end_point[0]), &(vector[0]));
  return vector;
}

void make_vector_from_two_points(double const * start_point, double const * end_point, double * vector) {
  for (unsigned int i = 0; i < 3; ++i) { vector[i] = end_point[i] - start_point[i]; }
}
