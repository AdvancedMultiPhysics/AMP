
#ifndef HEX8_ELEMENT_T
#define HEX8_ELEMENT_T

#include <ampmesh/triangle_t.h>
#include <vector>

double compute_inverse_3_by_3_matrix(double const *mat, double *inv);
void compute_n_by_n_matrix_times_vector(unsigned int n, double const *mat, double const *vec, double *res);

class hex8_element_t {
  public:
    hex8_element_t(double const *p);
    void set_support_points(double const *p);
    double const * get_support_point(unsigned int i) const;
    double const * get_support_points() const;
    double const * get_bounding_box();
    triangle_t * get_bounding_polyhedron();
    double const * get_scaling_factors();
    void scale_support_points();
    void unscale_support_points();
    bool within_bounding_box(double const *p, double tolerance = 1.0e-12);
    bool within_bounding_polyhedron(double const *p, double tolerance = 1.0e-12);
    // this the user responsability to call first within_bounding_box(...) and within_bounding_polyhedron(...)
    // before trying to map a point so that newton won't fail
    void map_global_to_local(double const *global_coordinates, double *local_coordinates);
    void map_local_to_global(double const *local_coordinates, double *global_coordinates);
    bool contains_point(double const *coordinates, bool coordinates_are_local = false, double tolerance = 1.0e-12);
    void project_on_face(unsigned int f, double const *local_coordinates, double *local_coordinates_on_face, double *shift_global_coordinates);

    static void project_on_face(unsigned int f, double const *local_coordinates, double *local_coordinates_on_face);
    static void get_basis_functions_values(double const *x, double *basis_functions_values);
    static void get_basis_functions_derivatives( double const *x, double *basis_functions_derivatives);
    static void get_basis_functions_values_on_face(double const *x, double *basis_functions_values);
    static unsigned int const * get_face(unsigned int i);
    static unsigned int const * get_faces();

  private:
    // numbering of the 8 support points (or nodes) follows libmesh hex8 convention which is as follows
    //
    //       7        6
    //        o--------o
    //       /:       /|
    //      / :      / |
    //   4 /  :   5 /  |
    //    o--------o   |
    //    |   o....|...o 2
    //    |  .3    |  /
    //    | .      | /
    //    |.       |/
    //    o--------o
    //    0        1
    //
    //
    // reference frame xyz 
    //    z   y
    //    |  .
    //    | .
    //    |.
    //    o------ x
    //
    //
    // node 0 -> x y z = -1 -1 -1
    //      1            +1 -1 -1
    //      2            +1 +1 -1
    //      3            -1 +1 -1
    //      4            -1 -1 +1
    //      5            +1 -1 +1
    //      6            +1 +1 +1
    //      7            -1 +1 +1
    //
    // numbering of the faces is
    // face 0 -> at z=-1 supported by nodes 0321 
    //      1       y=-1                    0154 
    //      2       x=+1                    1265 
    //      3       y=+1                    2376 
    //      4       x=-1                    3047 
    //      5       z=+1                    4567 
    //
    std::vector<double> support_points, point_candidate;

    // faces are oriented and defined by their 4 support nodes
    //   3          2    
    //    o--------o    
    //    |        |
    //    |        |   
    //    |        |  
    //    |        | 
    //    o--------o
    //   0          1
    //
    // coordinates on the face are given in the following xy reference frame
    // 
    //    y
    //    |
    //    |
    //    |
    //    o------ x
    //
    static unsigned int faces[];

    bool support_points_scaled;
    bool scaling_factors_updated;
    std::vector<double> scaling_factors;
    bool support_points_translated;
    bool translation_vector_updated;
    std::vector<double> translation_vector;

    bool bounding_box_updated, bounding_polyhedron_updated;

    std::vector<double> bounding_box;
    std::vector<triangle_t> bounding_polyhedron, tmp_triangles;

    bool center_of_element_data_updated;
    std::vector<double> center_of_element_local_coordinates, center_of_element_global_coordinates, jacobian_matrix_at_center_of_element, inverse_jacobian_matrix_at_center_of_element;

    bool memory_allocated_for_newton;
    std::vector<double> residual_vector, jacobian_matrix, inverse_jacobian_matrix, inverse_jacobian_matrix_times_residual_vector;
    std::vector<double> basis_functions_values, basis_functions_derivatives;

    void compute_scaling_factors();
    void compute_translation_vector();
    void build_bounding_box();
    void build_bounding_polyhedron();
    void compute_center_of_element_data();
    // residual vector f = (sum_i x_i b_i, sum_i y_i b_i, sum_i z_i b_i)^t - (x, y, z)^t
    // where the x_i y_i z_i, i=0...7, are the coordinates of the support points and the b_i are basis functions
    void compute_residual_vector(double const *x, double *f);
    // jacobian matrix J_ij = df_i / dj
    void compute_jacobian_matrix(double const *x, double *J);
    void compute_initial_guess(double *x);
    // map the coordinates of the point candidate onto the reference frame of the volume element defined by the support points
    double solve_newton(double *x, double abs_tol = 1.0e-14, double rel_tol = 1.0e-14, unsigned int max_iter = 100, bool verbose = false);
};

#endif // HEX8_ELEMENT_T


