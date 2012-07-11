#include <ampmesh/latex_visualization_tools.h>
#include <ampmesh/euclidean_geometry_tools.h>

#include <vector>
#include <string>
#include <cassert>
#include <iostream>

std::string rubiks_cube_color_arrangement[6] = { "orange", "green", "white", "blue", "yellow", "red" };

void write_point(double const * p, std::ostream & os) {
  os<<"("<<p[0]<<","<<p[1]<<","<<p[2]<<")";
}

void write_cycle(unsigned int n, double const * * c, std::ostream & os) {
  for (unsigned int i = 0; i < n; ++i) {
    write_point(c[i], os);
    os<<" -- ";
  } // end for i
  os<<"cycle ;\n";
}

void write_face(double const * * f, std::ostream & os) {
  write_cycle(4, f, os);
}

void write_triangle(double const * * t, std::ostream & os) {
  write_cycle(3, t, os);
}

void draw_face(hex8_element_t * e_ptr, unsigned int f, const std::string & option, std::ostream & os) {
  assert(f < 6);
  std::vector<double const *> sp_ptr(4);
  unsigned int const * f_ptr = e_ptr->get_face(f);
  for (unsigned int p = 0; p < 4; ++p) { sp_ptr[p] = e_ptr->get_support_point(f_ptr[p]); } 
  os<<"\\draw["<<option<<"]\n";
  write_face(&(sp_ptr[0]), os);
}

void draw_triangle(triangle_t * t_ptr, const std::string & option, std::ostream & os) {
  std::vector<double const *> sp_ptr(3);
  for (unsigned int i = 0; i < 3; ++i) { sp_ptr[i] = t_ptr->get_support_point_ptr(i); }
  os<<"\\draw["<<option<<"]\n";
  write_triangle(&(sp_ptr[0]), os);
}

void draw_bounding_box(hex8_element_t * e_ptr, double const * point_of_view, std::ostream & os) {
  double const *bb_ptr = e_ptr->get_bounding_box();
  std::string plane[6] = { "yx", "zx", "zy", "zx", "zy", "yx" };
  std::string fixed_coord[6] = { "z", "y", "x", "y", "x", "z" };
  unsigned int fixed_coord_index[6] = { 2, 1, 3, 4, 0, 5 };

  unsigned int first_point_second_coord_index[6] = { 0, 0, 1, 0, 1, 0 }; 
  unsigned int first_point_first_coord_index[6] = { 1, 2, 2, 2, 2, 1 }; 
  unsigned int second_point_second_coord_index[6] = { 3, 3, 4, 3, 4, 3 }; 
  unsigned int second_point_first_coord_index[6] = { 4, 5, 5, 5, 5, 4 }; 

  double normals[18] = {
    0.0, 0.0, -1.0,
    0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0
  };

  os<<"\\tikzset{facestyle/.style={fill=none,draw=black,line join=round}}\n";
  for (unsigned int f = 0; f < 6; ++f) {
    os<<"\\begin{scope}[canvas is "<<plane[f]<<" plane at "<<fixed_coord[f]<<"="<<bb_ptr[fixed_coord_index[f]]<<"]\n";
    os<<"\\path[facestyle"<<((compute_scalar_product(point_of_view, normals+3*f) > 0.0) ? "" : ",dotted")<<"] ";
    os<<"("<<bb_ptr[first_point_first_coord_index[f]]<<","<<bb_ptr[first_point_second_coord_index[f]]<<")";
    os<<" rectangle ";
    os<<"("<<bb_ptr[second_point_first_coord_index[f]]<<","<<bb_ptr[second_point_second_coord_index[f]]<<")";
    os<<" ;\n";
    os<<"\\end{scope}\n";
  } // end for f
}  

void draw_bounding_polyhedron(hex8_element_t * e_ptr, double const * point_of_view, std::ostream & os) {
  os<<"\\tikzset{facestyle/.style={opacity=0.4,line join=round}}\n";
  std::vector<std::string> options(12, "facestyle,");
  triangle_t * t_ptr = e_ptr->get_bounding_polyhedron();
  for (unsigned int f = 0; f < 6; ++f) { 
    for (unsigned int t = 0; t < 2; ++t) {
      if (compute_scalar_product(point_of_view, (t_ptr+2*f+t)->get_normal()) > 0.0) {
//        options[2*f+t] += "fill=" + rubiks_cube_color_arrangement[f];
        options[2*f+t] += "fill=none";
      } else {
        options[2*f+t] += "fill=none,dotted";
      } // end if
      draw_triangle(t_ptr+2*f+t, options[2*f+t]);
    } // end for t
  } // end for f
}

void draw_hex8_element(hex8_element_t * e_ptr, double const * point_of_view, std::ostream & os) {
  os<<"\\tikzset{facestyle/.style={opacity=0.4,line join=round}}\n";
  std::vector<std::string> options(6, "facestyle,");
  triangle_t * t_ptr = e_ptr->get_bounding_polyhedron();
  for (unsigned int f = 0; f < 6; ++f) { 
    if (compute_scalar_product(point_of_view, (t_ptr+2*f)->get_normal()) > 0.0) {
      options[f] += "fill=" + rubiks_cube_color_arrangement[f];
//      options[f] += "fill=none";
    } else {
      options[f] += "fill=none,dotted";
    } // end if
    draw_face(e_ptr, f, options[f]); 
  } 
}

void draw_point(double const * p, std::string option, std::ostream & os) {
  os<<"\\node["<<option<<"] at ";
  write_point(p, os);
  os<<" {+} ;\n";
}

