#include "AMP/mesh/hex8_element_t.h"
#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>


hex8_element_t::hex8_element_t( double const *p ) : memory_allocated_for_newton( false )
{
    //  newton_count = 0;
    set_support_points( p );
}

hex8_element_t::~hex8_element_t()
{
    clear_triangles_ptr( tmp_triangles_ptr );
    clear_triangles_ptr( bounding_polyhedron );
}

void hex8_element_t::set_support_points( double const *p )
{
    std::copy( p, p + 24, &( support_points[0] ) );
    bounding_box_updated           = false;
    bounding_polyhedron_updated    = false;
    center_of_element_data_updated = false;
    scaling_factors_updated        = false;
    support_points_scaled          = false;
}

void hex8_element_t::scale_support_points()
{
    AMP_DEBUG_ASSERT( !support_points_scaled );
    if ( !scaling_factors_updated ) {
        compute_scaling_factors();
    }
    scale_points( &( scaling_factors[0] ), 8, &( support_points[0] ) );
    support_points_scaled = true;
}

void hex8_element_t::unscale_support_points()
{
    AMP_DEBUG_ASSERT( support_points_scaled );
    for ( unsigned int i = 0; i < 3; ++i ) {
        scale_points( i, 1.0 / scaling_factors[i], 8, &( support_points[0] ) );
    } // end for i
    support_points_scaled = false;
}

void hex8_element_t::compute_scaling_factors()
{
    AMP_DEBUG_ASSERT( !scaling_factors_updated );
    if ( !bounding_box_updated ) {
        build_bounding_box();
    };
    if ( scaling_factors.size() == 0 ) {
        scaling_factors.resize( 3 );
    }
    for ( unsigned int i = 0; i < 3; ++i ) {
        scaling_factors[i] = bounding_box[i + 3] - bounding_box[i + 0];
    } // end for i
    scaling_factors_updated = true;
}

double const *hex8_element_t::get_scaling_factors()
{
    if ( !scaling_factors_updated ) {
        compute_scaling_factors();
    }
    return &( scaling_factors[0] );
}

void hex8_element_t::compute_center_of_element_data()
{
    AMP_DEBUG_ASSERT( !center_of_element_data_updated );
    compute_jacobian_matrix( &( center_of_element_local_coordinates[0] ),
                             &( jacobian_matrix_at_center_of_element[0] ) );
    compute_inverse_3_by_3_matrix( &( jacobian_matrix_at_center_of_element[0] ),
                                   &( inverse_jacobian_matrix_at_center_of_element[0] ) );
    map_local_to_global( &( center_of_element_local_coordinates[0] ),
                         &( center_of_element_global_coordinates[0] ) );
    center_of_element_data_updated = true;
}

double const *hex8_element_t::get_support_point( unsigned int i ) const
{
    AMP_DEBUG_ASSERT( i < 8 );
    return &( support_points[3 * i] );
}

double const *hex8_element_t::get_support_points() const { return &( support_points[0] ); }

unsigned int const *hex8_element_t::get_face( unsigned int i )
{
    AMP_DEBUG_ASSERT( i < 6 );
    return &( faces[4 * i] );
}
unsigned int const *hex8_element_t::get_faces() { return &( faces[0] ); }

double const *hex8_element_t::get_bounding_box()
{
    if ( !bounding_box_updated ) {
        build_bounding_box();
    };
    return &( bounding_box[0] );
}

triangle_t **hex8_element_t::get_bounding_polyhedron()
{
    if ( !bounding_polyhedron_updated ) {
        build_bounding_polyhedron();
    };
    return &( bounding_polyhedron[0] );
}

bool hex8_element_t::within_bounding_box( double const *p, double tolerance )
{
    if ( !bounding_box_updated ) {
        build_bounding_box();
    };
    for ( unsigned int j = 0; j < 3; ++j ) {
        // should we scale the tolerance?
        if ( ( bounding_box[j + 0] - tolerance * ( bounding_box[j + 3] - bounding_box[j + 0] ) >
               p[j] ) ||
             ( bounding_box[j + 3] + tolerance * ( bounding_box[j + 3] - bounding_box[j + 0] ) <
               p[j] ) ) {
            return false;
        }
    } // end for j
    return true;
}

unsigned int hex8_element_t::faces[24] = { 0, 3, 2, 1, 0, 1, 5, 4, 1, 2, 6, 5,
                                           2, 3, 7, 6, 3, 0, 4, 7, 4, 5, 6, 7 };

void hex8_element_t::clear_triangles_ptr( std::vector<triangle_t *> &triangles_ptr )
{
    for ( auto &elem : triangles_ptr ) {
        if ( elem != nullptr ) {
            delete elem;
        }
        elem = nullptr;
    } // end for i
    triangles_ptr.clear();
}

void hex8_element_t::build_bounding_polyhedron()
{
    if ( bounding_polyhedron.size() == 0 ) {
        bounding_polyhedron.reserve( 12 );
        tmp_triangles_ptr.reserve( 4 );
    }
    AMP_DEBUG_ASSERT( !bounding_polyhedron_updated );
    clear_triangles_ptr( bounding_polyhedron );
    for ( unsigned int i = 0; i < 6; ++i ) {
        clear_triangles_ptr( tmp_triangles_ptr );
        //   3
        //    o
        //    | .
        //    |   .
        //    |     .
        //    |       .
        //    o--------o
        //   0          1
        tmp_triangles_ptr.push_back( new triangle_t( get_support_point( faces[4 * i + 0] ),
                                                     get_support_point( faces[4 * i + 1] ),
                                                     get_support_point( faces[4 * i + 3] ) ) );
        //   3          2
        //    o--------o
        //      .      |
        //        .    |
        //          .  |
        //            .|
        //             o
        //              1
        tmp_triangles_ptr.push_back( new triangle_t( get_support_point( faces[4 * i + 2] ),
                                                     get_support_point( faces[4 * i + 3] ),
                                                     get_support_point( faces[4 * i + 1] ) ) );
        //              2
        //             o
        //           . |
        //         .   |
        //       .     |
        //     .       |
        //    o--------o
        //   0          1
        tmp_triangles_ptr.push_back( new triangle_t( get_support_point( faces[4 * i + 1] ),
                                                     get_support_point( faces[4 * i + 2] ),
                                                     get_support_point( faces[4 * i + 0] ) ) );
        //   3          2
        //    o--------o
        //    |      .
        //    |    .
        //    |  .
        //    |.
        //    o
        //   0
        tmp_triangles_ptr.push_back( new triangle_t( get_support_point( faces[4 * i + 3] ),
                                                     get_support_point( faces[4 * i + 0] ),
                                                     get_support_point( faces[4 * i + 2] ) ) );

        // this might need scaling
        double tolerance = 0.0;
        tolerance += compute_distance_between_two_points( get_support_point( faces[4 * i + 0] ),
                                                          get_support_point( faces[4 * i + 1] ) );
        tolerance += compute_distance_between_two_points( get_support_point( faces[4 * i + 1] ),
                                                          get_support_point( faces[4 * i + 2] ) );
        tolerance += compute_distance_between_two_points( get_support_point( faces[4 * i + 2] ),
                                                          get_support_point( faces[4 * i + 3] ) );
        tolerance += compute_distance_between_two_points( get_support_point( faces[4 * i + 3] ),
                                                          get_support_point( faces[4 * i + 0] ) );
        tolerance *= 0.25e-12;
        if ( tmp_triangles_ptr[0]->above_point( tmp_triangles_ptr[2]->get_centroid(),
                                                tolerance ) ) {
            AMP_ASSERT( tmp_triangles_ptr[0]->above_point( tmp_triangles_ptr[3]->get_centroid(),
                                                           tolerance ) );
            AMP_ASSERT( tmp_triangles_ptr[1]->above_point( tmp_triangles_ptr[2]->get_centroid(),
                                                           tolerance ) );
            AMP_ASSERT( tmp_triangles_ptr[1]->above_point( tmp_triangles_ptr[3]->get_centroid(),
                                                           tolerance ) );
            // will fail if the four points are coplanar
            /*      AMP_ASSERT(!tmp_triangles[2].above_point(tmp_triangles[0].get_centroid()));
                    AMP_ASSERT(!tmp_triangles[2].above_point(tmp_triangles[1].get_centroid()));
                    AMP_ASSERT(!tmp_triangles[3].above_point(tmp_triangles[0].get_centroid()));
                    AMP_ASSERT(!tmp_triangles[3].above_point(tmp_triangles[1].get_centroid()));*/
            bounding_polyhedron.push_back( tmp_triangles_ptr[0] );
            bounding_polyhedron.push_back( tmp_triangles_ptr[1] );
            tmp_triangles_ptr[0] = nullptr;
            tmp_triangles_ptr[1] = nullptr;
        } else {
            /*
               AMP_ASSERT(!tmp_triangles[0].above_point(tmp_triangles[3].get_centroid()));
               AMP_ASSERT(!tmp_triangles[1].above_point(tmp_triangles[2].get_centroid()));
               AMP_ASSERT(!tmp_triangles[1].above_point(tmp_triangles[3].get_centroid()));*/
            AMP_ASSERT( tmp_triangles_ptr[2]->above_point( tmp_triangles_ptr[0]->get_centroid(),
                                                           tolerance ) );
            AMP_ASSERT( tmp_triangles_ptr[2]->above_point( tmp_triangles_ptr[1]->get_centroid(),
                                                           tolerance ) );
            AMP_ASSERT( tmp_triangles_ptr[3]->above_point( tmp_triangles_ptr[0]->get_centroid(),
                                                           tolerance ) );
            AMP_ASSERT( tmp_triangles_ptr[3]->above_point( tmp_triangles_ptr[1]->get_centroid(),
                                                           tolerance ) );
            bounding_polyhedron.push_back( tmp_triangles_ptr[2] );
            bounding_polyhedron.push_back( tmp_triangles_ptr[3] );
            tmp_triangles_ptr[2] = nullptr;
            tmp_triangles_ptr[3] = nullptr;
        } // end if
    }     // end for i
    bounding_polyhedron_updated = true;
}

bool hex8_element_t::within_bounding_polyhedron( double const *p, double tolerance )
{
    if ( !bounding_polyhedron_updated ) {
        build_bounding_polyhedron();
    }
    for ( unsigned int i = 0; i < 6; ++i ) {
        if ( ( !bounding_polyhedron[2 * i + 0]->above_point( p, tolerance ) ) ||
             ( !bounding_polyhedron[2 * i + 1]->above_point( p, tolerance ) ) ) {
            return false;
        }
    } // end for i
    return true;
}

void hex8_element_t::map_global_to_local( double const *global_coordinates,
                                          double *local_coordinates )
{
    std::copy(
        &( global_coordinates[0] ), &( global_coordinates[0] ) + 3, &( point_candidate[0] ) );
    solve_newton( &( local_coordinates[0] ) );
}

void hex8_element_t::map_local_to_global( double const *local_coordinates,
                                          double *global_coordinates )
{
    double tmp[3];
    std::copy( point_candidate, point_candidate + 3, tmp );
    std::fill( &( point_candidate[0] ), &( point_candidate[0] ) + 3, 0.0 );
    compute_residual_vector( &( local_coordinates[0] ), &( global_coordinates[0] ) );
    std::copy( &( tmp[0] ), &( tmp[0] ) + 3, &( point_candidate[0] ) );
}

bool hex8_element_t::contains_point( double const *coordinates,
                                     bool coordinates_are_local,
                                     double tolerance )
{
    std::vector<double> local_coordinates( 3 );
    if ( !coordinates_are_local ) {
        if ( !within_bounding_box( coordinates, tolerance ) ) {
            return false;
        }
        if ( !within_bounding_polyhedron( coordinates, tolerance ) ) {
            return false;
        }
        std::copy( &( coordinates[0] ), &( coordinates[0] ) + 3, &( point_candidate[0] ) );
        solve_newton( &( local_coordinates[0] ) );
    } else {
        std::copy( &( coordinates[0] ), &( coordinates[0] ) + 3, &( local_coordinates[0] ) );
    } // end if
    for ( unsigned int i = 0; i < 3; ++i ) {
        if ( fabs( local_coordinates[i] ) > 1.0 + tolerance ) {
            return false;
        } // end if
    }     // end for i
    return true;
}

void hex8_element_t::build_bounding_box()
{
    if ( bounding_box.size() == 0 ) {
        bounding_box.resize( 6 );
    }
    AMP_DEBUG_ASSERT( !bounding_box_updated );
    for ( unsigned int j = 0; j < 3; ++j ) {
        bounding_box[j + 0] = support_points[3 * 0 + j];
        bounding_box[j + 3] = support_points[3 * 0 + j];
    } // end for j
    for ( unsigned int i = 1; i < 8; ++i ) {
        for ( unsigned int j = 0; j < 3; ++j ) {
            if ( bounding_box[j + 0] > support_points[3 * i + j] ) {
                bounding_box[j + 0] = support_points[3 * i + j];
            } // end if
            if ( bounding_box[j + 3] < support_points[3 * i + j] ) {
                bounding_box[j + 3] = support_points[3 * i + j];
            } // end if
        }     // end for j
    }         // end for i
    bounding_box_updated = true;
}

void hex8_element_t::compute_residual_vector( double const *x, double *f )
{
    if ( basis_functions_values.size() == 0 ) {
        basis_functions_values.resize( 8 );
    }
    get_basis_functions_values( x, &( basis_functions_values[0] ) );
    for ( unsigned int i = 0; i < 3; ++i ) {
        f[i] = -point_candidate[i];
        for ( unsigned int j = 0; j < 8; ++j ) {
            f[i] += support_points[3 * j + i] * basis_functions_values[j];
        } // end for j
    }     // end for i
}

void hex8_element_t::compute_jacobian_matrix( double const *x, double *J )
{
    if ( basis_functions_derivatives.empty() ) {
        basis_functions_derivatives.resize( 24 );
    }
    get_basis_functions_derivatives( x, &( basis_functions_derivatives[0] ) );
    for ( unsigned int i = 0; i < 3; ++i ) {
        for ( unsigned int j = 0; j < 3; ++j ) {
            J[3 * i + j] = 0.0;
            for ( unsigned int k = 0; k < 8; ++k ) {
                J[3 * i + j] += support_points[3 * k + i] * basis_functions_derivatives[8 * j + k];
            } // end for k
        }     // end for j
    }         // end for i
}

double compute_inverse_3_by_3_matrix( double const *A, double *inverse_of_A )
{
    double determinant = A[0] * ( A[4] * A[8] - A[5] * A[7] ) -
                         A[1] * ( A[3] * A[8] - A[5] * A[6] ) +
                         A[2] * ( A[3] * A[7] - A[4] * A[6] );
    double one_over_determinant = 1.0 / determinant;
    inverse_of_A[0]             = one_over_determinant * ( A[4] * A[8] - A[5] * A[7] );
    inverse_of_A[1]             = one_over_determinant * ( A[2] * A[7] - A[1] * A[8] );
    inverse_of_A[2]             = one_over_determinant * ( A[1] * A[5] - A[2] * A[4] );
    inverse_of_A[3]             = one_over_determinant * ( A[5] * A[6] - A[3] * A[8] );
    inverse_of_A[4]             = one_over_determinant * ( A[0] * A[8] - A[2] * A[6] );
    inverse_of_A[5]             = one_over_determinant * ( A[2] * A[3] - A[0] * A[5] );
    inverse_of_A[6]             = one_over_determinant * ( A[3] * A[7] - A[4] * A[6] );
    inverse_of_A[7]             = one_over_determinant * ( A[1] * A[6] - A[0] * A[7] );
    inverse_of_A[8]             = one_over_determinant * ( A[0] * A[4] - A[1] * A[3] );
    return determinant;
}

void compute_n_by_n_matrix_times_vector( unsigned int n,
                                         double const *A,
                                         double const *x,
                                         double *b )
{
    for ( unsigned int i = 0; i < n; ++i ) {
        b[i] = 0.0;
        for ( unsigned int j = 0; j < n; ++j ) {
            b[i] += A[n * i + j] * x[j];
        } // end for j
    }     // end for i
}

void hex8_element_t::compute_initial_guess( double *initial_guess )
{
    if ( !center_of_element_data_updated ) {
        compute_center_of_element_data();
    }
    std::vector<double> tmp( 3 );
    make_vector_from_two_points(
        &( center_of_element_global_coordinates[0] ), &( point_candidate[0] ), &( tmp[0] ) );
    compute_n_by_n_matrix_times_vector(
        3, &( inverse_jacobian_matrix_at_center_of_element[0] ), &( tmp[0] ), initial_guess );
}


double hex8_element_t::solve_newton(
    double *x, double abs_tol, double rel_tol, unsigned int max_iter, bool verbose )
{
    if ( !memory_allocated_for_newton ) {
        residual_vector.resize( 3 );
        jacobian_matrix.resize( 9 );
        inverse_jacobian_matrix.resize( 9 );
        inverse_jacobian_matrix_times_residual_vector.resize( 3 );

        center_of_element_local_coordinates.resize( 3, 0.0 );
        center_of_element_global_coordinates.resize( 3 );
        jacobian_matrix_at_center_of_element.resize( 9 );
        inverse_jacobian_matrix_at_center_of_element.resize( 9 );

        basis_functions_values.resize( 8 );
        basis_functions_derivatives.resize( 24 );

        memory_allocated_for_newton = true;
    }

    //  std::fill(&(x[0]), &(x[0])+3, 0.0);
    compute_initial_guess( &( x[0] ) );

    compute_residual_vector( &x[0], &( residual_vector[0] ) );
    double residual_norm = std::sqrt( std::inner_product(
        residual_vector.begin(), residual_vector.end(), residual_vector.begin(), 0.0 ) );
    double tol           = abs_tol + rel_tol * residual_norm;

    for ( unsigned int iter = 0; iter < max_iter; ++iter ) {
        //    ++newton_count;
        if ( verbose ) {
            std::cout << iter << "  " << residual_norm << std::endl;
        }
        if ( residual_norm < tol ) {
            if ( verbose ) {
                std::cout << "converged at iteration " << iter << " with residual norm "
                          << residual_norm << "\n";
            }
            return residual_norm;
        } // end if
        compute_jacobian_matrix( &( x[0] ), &( jacobian_matrix[0] ) );
        double det = compute_inverse_3_by_3_matrix( &( jacobian_matrix[0] ),
                                                    &( inverse_jacobian_matrix[0] ) );
        AMP_ASSERT( !AMP::Utilities::approx_equal( det, 0.0, 1.0e-14 ) );
        compute_n_by_n_matrix_times_vector( 3,
                                            &( inverse_jacobian_matrix[0] ),
                                            &( residual_vector[0] ),
                                            &( inverse_jacobian_matrix_times_residual_vector[0] ) );
        bool line_search_passed = false;
        for ( double alpha = 1.0; alpha > 1.0e-14; alpha /= 2.0 ) {
            std::vector<double> tmp( 3 );
            for ( unsigned int i = 0; i < 3; ++i ) {
                tmp[i] = x[i] - alpha * inverse_jacobian_matrix_times_residual_vector[i];
            };
            compute_residual_vector( &( tmp[0] ), &( residual_vector[0] ) );
            double tmp_residual_norm = std::sqrt( std::inner_product(
                residual_vector.begin(), residual_vector.end(), residual_vector.begin(), 0.0 ) );
            if ( tmp_residual_norm < residual_norm ) {
                std::copy( tmp.begin(), tmp.end(), &( x[0] ) );
                residual_norm      = tmp_residual_norm;
                line_search_passed = true;
                break;
            } // end if
        }     // end for
        AMP_ASSERT( line_search_passed );
    } // end for
    std::cerr << "failed to converge with tolerance " << tol << " after " << max_iter - 1
              << " iterations (residual norm was " << residual_norm << ")" << std::endl;
    std::cerr << "support_points=\n";
    for ( unsigned int i = 0; i < 8; ++i ) {
        std::cerr << i << "  [" << support_points[3 * i] << ", " << support_points[3 * i + 1]
                  << ", " << support_points[3 * i + 2] << "]\n";
    } // end for i
    std::cerr << "bounding_box=\n";
    for ( unsigned int i = 0; i < 3; ++i ) {
        std::cerr << bounding_box[i] << "  " << bounding_box[i + 3] << "\n";
    }
    std::cerr << "point_candidate=[" << point_candidate[0] << ", " << point_candidate[1] << ", "
              << point_candidate[2] << "]\n";
    for ( unsigned int i = 0; i < 1000; ++i ) {
        std::cerr << std::flush;
    }
    AMP_ERROR( "Not Finished" );
    return 0.0;
}

void hex8_element_t::get_basis_functions_values( double const *x, double *basis_functions_values )
{
    basis_functions_values[0] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 - x[1] ) * ( 1.0 - x[2] );
    basis_functions_values[1] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 - x[1] ) * ( 1.0 - x[2] );
    basis_functions_values[2] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 + x[1] ) * ( 1.0 - x[2] );
    basis_functions_values[3] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 + x[1] ) * ( 1.0 - x[2] );
    basis_functions_values[4] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 - x[1] ) * ( 1.0 + x[2] );
    basis_functions_values[5] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 - x[1] ) * ( 1.0 + x[2] );
    basis_functions_values[6] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 + x[1] ) * ( 1.0 + x[2] );
    basis_functions_values[7] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 + x[1] ) * ( 1.0 + x[2] );
}

void hex8_element_t::get_basis_functions_derivatives( double const *x,
                                                      double *basis_functions_derivatives )
{
    basis_functions_derivatives[0] = 0.125 * ( -1.0 ) * ( 1.0 - x[1] ) * ( 1.0 - x[2] );
    basis_functions_derivatives[1] = 0.125 * ( +1.0 ) * ( 1.0 - x[1] ) * ( 1.0 - x[2] );
    basis_functions_derivatives[2] = 0.125 * ( +1.0 ) * ( 1.0 + x[1] ) * ( 1.0 - x[2] );
    basis_functions_derivatives[3] = 0.125 * ( -1.0 ) * ( 1.0 + x[1] ) * ( 1.0 - x[2] );
    basis_functions_derivatives[4] = 0.125 * ( -1.0 ) * ( 1.0 - x[1] ) * ( 1.0 + x[2] );
    basis_functions_derivatives[5] = 0.125 * ( +1.0 ) * ( 1.0 - x[1] ) * ( 1.0 + x[2] );
    basis_functions_derivatives[6] = 0.125 * ( +1.0 ) * ( 1.0 + x[1] ) * ( 1.0 + x[2] );
    basis_functions_derivatives[7] = 0.125 * ( -1.0 ) * ( 1.0 + x[1] ) * ( 1.0 + x[2] );

    basis_functions_derivatives[8]  = 0.125 * ( 1.0 - x[0] ) * ( -1.0 ) * ( 1.0 - x[2] );
    basis_functions_derivatives[9]  = 0.125 * ( 1.0 + x[0] ) * ( -1.0 ) * ( 1.0 - x[2] );
    basis_functions_derivatives[10] = 0.125 * ( 1.0 + x[0] ) * ( +1.0 ) * ( 1.0 - x[2] );
    basis_functions_derivatives[11] = 0.125 * ( 1.0 - x[0] ) * ( +1.0 ) * ( 1.0 - x[2] );
    basis_functions_derivatives[12] = 0.125 * ( 1.0 - x[0] ) * ( -1.0 ) * ( 1.0 + x[2] );
    basis_functions_derivatives[13] = 0.125 * ( 1.0 + x[0] ) * ( -1.0 ) * ( 1.0 + x[2] );
    basis_functions_derivatives[14] = 0.125 * ( 1.0 + x[0] ) * ( +1.0 ) * ( 1.0 + x[2] );
    basis_functions_derivatives[15] = 0.125 * ( 1.0 - x[0] ) * ( +1.0 ) * ( 1.0 + x[2] );

    basis_functions_derivatives[16] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 - x[1] ) * ( -1.0 );
    basis_functions_derivatives[17] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 - x[1] ) * ( -1.0 );
    basis_functions_derivatives[18] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 + x[1] ) * ( -1.0 );
    basis_functions_derivatives[19] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 + x[1] ) * ( -1.0 );
    basis_functions_derivatives[20] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 - x[1] ) * ( +1.0 );
    basis_functions_derivatives[21] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 - x[1] ) * ( +1.0 );
    basis_functions_derivatives[22] = 0.125 * ( 1.0 + x[0] ) * ( 1.0 + x[1] ) * ( +1.0 );
    basis_functions_derivatives[23] = 0.125 * ( 1.0 - x[0] ) * ( 1.0 + x[1] ) * ( +1.0 );
}

void hex8_element_t::project_on_face( unsigned int f,
                                      double const *local_coordinates,
                                      double *local_coordinates_on_face,
                                      double *shift_global_coordinates )
{
    AMP_DEBUG_ASSERT( f < 6 );
    std::vector<double> projection_local_coordinates( local_coordinates, local_coordinates + 3 );
    /*  if (f == 0) {
        projection_local_coordinates[2] = -1.0;
      } else if (f == 1) {
        projection_local_coordinates[1] = -1.0;
      } else if (f == 2) {
        projection_local_coordinates[0] = 1.0;
      } else if (f == 3) {
        projection_local_coordinates[1] = 1.0;
      } else if (f == 4) {
        projection_local_coordinates[0] = -1.0;
      } else if (f == 5) {
        projection_local_coordinates[2] = 1.0;
      } else {
        AMP_ERROR("comment en es-tu arrive la tres cher?");
      } // end if*/
    map_local_to_face( f, local_coordinates, local_coordinates_on_face );
    map_face_to_local( f, local_coordinates_on_face, &( projection_local_coordinates[0] ) );


    std::vector<double> projection_global_coordinates( 3 );
    map_local_to_global( &( projection_local_coordinates[0] ),
                         &( projection_global_coordinates[0] ) );
    std::vector<double> global_coordinates( 3 );
    map_local_to_global( local_coordinates, ( &global_coordinates[0] ) );
    make_vector_from_two_points( ( &global_coordinates[0] ),
                                 &( projection_global_coordinates[0] ),
                                 shift_global_coordinates );
    //  map_local_to_face(f, local_coordinates, local_coordinates_on_face);
}

void hex8_element_t::compute_normal_to_face( unsigned int f,
                                             double const *local_coordinates,
                                             double const *global_coordinates,
                                             double *normal_vector )
{
    AMP_DEBUG_ASSERT( f < 6 );
    double tangential_vectors[6];
    double const perturbation = 1.0e-6;
    double direction          = 1.0;
    double perturbated_local_coordinates_on_face[2];
    double perturbated_local_coordinates[3];
    double perturbated_global_coordinates[3];
    for ( unsigned int d = 0; d < 2; ++d ) {
        map_local_to_face( f, local_coordinates, perturbated_local_coordinates_on_face );
        if ( perturbated_local_coordinates_on_face[d] > 0.0 ) {
            perturbated_local_coordinates_on_face[d] -= perturbation;
            direction *= -1.0;
        } else {
            perturbated_local_coordinates_on_face[d] += perturbation;
        } // end if
        map_face_to_local(
            f, perturbated_local_coordinates_on_face, perturbated_local_coordinates );
        map_local_to_global( perturbated_local_coordinates, perturbated_global_coordinates );
        make_vector_from_two_points(
            global_coordinates, perturbated_global_coordinates, &( tangential_vectors[3 * d] ) );
    } // end for d
    compute_cross_product( &( tangential_vectors[0] ), &( tangential_vectors[3] ), normal_vector );
    normalize_vector( normal_vector );
    if ( direction == -1.0 ) {
        normal_vector[0] *= direction;
        normal_vector[1] *= direction;
        normal_vector[2] *= direction;
    } else {
        AMP_DEBUG_ASSERT( direction == 1.0 );
    }
}

void hex8_element_t::map_face_to_local( unsigned int f,
                                        double const *local_coordinates_on_face,
                                        double *local_coordinates )
{
    AMP_DEBUG_ASSERT( f < 6 );
    if ( f == 0 ) {
        local_coordinates[0] = local_coordinates_on_face[1];
        local_coordinates[1] = local_coordinates_on_face[0];
        local_coordinates[2] = -1.0;
    } else if ( f == 1 ) {
        local_coordinates[0] = local_coordinates_on_face[0];
        local_coordinates[1] = -1.0;
        local_coordinates[2] = local_coordinates_on_face[1];
    } else if ( f == 2 ) {
        local_coordinates[0] = 1.0;
        local_coordinates[1] = local_coordinates_on_face[0];
        local_coordinates[2] = local_coordinates_on_face[1];
    } else if ( f == 3 ) {
        local_coordinates[0] = -local_coordinates_on_face[0];
        local_coordinates[1] = 1.0;
        local_coordinates[2] = local_coordinates_on_face[1];
    } else if ( f == 4 ) {
        local_coordinates[0] = -1.0;
        local_coordinates[1] = -local_coordinates_on_face[0];
        local_coordinates[2] = local_coordinates_on_face[1];
    } else if ( f == 5 ) {
        local_coordinates[0] = local_coordinates_on_face[0];
        local_coordinates[1] = local_coordinates_on_face[1];
        local_coordinates[2] = 1.0;
    } else {
        std::cerr << "comment en es-tu arrive la tres cher?" << std::endl;
        AMP_ASSERT( false );
    } // end if
}

/*
void hex8_element_t::project_on_face(unsigned int f, double const *local_coordinates, double
*local_coordinates_on_face)
{
  std::cout<<"hex8_element_t::project_on_face(...) is deprecated, use
hex8_element_t::map_local_to_face(...) instead\n";
  map_local_to_face(f, local_coordinates, local_coordinates_on_face);
}
*/

void hex8_element_t::map_local_to_face( unsigned int f,
                                        double const *local_coordinates,
                                        double *local_coordinates_on_face )
{
    AMP_DEBUG_ASSERT( f < 6 );
    if ( f == 0 ) {
        local_coordinates_on_face[0] = local_coordinates[1];
        local_coordinates_on_face[1] = local_coordinates[0];
    } else if ( f == 1 ) {
        local_coordinates_on_face[0] = local_coordinates[0];
        local_coordinates_on_face[1] = local_coordinates[2];
    } else if ( f == 2 ) {
        local_coordinates_on_face[0] = local_coordinates[1];
        local_coordinates_on_face[1] = local_coordinates[2];
    } else if ( f == 3 ) {
        local_coordinates_on_face[0] = -local_coordinates[0];
        local_coordinates_on_face[1] = local_coordinates[2];
    } else if ( f == 4 ) {
        local_coordinates_on_face[0] = -local_coordinates[1];
        local_coordinates_on_face[1] = local_coordinates[2];
    } else if ( f == 5 ) {
        local_coordinates_on_face[0] = local_coordinates[0];
        local_coordinates_on_face[1] = local_coordinates[1];
    } else {
        std::cerr << "comment en es-tu arrive la tres cher?" << std::endl;
        AMP_ASSERT( false );
    } // end if
}

void hex8_element_t::get_basis_functions_values_on_face( double const *x, double *phi )
{
    phi[0] = 0.25 * ( 1.0 - x[0] ) * ( 1.0 - x[1] );
    phi[1] = 0.25 * ( 1.0 + x[0] ) * ( 1.0 - x[1] );
    phi[2] = 0.25 * ( 1.0 + x[0] ) * ( 1.0 + x[1] );
    phi[3] = 0.25 * ( 1.0 - x[0] ) * ( 1.0 + x[1] );
}

void hex8_element_t::get_local_coordinates_on_face( double const *phi, double *x )
{
    x[0] = 2.0 * ( phi[1] + phi[2] ) - 1.0;
    AMP_DEBUG_ASSERT( abs( 1.0 - 2.0 * ( phi[0] + phi[3] ) - x[0] ) < 1.0e-14 );
    x[1] = 1.0 - 2.0 * ( phi[0] + phi[1] );
    AMP_DEBUG_ASSERT( abs( 2.0 * ( phi[2] + phi[3] ) - 1.0 - x[1] ) < 1.0e-14 );
}

void hex8_element_t::get_normal_to_face( double const **support_points_ptr,
                                         double const *local_coordinates_on_face,
                                         double *normal_vector )
{
    double tangential_vectors[6];
    double const perturbation = 1.0e-6;
    double direction          = 1.0;
    double perturbated_local_coordinates_on_face[2], perturbated_global_coordinates[3];
    double global_coordinates[3];
    double basis_functions_values_on_face[4];
    get_basis_functions_values_on_face( local_coordinates_on_face, basis_functions_values_on_face );
    for ( unsigned int i = 0; i < 4; ++i ) {
        for ( unsigned int j = 0; j < 3; ++j ) {
            global_coordinates[j] += basis_functions_values_on_face[i] * support_points_ptr[i][j];
        } // end for j
    }     // end for i
    for ( unsigned int d = 0; d < 2; ++d ) {
        std::copy( local_coordinates_on_face,
                   local_coordinates_on_face + 2,
                   perturbated_local_coordinates_on_face );
        if ( perturbated_local_coordinates_on_face[d] > 0.0 ) {
            perturbated_local_coordinates_on_face[d] -= perturbation;
            direction *= -1.0;
        } else {
            perturbated_local_coordinates_on_face[d] += perturbation;
        } // end if
        get_basis_functions_values_on_face( perturbated_local_coordinates_on_face,
                                            basis_functions_values_on_face );
        std::fill( perturbated_global_coordinates, perturbated_global_coordinates + 3, 0.0 );
        for ( unsigned int i = 0; i < 4; ++i ) {
            for ( unsigned int j = 0; j < 3; ++j ) {
                perturbated_global_coordinates[j] +=
                    basis_functions_values_on_face[i] * support_points_ptr[i][j];
            } // end for j
        }     // end for i
        make_vector_from_two_points(
            global_coordinates, perturbated_global_coordinates, &( tangential_vectors[3 * d] ) );
    } // end for d
    compute_cross_product( &( tangential_vectors[0] ), &( tangential_vectors[3] ), normal_vector );
    normalize_vector( normal_vector );
    normal_vector[0] *= direction;
    normal_vector[1] *= direction;
    normal_vector[2] *= direction;
}

void hex8_element_t::compute_normal_to_face( unsigned int f,
                                             double const *local_coordinates_on_face,
                                             double *normal_vector )
{
    AMP_DEBUG_ASSERT( f < 6 );
    double tangential_vectors[6];
    double const perturbation = 1.0e-2;
    double direction          = 1.0;
    double perturbated_local_coordinates_on_face[2], perturbated_global_coordinates[3];
    double global_coordinates[3];
    double basis_functions_values_on_face[4];
    get_basis_functions_values_on_face( local_coordinates_on_face, basis_functions_values_on_face );
    std::fill( global_coordinates, global_coordinates + 3, 0.0 );
    for ( unsigned int i = 0; i < 4; ++i ) {
        for ( unsigned int j = 0; j < 3; ++j ) {
            global_coordinates[j] +=
                basis_functions_values_on_face[i] * support_points[3 * faces[4 * f + i] + j];
        } // end for j
    }     // end for i
    for ( unsigned int d = 0; d < 2; ++d ) {
        std::copy( local_coordinates_on_face,
                   local_coordinates_on_face + 2,
                   perturbated_local_coordinates_on_face );
        if ( perturbated_local_coordinates_on_face[d] > 0.0 ) {
            perturbated_local_coordinates_on_face[d] -= perturbation;
            direction *= -1.0;
        } else {
            perturbated_local_coordinates_on_face[d] += perturbation;
        } // end if
        get_basis_functions_values_on_face( perturbated_local_coordinates_on_face,
                                            basis_functions_values_on_face );
        std::fill( perturbated_global_coordinates, perturbated_global_coordinates + 3, 0.0 );
        for ( unsigned int i = 0; i < 4; ++i ) {
            for ( unsigned int j = 0; j < 3; ++j ) {
                perturbated_global_coordinates[j] +=
                    basis_functions_values_on_face[i] * support_points[3 * faces[4 * f + i] + j];
            } // end for j
        }     // end for i
        make_vector_from_two_points(
            global_coordinates, perturbated_global_coordinates, &( tangential_vectors[3 * d] ) );
    } // end for d
      //  normalize_vector(&(tangential_vectors[0]));
      //  normalize_vector(&(tangential_vectors[3]));
    compute_cross_product( &( tangential_vectors[0] ), &( tangential_vectors[3] ), normal_vector );
    normalize_vector( normal_vector );
    normal_vector[0] *= direction;
    normal_vector[1] *= direction;
    normal_vector[2] *= direction;
}

void hex8_element_t::compute_strain_tensor( double const *x, double const *u, double *epsilon )
{
    // ,x ,y ,z
    double nabla_phi[24];
    get_basis_functions_derivatives( x, nabla_phi );
    double J[9], invJ[9], tmp[3];
    compute_jacobian_matrix( x, J );
    compute_inverse_3_by_3_matrix( J, invJ );
    for ( unsigned int i = 0; i < 8; ++i ) {
        tmp[0] = nabla_phi[0 * 8 + i];
        tmp[1] = nabla_phi[1 * 8 + i];
        tmp[2] = nabla_phi[2 * 8 + i];
        // apply transpose inverse jacobian matrix
        nabla_phi[0 * 8 + i] = invJ[0] * tmp[0] + invJ[3] * tmp[1] + invJ[6] * tmp[2];
        nabla_phi[1 * 8 + i] = invJ[1] * tmp[0] + invJ[4] * tmp[1] + invJ[7] * tmp[2];
        nabla_phi[2 * 8 + i] = invJ[2] * tmp[0] + invJ[5] * tmp[1] + invJ[8] * tmp[2];
    } // end for i
    std::fill( epsilon, epsilon + 6, 0.0 );
    for ( unsigned int i = 0; i < 8; ++i ) {
        // xx yy zz yz xz xy
        epsilon[0] += u[3 * i + 0] * nabla_phi[0 * 8 + i];
        epsilon[1] += u[3 * i + 1] * nabla_phi[1 * 8 + i];
        epsilon[2] += u[3 * i + 2] * nabla_phi[2 * 8 + i];
        epsilon[3] += u[3 * i + 1] * nabla_phi[2 * 8 + i] + u[3 * i + 2] * nabla_phi[1 * 8 + i];
        epsilon[4] += u[3 * i + 0] * nabla_phi[2 * 8 + i] + u[3 * i + 2] * nabla_phi[0 * 8 + i];
        epsilon[5] += u[3 * i + 0] * nabla_phi[1 * 8 + i] + u[3 * i + 1] * nabla_phi[0 * 8 + i];
    } // end for i
}

void hex8_element_t::compute_rotation_tensor( double const *x, double const *u, double *omega )
{
    // ,x ,y ,z
    double nabla_phi[24];
    get_basis_functions_derivatives( x, nabla_phi );
    double J[9], invJ[9], tmp[3];
    compute_jacobian_matrix( x, J );
    compute_inverse_3_by_3_matrix( J, invJ );
    for ( unsigned int i = 0; i < 8; ++i ) {
        tmp[0] = nabla_phi[0 * 8 + i];
        tmp[1] = nabla_phi[1 * 8 + i];
        tmp[2] = nabla_phi[2 * 8 + i];
        // apply transpose inverse jacobian matrix
        nabla_phi[0 * 8 + i] = invJ[0] * tmp[0] + invJ[3] * tmp[1] + invJ[6] * tmp[2];
        nabla_phi[1 * 8 + i] = invJ[1] * tmp[0] + invJ[4] * tmp[1] + invJ[7] * tmp[2];
        nabla_phi[2 * 8 + i] = invJ[2] * tmp[0] + invJ[5] * tmp[1] + invJ[8] * tmp[2];
    } // end for i
    std::fill( omega, omega + 3, 0.0 );
    for ( unsigned int i = 0; i < 8; ++i ) {
        // yz xz xy
        omega[0] += u[3 * i + 1] * nabla_phi[2 * 8 + i] - u[3 * i + 2] * nabla_phi[1 * 8 + i];
        omega[1] += u[3 * i + 0] * nabla_phi[2 * 8 + i] - u[3 * i + 2] * nabla_phi[0 * 8 + i];
        omega[2] += u[3 * i + 0] * nabla_phi[1 * 8 + i] - u[3 * i + 1] * nabla_phi[0 * 8 + i];
    } // end for i
}

void compute_constitutive_matrix( double const E, double const nu, double *C )
{
    double const K = E / ( 3.0 * ( 1.0 - ( 2.0 * nu ) ) );
    double const G = E / ( 2.0 * ( 1.0 + nu ) );
    std::fill( C, C + 36, 0.0 );
    for ( size_t i = 0; i < 3; ++i ) {
        C[7 * i] += ( 2.0 * G );
    } // end for i
    for ( size_t i = 3; i < 6; ++i ) {
        C[7 * i] += G;
    } // end for i
    for ( size_t i = 0; i < 3; ++i ) {
        for ( size_t j = 0; j < 3; ++j ) {
            C[6 * i + j] += ( K - ( ( 2.0 * G ) / 3.0 ) );
        } // end for j
    }     // end for i
}

void compute_stress_tensor( double const *C, double const *epsilon, double *sigma )
{
    std::fill( sigma, sigma + 6, 0.0 );
    for ( size_t i = 0; i < 6; ++i ) {
        for ( size_t j = 0; j < 6; ++j ) {
            sigma[i] += ( C[6 * i + j] * epsilon[j] );
        } // end for j
    }     // end for i
}

void compute_traction( double const *sigma, double const *n, double *t )
{
    t[0] = sigma[0] * n[0] + sigma[5] * n[1] + sigma[4] * n[2];
    t[1] = sigma[5] * n[0] + sigma[1] * n[1] + sigma[3] * n[2];
    t[2] = sigma[4] * n[0] + sigma[3] * n[1] + sigma[2] * n[2];
}

double compute_von_mises_stress( double const *sigma )
{
    double sigma_v = std::sqrt(
        0.5 * ( std::pow( sigma[0] - sigma[1], 2 ) + std::pow( sigma[1] - sigma[2], 2 ) +
                std::pow( sigma[2] - sigma[0], 2 ) ) +
        3.0 * ( std::pow( sigma[3], 2 ) + std::pow( sigma[4], 2 ) + std::pow( sigma[5], 2 ) ) );
    return sigma_v;
}
