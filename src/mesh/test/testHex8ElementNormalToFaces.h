#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/mesh/hex8_element_t.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"

#include <cmath>
#include <iostream>
#include <random>


void test_normal( hex8_element_t *volume_element, unsigned int n_random_candidate_points = 20 )
{
    double normal_vector[3];
    double local_coordinates_on_face[2], local_coordinates[3], global_coordinates[3];
    double local_coordinates_on_face_check[2];
    double const *face_support_points_ptr[4];
    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( -1, 1 );
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &elem : local_coordinates_on_face ) {
            elem = dist( gen );
        }
        for ( unsigned int f = 0; f < 6; ++f ) {
            for ( unsigned int v = 0; v < 4; ++v ) {
                face_support_points_ptr[v] =
                    volume_element->get_support_point( volume_element->get_face( f )[v] );
            }
            volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
            volume_element->map_local_to_face(
                f, local_coordinates, local_coordinates_on_face_check );
            AMP_ASSERT( std::equal( local_coordinates_on_face,
                                    local_coordinates_on_face + 2,
                                    local_coordinates_on_face_check ) );
            volume_element->map_local_to_global( local_coordinates, global_coordinates );
            volume_element->compute_normal_to_face(
                f, local_coordinates, global_coordinates, normal_vector );
            std::cout << f << " { ";
            for ( auto &elem : normal_vector ) {
                std::cout << elem << " ";
            }
            std::cout << "}  ";
            volume_element->compute_normal_to_face( f, local_coordinates, normal_vector );
            std::cout << f << " { ";
            for ( auto &elem : normal_vector ) {
                std::cout << elem << " ";
            }
            std::cout << "}  ";
            volume_element->get_normal_to_face(
                face_support_points_ptr, local_coordinates_on_face, normal_vector );
            std::cout << f << " { ";
            for ( auto &elem : normal_vector ) {
                std::cout << elem << " ";
            }
            std::cout << "}\n";
        } // end for f
    } // end for i
}

inline bool soft_equal_to( double x, double y ) { return fabs( x - y ) < 1.0e-15; }

void test_recovering_local_coordinates_on_face_from_basis_functions_values(
    unsigned int n_random_candidate_points = 1000 )
{
    double x[2], x_prime[2], phi[4];
    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( -1, 1 );
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &elem : x ) {
            elem = dist( gen );
        } // end for j
        hex8_element_t::get_basis_functions_values_on_face( x, phi );
        hex8_element_t::get_local_coordinates_on_face( phi, x_prime );
        AMP_ASSERT( std::equal( x, x + 2, x_prime, soft_equal_to ) );
    } // end for i
}

unsigned int perform_battery_of_tests( hex8_element_t *volume_element,
                                       double const *normal_to_faces          = nullptr,
                                       unsigned int n_random_candidate_points = 1000,
                                       double tolerance                       = 1.0e-12 )
{

    double local_coordinates_on_face[2], computed_normal_vector[3], error_vector[3],
        error_vector_norm;
    unsigned int count_tests_failing = 0;
    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( -1, 1 );
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &elem : local_coordinates_on_face ) {
            elem = dist( gen );
        }
        for ( unsigned int f = 0; f < 6; ++f ) {
            volume_element->compute_normal_to_face(
                f, local_coordinates_on_face, computed_normal_vector );
            if ( normal_to_faces != nullptr ) {
                //        std::transform(computed_normal_vector, computed_normal_vector+3,
                //        normal_to_faces+3*f,
                //        error_vector, std::minus<double>());
                make_vector_from_two_points(
                    computed_normal_vector, normal_to_faces + 3 * f, error_vector );
                error_vector_norm = compute_vector_norm( error_vector );
                if ( error_vector_norm > tolerance ) {
                    ++count_tests_failing;

                    std::cout << error_vector_norm << "  ";
                    //        if (!std::equal(computed_normal_vector, computed_normal_vector+3,
                    //        normal_to_faces+3*f)) {
                    //        ++count_tests_failing;// }
                    std::cout << i << "  " << f << "  ";
                    std::cout << "{ ";
                    for ( auto &elem : computed_normal_vector ) {
                        std::cout << elem << " ";
                    }
                    std::cout << "} ";
                    std::cout << "{ ";
                    for ( unsigned int d = 0; d < 3; ++d ) {
                        std::cout << normal_to_faces[3 * f + d] << " ";
                    }
                    std::cout << "} ";
                    std::cout << local_coordinates_on_face[0] << " "
                              << local_coordinates_on_face[1];
                    std::cout << "\n" << std::flush;
                }
            } // end if
        } // end for f

        //    tol = tol_abs+tol_rel*fabs(interpolated_value);
        //    if (interpolation_error > tol) { ++count_tests_failing; }
    } // end for i
    return count_tests_failing;
}

void testHex8ElementNormalToFaces( AMP::UnitTest &ut )
{
    const double pi   = 3.141592653589793;
    double points[24] = {
        -1.0, -1.0, -1.0, // 0
        +1.0, -1.0, -1.0, // 1
        +1.0, +1.0, -1.0, // 2
        -1.0, +1.0, -1.0, // 3
        -1.0, -1.0, +1.0, // 4
        +1.0, -1.0, +1.0, // 5
        +1.0, +1.0, +1.0, // 6
        -1.0, +1.0, +1.0  // 7
    };
    double normal_to_faces[18] = {
        0.0,  0.0,  -1.0, // 0
        0.0,  -1.0, 0.0,  // 1
        +1.0, 0.0,  0.0,  // 2
        0.0,  +1.0, 0.0,  // 3
        -1.0, 0.0,  0.0,  // 4
        0.0,  0.0,  +1.0, // 5
    };

    hex8_element_t volume_element( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, normal_to_faces ) == 0 );
    //  test_normal(&volume_element);

    double scaling_factors[3] = { 4.0, 2.0, 1.0 };
    scale_points( scaling_factors, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, normal_to_faces ) == 0 );

    double translation_vector[3] = { 3.0, 1.0, 5.0 };
    translate_points( translation_vector, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, normal_to_faces ) == 0 );

    rotate_points( 2, pi / 2.0, 8, points );
    rotate_points( 2, pi / 2.0, 6, normal_to_faces );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, normal_to_faces ) == 0 );

    rotate_points( 0, -0.75 * pi, 8, points );
    rotate_points( 0, -0.75 * pi, 6, normal_to_faces );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, normal_to_faces ) == 0 );

    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( -0.1, 0.1 );
    for ( auto &point : points )
        point += dist( gen );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element ) ==
                0 ); // isn't actually testing anything

    test_recovering_local_coordinates_on_face_from_basis_functions_values();

    ut.passes( "testHex8ElementNormalToFaces" );
}
