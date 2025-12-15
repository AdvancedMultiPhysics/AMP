#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/mesh/triangle_t.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>


void test_above_point( triangle_t *t_ptr, unsigned int n_random_candidate_points = 10000 )
{
    double const *centroid = t_ptr->get_centroid();
    double const *normal   = t_ptr->get_normal();
    double edges[9]        = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    make_vector_from_two_points(
        t_ptr->get_support_point_ptr( 0 ), t_ptr->get_support_point_ptr( 1 ), edges + 0 );
    make_vector_from_two_points(
        t_ptr->get_support_point_ptr( 1 ), t_ptr->get_support_point_ptr( 2 ), edges + 3 );
    make_vector_from_two_points(
        t_ptr->get_support_point_ptr( 2 ), t_ptr->get_support_point_ptr( 0 ), edges + 6 );
    double random_candidate_point[3] = { 0, 0, 0 }, random_motion_along_edges[3] = { 0, 0, 0 };
    double random_motion_along_normal;
    bool triangle_above_random_candidate_point;
    double tolerance = 1.0e-12;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -1, 1 );
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &random_motion_along_edge : random_motion_along_edges ) {
            random_motion_along_edge = dist( gen );
        }
        random_motion_along_normal            = dist( gen );
        triangle_above_random_candidate_point = !( random_motion_along_normal > tolerance );
        for ( unsigned int j = 0; j < 3; ++j ) {
            random_candidate_point[j] = centroid[j] + random_motion_along_normal * normal[j] +
                                        random_motion_along_edges[0] * edges[0 + j] +
                                        random_motion_along_edges[1] * edges[3 + j] +
                                        random_motion_along_edges[2] * edges[6 + j];
        }
        AMP_ASSERT( t_ptr->above_point( random_candidate_point, tolerance ) ==
                    triangle_above_random_candidate_point );
    } // end for i
}

void test_project_point( triangle_t *t_ptr, unsigned int n_random_candidate_points = 10000 )
{
    double const *triangle_centroid = t_ptr->get_centroid();
    double const *triangle_normal   = t_ptr->get_normal();
    double projection[3], projection_error[3];
    double random_candidate_point[3], random_motion_along_normal, random_motion_along_line;
    bool triangle_contains_random_candidate_point;
    double tolerance = 1.0e-12;
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -1, 1 );
    for ( unsigned int i = 0; i < 3; ++i ) {
        edge_t *e_ptr                = t_ptr->get_edge( i );
        double const *edge_normal    = e_ptr->get_normal();
        double const *edge_center    = e_ptr->get_center();
        double const *edge_direction = e_ptr->get_direction();
        AMP_ASSERT( fabs( compute_scalar_product( edge_direction, edge_normal ) ) < 1.0e-14 );
        AMP_ASSERT( fabs( compute_scalar_product( triangle_normal, edge_normal ) ) < 1.0e-14 );
        AMP_ASSERT( fabs( compute_scalar_product( triangle_normal, edge_direction ) ) < 1.0e-14 );
        double triangle_centroid_to_edge_center_line[3];
        make_vector_from_two_points(
            triangle_centroid, edge_center, triangle_centroid_to_edge_center_line );
        // bravo!
        double line_length =
            compute_scalar_product( triangle_centroid_to_edge_center_line, edge_normal );
        for ( unsigned int k = 0; k < 3; ++k ) {
            triangle_centroid_to_edge_center_line[k] = line_length * edge_normal[k];
        }
        for ( unsigned int j = 0; j < n_random_candidate_points; ++j ) {
            random_motion_along_normal = dist( gen );
            random_motion_along_line   = dist( gen );
            for ( unsigned int k = 0; k < 3; ++k ) {
                random_candidate_point[k] =
                    edge_center[k] +
                    random_motion_along_line * triangle_centroid_to_edge_center_line[k] +
                    random_motion_along_normal * triangle_normal[k];
            }
            triangle_contains_random_candidate_point = ( random_motion_along_line < tolerance );
            AMP_ASSERT( e_ptr->above_point( random_candidate_point, tolerance ) ==
                        triangle_contains_random_candidate_point );
            AMP_ASSERT( t_ptr->contains_point( random_candidate_point, tolerance ) ==
                        triangle_contains_random_candidate_point );
            AMP_ASSERT( ( t_ptr->project_point( random_candidate_point, projection, tolerance ) ==
                          -1 ) == triangle_contains_random_candidate_point );
            if ( !triangle_contains_random_candidate_point ) {
                for ( unsigned int k = 0; k < 3; ++k ) {
                    projection_error[k] = projection[k] - edge_center[k];
                }
            } else {
                for ( unsigned int k = 0; k < 3; ++k ) {
                    projection_error[k] =
                        projection[k] - edge_center[k] -
                        random_motion_along_line * triangle_centroid_to_edge_center_line[k];
                }
            } // end for if
            AMP_ASSERT( compute_vector_norm( projection_error ) < 1.0e-14 );
        } // end for j
    }     // end for i

    // this might fail for very obtus triangles...
    for ( unsigned int i = 0; i < 3; ++i ) {
        double triangle_centroid_to_support_point_line[3];
        double const *triangle_support_point = t_ptr->get_support_point_ptr( i );
        make_vector_from_two_points(
            triangle_centroid, triangle_support_point, triangle_centroid_to_support_point_line );
        for ( unsigned int j = 0; j < n_random_candidate_points; ++j ) {
            random_motion_along_normal = dist( gen );
            random_motion_along_line   = dist( gen );
            for ( unsigned int k = 0; k < 3; ++k ) {
                random_candidate_point[k] =
                    triangle_support_point[k] +
                    random_motion_along_line * triangle_centroid_to_support_point_line[k] +
                    random_motion_along_normal * triangle_normal[k];
            }
            triangle_contains_random_candidate_point = ( random_motion_along_line < tolerance );
            AMP_ASSERT( t_ptr->contains_point( random_candidate_point, tolerance ) ==
                        triangle_contains_random_candidate_point );
            AMP_ASSERT( ( t_ptr->project_point( random_candidate_point, projection, tolerance ) ==
                          -1 ) == triangle_contains_random_candidate_point );
            if ( !triangle_contains_random_candidate_point ) {
                for ( unsigned int k = 0; k < 3; ++k ) {
                    projection_error[k] = projection[k] - triangle_support_point[k];
                }
            } else {
                for ( unsigned int k = 0; k < 3; ++k ) {
                    projection_error[k] =
                        projection[k] - triangle_support_point[k] -
                        random_motion_along_line * triangle_centroid_to_support_point_line[k];
                }
            } // end for if
            AMP_ASSERT( compute_vector_norm( projection_error ) < 1.0e-14 );
        } // end for j
    }     // end for i
}

void test_return_status( triangle_t *t_ptr )
{
    int status;
    double projection[3];
    double tolerance = 1.0e-12;

    for ( unsigned int i = 0; i < 3; ++i ) {
        edge_t *e_ptr             = t_ptr->get_edge( i );
        double const *edge_center = e_ptr->get_center();
        status                    = e_ptr->project_point( edge_center, projection, tolerance );
        AMP_ASSERT( status == 2 );
        for ( unsigned int j = 0; j < 2; ++j ) {
            double const *edge_support_point = e_ptr->get_support_point_ptr( j );
            status = e_ptr->project_point( edge_support_point, projection, tolerance );
            AMP_ASSERT( status == static_cast<signed int>( j ) );
        } // end for j
    }     // end for i

    double const *triangle_centroid = t_ptr->get_centroid();
    status = t_ptr->project_point( triangle_centroid, projection, tolerance );
    AMP_ASSERT( status == -1 );
    for ( unsigned int i = 0; i < 3; ++i ) {
        edge_t *e_ptr             = t_ptr->get_edge( i );
        double const *edge_center = e_ptr->get_center();
        status                    = t_ptr->project_point( edge_center, projection, tolerance );
        AMP_ASSERT( status == static_cast<signed int>( 3 + i ) );
    } // end for i
    for ( unsigned int i = 0; i < 3; ++i ) {
        double const *triangle_support_point = t_ptr->get_support_point_ptr( i );
        status = t_ptr->project_point( triangle_support_point, projection, tolerance );
        AMP_ASSERT( status == static_cast<signed int>( i ) );
    } // end for i
}

void test_project_and_orthogonalize()
{
    double direction[3] = { 1.0, 0.0, 0.0 };
    double vector[3]    = { 1.0, 1.0, 1.0 };
    double tmp[3];
    std::copy( vector, vector + 3, tmp );
    project_vector_onto_direction( direction, tmp );
    AMP_ASSERT( std::equal( tmp, tmp + 3, direction ) );
    orthogonalize_vector_against_direction( direction, tmp );
    AMP_ASSERT( compute_vector_norm( tmp ) == 0.0 );
    std::copy( vector, vector + 3, tmp );
    orthogonalize_vector_against_direction( direction, tmp );
    AMP_ASSERT( compute_scalar_product( tmp, direction ) == 0.0 );
    project_vector_onto_direction( direction, tmp );
    AMP_ASSERT( compute_vector_norm( tmp ) == 0.0 );
}

void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    const double pi  = 3.141592653589793;
    double points[9] = {
        0.0, 0.0, 0.0, // 0
        1.0, 0.0, 0.0, // 1
        0.0, 1.0, 0.0, // 2
    };

    double scaling_factors[3] = { 4.0, 2.0, 1.0 };
    scale_points( scaling_factors, 3, points );

    double translation_vector[3] = { 3.0, 1.0, 5.0 };
    translate_points( translation_vector, 3, points );

    rotate_points( 2, pi / 3.0, 3, points );

    rotate_points( 0, 0.75 * pi, 3, points );

    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -0.1, 0.1 );
    for ( auto &point : points ) {
        point += dist( gen );
    }

    triangle_t triangle( points, points + 3, points + 6 );

    test_above_point( &triangle );

    test_project_point( &triangle );

    test_return_status( &triangle );

    test_project_and_orthogonalize();

    ut->passes( exeName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testTriangleProjection";

    myTest( &ut, exeName );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
