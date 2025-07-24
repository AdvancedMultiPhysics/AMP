#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/mesh/hex8_element_t.h"
#include "AMP/mesh/testHelpers/latex_visualization_tools.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"

#include <fstream>


class soft_equal_to
{
public:
    explicit soft_equal_to( double const tol = 1.0e-15 ) : tolerance( tol ){};
    inline bool operator()( double const x, double const y )
    {
        return std::abs( x - y ) < tolerance;
    };

private:
    double tolerance;
};


unsigned int test_mapping_global_to_local( hex8_element_t *volume_element,
                                           unsigned int n_random_candidate_points = 10000,
                                           double abs_tol                         = 1.0e-12,
                                           double rel_tol                         = 1.0e-12 )
{
    double local_coordinates[3], global_coordinates[3], computed_local_coordinates[3], tol,
        error_vector[3];
    unsigned int count_tests_failing = 0;
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &local_coordinate : local_coordinates ) {
            local_coordinate = -1.0 + 2.0 * rand() / RAND_MAX;
        }
        volume_element->map_local_to_global( local_coordinates, global_coordinates );
        volume_element->map_global_to_local( global_coordinates, computed_local_coordinates );
        make_vector_from_two_points( computed_local_coordinates, local_coordinates, error_vector );
        tol = abs_tol + rel_tol * compute_vector_norm( local_coordinates );
        if ( compute_vector_norm( error_vector ) > tol ) {
            ++count_tests_failing;
        }
    } // end for i
    return count_tests_failing;
}

void test_mapping_face_to_local( hex8_element_t *volume_element,
                                 unsigned int n_random_candidate_points = 10000,
                                 double tol                             = 1.0e-16 )
{
    double local_coordinates_on_face[3], local_coordinates[3],
        computed_local_coordinates_on_face[3], computed_shift[3], error_vector[3];
    local_coordinates_on_face[2]          = 0.0;
    computed_local_coordinates_on_face[2] = 0.0;
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( unsigned int j = 0; j < 2; ++j ) {
            local_coordinates_on_face[j] = -1.0 + 2.0 * rand() / RAND_MAX;
        }
        for ( unsigned int f = 0; f < 6; ++f ) {
            volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
            volume_element->project_on_face(
                f, local_coordinates, computed_local_coordinates_on_face, computed_shift );
            AMP_ASSERT( compute_vector_norm( computed_shift ) < tol );
            make_vector_from_two_points(
                computed_local_coordinates_on_face, local_coordinates_on_face, error_vector );
            AMP_ASSERT( compute_vector_norm( error_vector ) < tol );
        } // end for f
    } // end for i
}

void test_basis_functions_values_on_face( hex8_element_t *volume_element,
                                          unsigned int n_random_candidate_points = 10000,
                                          double tol                             = 1.0e-16 )
{
    double local_coordinates_on_face[2], local_coordinates[3], basis_functions_values_on_face[4],
        basis_functions_values[8];
    unsigned int const *face_ordering;
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &elem : local_coordinates_on_face ) {
            elem = -1.0 + 2.0 * rand() / RAND_MAX;
        }
        for ( unsigned int f = 0; f < 6; ++f ) {
            face_ordering = hex8_element_t::get_face( f );
            volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
            hex8_element_t::get_basis_functions_values( local_coordinates, basis_functions_values );
            hex8_element_t::get_basis_functions_values_on_face( local_coordinates_on_face,
                                                                basis_functions_values_on_face );
            for ( unsigned int v = 0; v < 4; ++v ) {
                AMP_ASSERT( std::abs( basis_functions_values_on_face[v] -
                                      basis_functions_values[face_ordering[v]] ) < tol );
            } // end for v
        } // end for f
    } // end for i
}

void test_mapping_basis_functions_values_to_local_coordinates_on_face(
    unsigned int n_random_candidate_points = 1000, [[maybe_unused]] double tol = 1.0e-16 )
{
    double local_coordinates_on_face[2], basis_functions_values_on_face[4],
        computed_local_coordinates_on_face[2];
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( auto &elem : local_coordinates_on_face ) {
            elem = -1.0 + 2.0 * rand() / RAND_MAX;
        }
        hex8_element_t::get_basis_functions_values_on_face( local_coordinates_on_face,
                                                            basis_functions_values_on_face );
        hex8_element_t::get_local_coordinates_on_face( basis_functions_values_on_face,
                                                       computed_local_coordinates_on_face );
        AMP_ASSERT( std::equal( local_coordinates_on_face,
                                local_coordinates_on_face + 2,
                                computed_local_coordinates_on_face,
                                soft_equal_to() ) );
    } // end for i
}

void draw_axis( unsigned int a,
                hex8_element_t *volume_element,
                const std::string &axis_name,
                std::ostream &os )
{
    double local_coordinates[3] = { 0.0, 0.0, 0.0 };
    unsigned int n              = 10;
    std::vector<double> global_coordinates( 3 * ( n + 1 ) );
    for ( unsigned int i = 0; i <= n; ++i ) {
        local_coordinates[a] = static_cast<double>( i ) / static_cast<double>( n );
        volume_element->map_local_to_global( local_coordinates, &( global_coordinates[3 * i] ) );
    } // end for
    draw_line( n + 1, &global_coordinates[0], os, "black,->" );
    draw_point( &global_coordinates[3 * n], os, "black", axis_name );
}

void draw_lines_on_triangle( triangle_t *t, std::ostream &os )
{
    unsigned int n = 6;
    double const *p[3];
    for ( unsigned int k = 0; k < 3; ++k ) {
        p[0] = t->get_support_point_ptr( ( 0 + k ) % 3 );
        p[1] = t->get_support_point_ptr( ( 1 + k ) % 3 );
        p[2] = t->get_support_point_ptr( ( 2 + k ) % 3 );
        double b[3], e[3];
        for ( unsigned int i = 0; i < n; ++i ) {
            for ( unsigned int j = 0; j < 3; ++j ) {
                b[j] = p[1][j] +
                       ( p[0][j] - p[1][j] ) * static_cast<double>( i ) / static_cast<double>( n );
                e[j] = p[2][j] +
                       ( p[0][j] - p[2][j] ) * static_cast<double>( i ) / static_cast<double>( n );
            } // end for j
            draw_line( b, e, os );
        } // end for i
    } // end for k
}

void draw_lines_on_face( unsigned int f, hex8_element_t *volume_element, std::ostream &os )
{
    double local_coordinates_on_face[2], local_coordinates[3];
    unsigned int n = 6, m = 6;
    double global_coordinates[21] = { 0 };
    for ( unsigned int i = 0; i <= n; ++i ) {
        local_coordinates_on_face[0] =
            -1.0 + 2.0 * static_cast<double>( i ) / static_cast<double>( n );
        for ( unsigned int j = 0; j <= m; ++j ) {
            local_coordinates_on_face[1] =
                -1.0 + 2.0 * static_cast<double>( j ) / static_cast<double>( m );
            volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
            volume_element->map_local_to_global( local_coordinates, &global_coordinates[3 * j] );
        } // end for j
        draw_line( m + 1, global_coordinates, os, "black, dashed" );
    } // end for i
    for ( unsigned int j = 0; j <= m; ++j ) {
        local_coordinates_on_face[1] =
            -1.0 + 2.0 * static_cast<double>( j ) / static_cast<double>( m );
        for ( unsigned int i = 0; i <= n; ++i ) {
            local_coordinates_on_face[0] =
                -1.0 + 2.0 * static_cast<double>( i ) / static_cast<double>( n );
            volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
            volume_element->map_local_to_global( local_coordinates, &global_coordinates[3 * i] );
        } // end for j
        draw_line( n + 1, global_coordinates, os, "black, dashed" );
    } // end for j
}

void draw_tetrahedron( unsigned int f, hex8_element_t *volume_element, std::ostream &os )
{
    unsigned int const *faces = volume_element->get_faces();
    std::vector<double const *> p;
    p.push_back( volume_element->get_support_point( faces[4 * f + 0] ) );
    p.push_back( volume_element->get_support_point( faces[4 * f + 1] ) );
    p.push_back( volume_element->get_support_point( faces[4 * f + 2] ) );
    p.push_back( volume_element->get_support_point( faces[4 * f + 3] ) );
    std::vector<triangle_t> t;
    t.emplace_back( p[0], p[1], p[3] );
    t.emplace_back( p[2], p[3], p[1] );
    t.emplace_back( p[1], p[2], p[0] );
    t.emplace_back( p[3], p[0], p[2] );
    std::string option;
    std::vector<bool> b( 4, false );
    if ( t[0].above_point( t[2].get_centroid() ) ) {
        AMP_ASSERT( t[0].above_point( t[3].get_centroid() ) );
        AMP_ASSERT( t[1].above_point( t[2].get_centroid() ) );
        AMP_ASSERT( t[1].above_point( t[3].get_centroid() ) );
        b[0] = true;
        b[1] = true;
    } else {
        AMP_ASSERT( t[2].above_point( t[0].get_centroid() ) );
        AMP_ASSERT( t[2].above_point( t[1].get_centroid() ) );
        AMP_ASSERT( t[3].above_point( t[0].get_centroid() ) );
        AMP_ASSERT( t[3].above_point( t[1].get_centroid() ) );
        b[2] = true;
        b[3] = true;
    } // end if
    for ( unsigned int i = 0; i < 4; ++i ) {
        option = "fill=none";
        if ( b[i] ) {
            option.append( ",dotted" );
        } // end if
        draw_triangle( &t[i], os, option );
    } // end for i
    double local_coordinates_on_face[2] = { 0.0, 0.0 };
    double local_coordinates[3], global_coordinates[3];
    volume_element->map_face_to_local( f, local_coordinates_on_face, local_coordinates );
    volume_element->map_local_to_global( local_coordinates, global_coordinates );
    draw_point( global_coordinates, os, "red", "+" );
    os << "% \n";
    draw_lines_on_triangle( &t[0], os );
    os << "% \n";
    draw_lines_on_triangle( &t[1], os );
    os << "% \n";
    draw_lines_on_triangle( &t[2], os );
    os << "% \n";
    draw_lines_on_triangle( &t[3], os );
    os << "% \n";
}

void testHex8ElementMapping( AMP::UnitTest &ut )
{
    std::ofstream os;
    os.open( "testHex8ElementMapping.txt" );

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

    std::string labels_num[8] = { "0", "1", "2", "3", "4", "5", "6", "7" };

    hex8_element_t volume_element( points );
    draw_lines_on_face( 2, &volume_element, os );

    // shifting the points from [-1, 1]^3 to [0, 1]^3
    for ( auto &point : points ) {
        point = 0.5 * ( point + 1.0 );
    }

    volume_element.set_support_points( points );
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    test_mapping_face_to_local( &volume_element );

    test_basis_functions_values_on_face( &volume_element );

    test_mapping_basis_functions_values_to_local_coordinates_on_face();

    double scaling_factors[3] = { 4.0, 2.0, 1.0 };
    scale_points( scaling_factors, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    double translation_vector[3] = { 3.0, 1.0, 5.0 };
    translate_points( translation_vector, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    rotate_points( 2, pi / 3.0, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    rotate_points( 0, 0.75 * pi, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    srand( 0 );
    for ( auto &point : points ) {
        point += -0.1 + 0.8 * rand() / RAND_MAX;
    }

    volume_element.set_support_points( points );

    //  test_mapping(&volume_element, 1000000);
    AMP_ASSERT( test_mapping_global_to_local( &volume_element ) == 0 );

    translate_points( 0, 2.0, 8, points );
    translate_points( 1, 12.0, 8, points );
    translate_points( 2, 4.0, 8, points );
    volume_element.set_support_points( points );

    for ( unsigned int i = 0; i < 8; ++i ) {
        draw_point( volume_element.get_support_point( i ), os, "black", labels_num[i] );
    }

    draw_tetrahedron( 2, &volume_element, os );

    for ( unsigned int i = 0; i < 8; ++i ) {
        draw_point( volume_element.get_support_point( i ), os, "red" );
    }
    double point_of_view[3] = { 1.0, 1.0, 1.0 };

    os << "% volume element\n";
    draw_hex8_element( &volume_element, point_of_view, os );

    os << "% bounding polyhedron\n";
    draw_bounding_polyhedron( &volume_element, point_of_view, os );

    os << "% bounding box\n";
    draw_bounding_box( &volume_element, point_of_view, os );

    os.close();
    ut.passes( "testHex8ElementMapping" );
}
