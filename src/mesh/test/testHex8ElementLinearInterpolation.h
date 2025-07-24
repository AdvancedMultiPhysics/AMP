#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/mesh/hex8_element_t.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/UtilityMacros.h"

#include <cmath>
#include <functional>
#include <iostream>


double my_function( double const *xyz )
{
    double x = xyz[0], y = xyz[1], z = xyz[2];
    return ( 1.0 + 6.0 * x ) * ( 2.0 - 5.0 * y ) * ( 3.0 + 4.0 * z );
}

double my_function_no_cross_terms( double const *xyz )
{
    double x = xyz[0], y = xyz[1], z = xyz[2];
    return 1.0 + 6.0 * x - 5.0 * y + 4.0 * z;
}

unsigned int perform_battery_of_tests( hex8_element_t *volume_element,
                                       std::function<double( const double * )> fun,
                                       unsigned int n_random_candidate_points = 1000,
                                       double tol_abs                         = 1.0e-12,
                                       double tol_rel                         = 1.0e-12 )
{
    std::vector<double> f_at_support_points( 8 );
    for ( unsigned int i = 0; i < 8; ++i )
        f_at_support_points[i] = fun( volume_element->get_support_point( i ) );


    std::vector<double> local_coordinates( 3 ), global_coordinates( 3 );
    std::vector<double> basis_functions_values( 8 );
    double interpolated_value, f_at_candidate_point, interpolation_error, tol;
    unsigned int count_tests_failing = 0;
    for ( unsigned int i = 0; i < n_random_candidate_points; ++i ) {
        for ( unsigned int j = 0; j < 3; ++j )
            local_coordinates[j] = -1.0 + 2.0 * rand() / RAND_MAX;

        hex8_element_t::get_basis_functions_values( local_coordinates.data(),
                                                    basis_functions_values.data() );
        interpolated_value = 0.0;
        for ( unsigned int j = 0; j < 8; ++j )
            interpolated_value += f_at_support_points[j] * basis_functions_values[j];

        volume_element->map_local_to_global( local_coordinates.data(), global_coordinates.data() );
        f_at_candidate_point = fun( global_coordinates.data() );
        interpolation_error  = fabs( interpolated_value - f_at_candidate_point );
        tol                  = tol_abs + tol_rel * fabs( interpolated_value );
        if ( interpolation_error > tol )
            ++count_tests_failing;
    }
    return count_tests_failing;
}

void testHex8ElementLinearInterpolation( AMP::UnitTest &ut )
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

    hex8_element_t volume_element( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) == 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );
    srand( 0 );

    double scaling_factors[3] = { 4.0, 2.0, 1.0 };
    scale_points( scaling_factors, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) == 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    double translation_vector[3] = { 3.0, 1.0, 5.0 };
    translate_points( translation_vector, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) == 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    rotate_points( 2, pi / 2.0, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) == 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    rotate_points( 0, -0.75 * pi, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) > 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    rotate_points( 0, 0.75 * pi, 8, points );
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) == 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    for ( auto &point : points ) {
        point += -0.1 + 0.2 * rand() / RAND_MAX;
    }
    volume_element.set_support_points( points );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function ) > 0 );
    AMP_ASSERT( perform_battery_of_tests( &volume_element, my_function_no_cross_terms ) == 0 );

    ut.passes( "testHex8ElementLinearInterpolation" );
}
