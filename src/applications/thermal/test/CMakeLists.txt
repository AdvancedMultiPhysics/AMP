# Copy input files
COPY_TEST_FILE( input_test_thermal )
COPY_TEST_FILE( input_test_thermal2 )
COPY_TEST_FILE( input_test_thermal3 )
COPY_TEST_FILE( input_test_thermal_impedance )
COPY_TEST_FILE( input_YbYAG )

# Add tests
ADD_AMP_PROVISIONAL_TEST( test_thermal )
IF ( USE_LIBMESH AND USE_PETSC AND USE_TRILINOS )
    ADD_AMP_TEST( test_thermal input_test_thermal )
ENDIF()
