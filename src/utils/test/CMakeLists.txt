############################################################################
#  Configure tests
############################################################################

# Add the data files needed for the tests
COPY_TEST_FILE( point_set_2d )
COPY_TEST_FILE( point_set_2d_2 )
COPY_TEST_FILE( point_set_2d_3 )
COPY_TEST_FILE( point_set_2d_4 )
COPY_TEST_FILE( input_InputManager )
COPY_TEST_FILE( laser_plasma_input.txt )
COPY_TEST_FILE( input_Database )
COPY_TEST_FILE( input_SAMRAI )
COPY_TEST_FILE( input_NRDF )
COPY_TEST_FILE( input_NRDF2 )
COPY_TEST_FILE( library.yml )
COPY_TEST_FILE( Johnson.yml )

# Add the tests
ADD_AMP_TEST( test_compile )
ADD_AMP_TEST( test_Units )
ADD_AMP_TEST( test_Database input_Database laser_plasma_input.txt library.yml Johnson.yml )
# ADD_AMP_TEST( test_Database input_Database laser_plasma_input.txt library.yml Johnson.yml PROCS 2 )
ADD_AMP_TEST( test_Database2 )
ADD_AMP_TEST( test_intN )
ADD_AMP_TEST( test_kdtree )
ADD_AMP_TEST( test_MathExpr )
ADD_AMP_TEST( test_MemoryAllocators )
ADD_AMP_TEST_1_2_4( test_AMPManager )
ADD_AMP_TEST_1_2_4( test_AMP_MPI )
ADD_AMP_TEST_1_2_4( test_Utilities )
ADD_AMP_TEST_1_2_4( test_InputManager )
ADD_AMP_TEST_1_2_4( testManufacturedSolution )
ADD_AMP_TEST( test_DelaunayHelpers )
ADD_AMP_TEST( test_DelaunayInterpolation point_set_2d point_set_2d_2 point_set_2d_3 point_set_2d_4 )
IF( NOT USING_ICC )
    ADD_AMP_TEST( testArray )
ENDIF()
IF ( CMAKE_Fortran_COMPILER )
    ADD_AMP_TEST( test_Fortran )
ENDIF()
ADD_AMP_PROVISIONAL_TEST( test_crash )
#ADD_AMP_PROVISIONAL_TEST( test_crash_log )

