# Copy test inputs
COPY_TEST_FILE( input_testMoabBasedOperator )
COPY_TEST_FILE( SESSION.NAME )
COPY_TEST_FILE( pipe.rea )

# Copy test data
COPY_MESH_FILE( input.h5m )

# Add the tests
ADD_AMP_TEST_1_2_4( testMoabBasedOperator )
ADD_AMP_PROVISIONAL_TEST( testMoabInterface )
IF ( USE_NEK )
    ADD_AMP_TEST_PARALLEL( testMoabNekPipe 2 ) # This test doesn't work in serial
    ADD_AMP_TEST_PARALLEL( testMoabNekPipe 4 )
    ADD_AMP_TEST_PARALLEL( testMoabNekOperator 2 ) # By extension, neither does this
    ADD_AMP_TEST_PARALLEL( testMoabNekOperator 4 )
ENDIF()
