# Configure tests
COPY_TEST_FILE( input-test64bitVectors )
COPY_TEST_FILE( input_testMultiVector )

ADD_AMP_TEST_1_2_4( test_Vector GPU NO_RESOURCES )
ADD_AMP_TEST_1_2_4( testMultiVector GPU NO_RESOURCES )
ADD_AMP_TEST( test_VectorCopyCast )
ADD_AMP_TEST( test_VectorPerformance GPU )
ADD_AMP_TEST( test_Scalar )

IF ( USE_DEVICE )
  SET_DEVICE_COMPILER( test_VectorPerformance.cpp )
ENDIF()

# Add a provisional test for testing the performance of makeConsistent
ADD_AMP_TEST( testMakeConsistentPerformance PROCS 4 )

# Add a test for testing 64-bit vectors.  WARNING: this test uses a LOT of memory
ADD_AMP_TEST( test64bitVectors )
