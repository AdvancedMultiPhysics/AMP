# Configure tests
COPY_MESH_FILE( pellet_1x.e )
COPY_TEST_FILE( input-test64bitVectors )
COPY_TEST_FILE( input_testMultiVector )
ADD_AMP_TEST_1_2_4( test_Vector GPU )
ADD_AMP_TEST_1_2_4( testMultiVector GPU )
ADD_AMP_TEST( test_VectorCopyCast )
ADD_AMP_TEST( test_VectorPerformance GPU )
ADD_AMP_TEST( test_Scalar )

# Add a provisional test for testing the performance of makeConsistent
ADD_AMP_TEST( testMakeConsistentPerformance PROCS 4 )

# Add a test for testing 64-bit vectors.  WARNING: this test uses a LOT of memory
IF ( USE_LIBMESH AND AMP_DATA )
    ADD_AMP_TEST( test64bitVectors )
ENDIF()
