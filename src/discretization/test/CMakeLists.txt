# Configure tests
ADD_AMP_TEST_1_2_4( test_Discretization NO_RESOURCES )
ADD_AMP_TEST_1_2_4( test_Neighbors NO_RESOURCES )
IF ( USE_LIBMESH AND AMP_DATA )
    COPY_MESH_FILE( pellet_1x.e )
    ADD_AMP_TEST_1_2_4( test_libmeshElement NO_RESOURCES )
ENDIF()
