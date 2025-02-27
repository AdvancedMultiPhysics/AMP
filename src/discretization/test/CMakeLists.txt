############################################################################
#  Configure tests
############################################################################

# Copy meshes
IF ( AMP_DATA )
    COPY_MESH_FILE( pellet_1x.e )
    COPY_MESH_FILE( clad_1x_1pellet.e )
    COPY_MESH_FILE( cube343H27.e )
    COPY_MESH_FILE( pellet_lo_res.e )
    COPY_MESH_FILE( clad_lo_res_4pellet.e )
    COPY_MESH_FILE( cylinder270.e )
    COPY_MESH_FILE( cylinder4K.e )
ENDIF()

# Add tests
ADD_AMP_TEST_1_2_4( test_Discretization )
ADD_AMP_TEST_1_2_4( test_Neighbors )

ADD_AMP_TEST_1_2_4( test_vertexRemoteDOFs )
COPY_TEST_FILE( input_testRemoteDOFs-boxmesh-1 )
COPY_TEST_FILE( input_testRemoteDOFs-boxmesh-2 )
COPY_TEST_FILE( input_testRemoteDOFs-libmesh-1 )
COPY_TEST_FILE( input_testRemoteDOFs-libmesh-2 )

IF ( USE_LIBMESH )
    ADD_AMP_TEST_1_2_4( test_libmeshElement )
ENDIF()
