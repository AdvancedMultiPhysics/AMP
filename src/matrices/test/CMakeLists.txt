############################################################################
#  Configure tests
############################################################################

# Function to add a serial or 1,2,4 processor test dependent on the existence of a parallel matrix
FUNCTION( ADD_AMP_TEST_124_MATRIX EXEFILE ${ARGN} )
    IF ( USE_TRILINOS )
        # We have a parallel matrix
        ADD_AMP_TEST_1_2_4( ${EXEFILE} ${ARGN} )
    ELSE()
        # We do not have a parallel matrix
        ADD_AMP_TEST( ${EXEFILE} ${ARGN} )
    ENDIF()
ENDFUNCTION()


# Copy meshes
IF ( AMP_DATA )
    COPY_MESH_FILE( pellet_1x.e )
    COPY_MESH_FILE( pellet_lo_res.e )
    COPY_MESH_FILE( clad_1x_1pellet.e )
    COPY_MESH_FILE( clad_1x_2pellet.e )
    COPY_MESH_FILE( cube343H27.e )
    COPY_MESH_FILE( mpcMesh-1 )
ENDIF()
FILE( COPY "${AMP_SOURCE_DIR}/mesh/test/data/input_AMP_mesh_zoo_3D" DESTINATION . )
FILE( COPY "${AMP_SOURCE_DIR}/mesh/test/data/input_AMP_mesh_zoo_3D_hires" DESTINATION . )


# Add matrix test
ADD_AMP_TEST_1_2_4( test_Matrix )
ADD_AMP_TEST_1_2_4( test_RectangularMatricesMultiplication )

ADD_AMP_TEST_124_MATRIX( testMatMultCoarsen )

ADD_AMP_TEST_124_MATRIX( testMatVecPerf )
COPY_TEST_FILE( input_testMatVecPerf-1 )
COPY_TEST_FILE( input_testMatVecPerf-2 )
COPY_TEST_FILE( input_testMatVecPerf-3 )

ADD_AMP_TEST_124_MATRIX( testMatOpDev )
ADD_AMP_TEST_124_MATRIX( testMatCopyCast )
IF ( USE_DEVICE )
    SET_DEVICE_COMPILER( testMatCopyCast )
ENDIF()

ADD_AMP_TEST( test_getRow input_AMP_mesh_zoo_3D )
ADD_AMP_TEST( test_getRow input_AMP_mesh_zoo_3D PROCS 4 )

IF( USE_LIBMESH AND AMP_DATA )
  ADD_AMP_TEST_124_MATRIX( testMatCreate )
  IF ( USE_TRILINOS OR USE_PETSC ) 
    ADD_AMP_TEST_124_MATRIX( testMatVec )
  ENDIF()
  COPY_TEST_FILE( input_testMatCreate-1 )
  COPY_TEST_FILE( input_testMatVec-1 )
  COPY_TEST_FILE( input_testMatVec-2 )
  ADD_AMP_TEST_1_2_4( testMatMultiply )
ENDIF()

IF ( USE_TRILINOS ) 
    ADD_AMP_TEST( test_Epetra_FECrsMatrix_bug PROCS 2 )
ENDIF()


