INITIALIZE_TESTBUILDER()

# Add node-node tests
ADD_TB_TEST_1_2_4( testNodeToNodeMap inputNodeToNodeMap-1 )
COPY_TEST_FILE( inputNodeToNodeMap-1 )

# Test strided z axis map
ADD_TB_TEST( testStridedZAxisMap )
COPY_TEST_FILE( input_testStridedZAxisMap )

# Test maps that use libmesh
IF ( AMP_DATA AND USE_LIBMESH )

    # Add Jacobian map
    ADD_TB_PROVISIONAL_TEST( testJacobianMap )

    # Test ScalarN2GZAxis map
    ADD_TB_TEST_1_2_4( testScalarN2GZAxisMap )
    COPY_TEST_FILE( inputScalarN2GZAxisMap-1 )

    # Test ScalarZAxis map
    ADD_TB_TEST_1_2_4( testScalarZAxisMap )
    COPY_TEST_FILE( inputScalarZAxisMap-1 )

    # Test map operators
    ADD_TB_TEST_1_2_4( testMapOperator )
    COPY_TEST_FILE( input_testMapOperator )

    # Test CladToSubchannel map
    ADD_TB_TEST_1_2_4( testCladToSubchannelMap )
    ADD_TB_TEST_1_2_4( testSubchannelToCladMap )
    COPY_TEST_FILE( inputCladToSubchannelMap-1 )
    COPY_TEST_FILE( inputSubchannelToCladMap-1 )

ENDIF()

FINALIZE_TESTBUILDER()
