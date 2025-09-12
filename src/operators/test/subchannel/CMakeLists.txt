# Copy test inputs
COPY_TEST_FILE( input_testSubchannelUtilities )
COPY_TEST_FILE( input_testSubchannelPhysicsModel )
COPY_TEST_FILE( input_testSubchannelTwoEqNonlinearOperator )
COPY_TEST_FILE( input_testSubchannelTwoEqLinearOperator )
COPY_TEST_FILE( input_testSubchannelFourEqNonlinearOperator )
COPY_TEST_FILE( input_testSubchannelFourEqLinearOperator )

# Copy test data

# Add the tests
INITIALIZE_TESTBUILDER()
ADD_TB_TEST( testSubchannelHelpers )
ADD_TB_TEST_1_2_4( testSubchannelPhysicsModel )
ADD_TB_TEST( testSubchannelTwoEqNonlinearOperator )
ADD_TB_TEST( testSubchannelTwoEqLinearOperator )
ADD_TB_TEST( testSubchannelFourEqNonlinearOperator )
ADD_TB_TEST( testSubchannelFourEqLinearOperator )
FINALIZE_TESTBUILDER()
