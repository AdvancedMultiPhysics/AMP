# Copy test inputs
COPY_TEST_FILE( TruncationError-Parameters )
COPY_TEST_FILE( Manufactured-Parameters )
COPY_TEST_FILE( input_RadiationDiffusionFD-1D )
COPY_TEST_FILE( input_RadiationDiffusionFD-2D )
COPY_TEST_FILE( input_RadiationDiffusionFD-3D )
COPY_TEST_FILE( input_testTruncation-RadiationDiffusionFD-1D )
COPY_TEST_FILE( input_testTruncation-RadiationDiffusionFD-2D )
COPY_TEST_FILE( input_testTruncation-RadiationDiffusionFD-3D )
ADD_AMP_TEST( testTruncation-RadiationDiffusionFD )


