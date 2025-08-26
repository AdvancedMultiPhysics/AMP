# Copy test inputs
COPY_TEST_FILE( input_db )
ADD_AMP_TEST( radiationDiffusionFD-test )

COPY_TEST_FILE( input_testRadiationDiffusionFD-truncation )
ADD_AMP_TEST( radiationDiffusionFD-truncationTest )


