############################################################################
#  Configure tests
############################################################################

# Copy inputs
COPY_TEST_FILE( input_geometry_zoo )
COPY_TEST_FILE( input_MeshGeometry )

# Add tests
ADD_AMP_TEST( test_Geometry input_geometry_zoo )
# ADD_AMP_TEST( test_Geometry input_MeshGeometry )
ADD_AMP_TEST( test_GeometryHelpers )

