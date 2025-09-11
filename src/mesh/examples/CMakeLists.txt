BEGIN_EXAMPLE_CONFIG( mesh )

# Copy meshes and input files
COPY_EXAMPLE_DATA_FILE( input_createMesh-cube )
COPY_EXAMPLE_DATA_FILE( input_createMesh-2d )
COPY_EXAMPLE_DATA_FILE( input_createMesh-3d )
COPY_EXAMPLE_DATA_FILE( input_createMesh-Multimesh )

# Add the examples: Note: each test executable must be added at least once to ensure it compiles
ADD_AMP_EXAMPLE( createMesh input_createMesh-cube NO_RESOURCES )
ADD_AMP_EXAMPLE( createMesh input_createMesh-2d NO_RESOURCES )
ADD_AMP_EXAMPLE( createMesh input_createMesh-3d NO_RESOURCES )
ADD_AMP_EXAMPLE( createMesh input_createMesh-Multimesh NO_RESOURCES )

# Install the examples
INSTALL_AMP_EXAMPLE( mesh )
