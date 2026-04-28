BEGIN_EXAMPLE_CONFIG( mesh )

# Copy meshes and input files
COPY_EXAMPLE_DATA_FILE( input_HierarchyExample_16b4l )

# Add the examples:
#    ADD_SAMRSOLVERS_EXAMPLE( exampleName  N_procs  inputArgs )
# Note: each test executable must be added at least once to ensure it compiles

ADD_SAMRSOLVERS_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 1)

IF ( USE_MPI )
ADD_SAMRSOLVERS_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 2)
ADD_SAMRSOLVERS_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 4)
ENDIF()


# Install the examples
INSTALL_SAMRSOLVERS_EXAMPLE( mesh )
