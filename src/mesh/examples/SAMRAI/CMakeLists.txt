BEGIN_EXAMPLE_CONFIG( mesh )

COPY_EXAMPLE_DATA_FILE( input_HierarchyExample_16b4l )

ADD_AMP_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 1)
IF ( USE_MPI )
ADD_AMP_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 2)
ADD_AMP_EXAMPLE( createAMRHierarchy input_HierarchyExample_16b4l NO_RESOURCES PROCS 4)
ENDIF()


# Set compile properties
IF ( USE_CUDA )
    FILE( GLOB SAMRAI_FILES *.cpp )
    SET_SOURCE_FILES_PROPERTIES( ${SAMRAI_FILES} PROPERTIES LANGUAGE CUDA )
ENDIF()
