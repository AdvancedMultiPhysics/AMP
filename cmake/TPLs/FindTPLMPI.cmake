INCLUDE(TribitsTplDeclareLibraries)

# Either the MPI compiler wrappers take care of these or the user has to set
# the explicitly using basic compile flags and ${PROJECT_NAME}_EXTRA_LINK_FLAGS.
GLOBAL_SET(TPL_MPI_INCLUDE_DIRS)
GLOBAL_SET(TPL_MPI_LIBRARIES)
GLOBAL_SET(TPL_MPI_LIBRARY_DIRS)

IF(WIN32 AND TPL_ENABLE_MPI)
    FIND_PACKAGE(MPI)
    INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
    GLOBAL_SET(TPL_MPI_INCLUDE_DIRS ${MPI_INCLUDE_PATH})
    GLOBAL_SET(TPL_MPI_LIBRARIES ${MPI_LIBRARIES})
ENDIF()

IF ( NOT MPIEXEC_NUMPROC_FLAG )
    SET( MPIEXEC_NUMPROC_FLAG "-n" )
ENDIF()
IF ( NOT MPIEXEC )
    SET( MPIEXEC MPI )
ENDIF()
IF ( USE_MPI_FOR_SERIAL_TESTS )
    GLOBAL_SET( USE_EXT_MPI_FOR_SERIAL_TESTS ${USE_MPI_FOR_SERIAL_TESTS} )
ENDIF()

GLOBAL_SET( USE_EXT_MPI ON )
GLOBAL_SET( MPIEXEC ${MPIEXEC} )
GLOBAL_SET( MPIEXEC_NUMPROC_FLAG ${MPIEXEC_NUMPROC_FLAG} )

# Add the definitions
SET( USE_EXT_MPI 1 )

