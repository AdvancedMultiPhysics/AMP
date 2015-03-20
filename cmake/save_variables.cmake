# Save the necessary cmake variables to a file for applications to load
# Note: we need to save the external libraries in the same order as AMP for consistency
FUNCTION( SAVE_CMAKE_FLAGS )
    # Don't force downstream apps from using certain warnings
    STRING(REGEX REPLACE "-Wextra" "" CMAKE_C_FLAGS_2   "${CMAKE_C_FLAGS}"   )
    STRING(REGEX REPLACE "-Wextra" "" CMAKE_CXX_FLAGS_2 "${CMAKE_CXX_FLAGS}" )
    # Write the header (comments)
    FILE(WRITE  ${AMP_INSTALL_DIR}/amp.cmake "# This is a automatically generate file to include AMP within another application\n\n" )
    # Write the compilers and compile flags
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the compilers and compile flags\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_C_COMPILER ${CMAKE_C_COMPILER})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_Fortran_COMPILER ${CMAKE_Fortran_COMPILER})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_FORTRAN ${USE_FORTRAN})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS_2}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS_2}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_Fortran_FLAGS \"${CMAKE_Fortran_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(LDFLAGS \"${LDFLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(COMPILE_MODE ${COMPILE_MODE})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(DISABLE_GXX_DEBUG ${DISABLE_GXX_DEBUG})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS \"${CMAKE_EXE_LINK_DYNAMIC_C_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS \"${CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_SHARED_LIBRARY_C_FLAGS \"${CMAKE_SHARED_LIBRARY_C_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_SHARED_LIBRARY_CXX_FLAGS \"${CMAKE_SHARED_LIBRARY_CXX_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_SHARED_LINKER_FLAGS \"${CMAKE_SHARED_LINKER_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS \"${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS}\")\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS \"${CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS}\")\n" )
    # Write the AMP_DATA and AMP_SOURCE paths
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the AMP data and source directories\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_DATA ${AMP_DATA})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_SOURCE ${AMP_SOURCE_DIR})\n" )
    # Create the AMP libraries and include paths
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the AMP libraries\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_LIBS ${AMP_LIBS})\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${AMP_TRUNK}/external/boost/include )\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${AMP_INSTALL_DIR}/include )\n" )
    IF ( USE_AMP_UTILS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_UTILS ${USE_AMP_UTILS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_UTILS ) \n" )
    ENDIF()
    IF ( USE_AMP_MESH )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_MESH ${USE_AMP_MESH}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_MESH ) \n" )
    ENDIF()
    IF ( USE_AMP_DISCRETIZATION )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_DISCRETIZATION ${USE_AMP_DISCRETIZATION}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_DISCRETIZATION ) \n" )
    ENDIF()
    IF ( USE_AMP_VECTORS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_VECTORS ${USE_AMP_VECTORS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_VECTORS ) \n" )
    ENDIF()
    IF ( USE_AMP_MATRICES )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_MATRICES ${USE_AMP_MATRICES}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_MATRICES ) \n" )
    ENDIF()
    IF ( USE_AMP_MATERIALS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_MATERIALS ${USE_AMP_MATERIALS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_MATERIALS ) \n" )
    ENDIF()
    IF ( USE_AMP_OPERATORS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_OPERATORS ${USE_AMP_OPERATORS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_OPERATORS ) \n" )
    ENDIF()
    IF ( USE_AMP_SOLVERS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_SOLVERS ${USE_AMP_SOLVERS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_SOLVERS ) \n" )
    ENDIF()
    IF ( USE_AMP_TIME_INTEGRATORS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_AMP_TIME_INTEGRATORS ${USE_AMP_TIME_INTEGRATORS}) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_AMP_TIME_INTEGRATORS ) \n" )
    ENDIF()
    # Create the external libraries and include paths in the order they are linked in AMP
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS )\n" )
    # Add boost
    IF ( USE_EXT_BOOST )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add boost\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${BOOST_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_BOOST 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_BOOST ) \n" )
    ENDIF()
    # Add Libmesh
    IF ( USE_EXT_LIBMESH )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Libmesh\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${LIBMESH_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${LIBMESH_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -DLIBMESH_ENABLE_PARMESH )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_LIBMESH 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_LIBMESH ) \n" )
    ENDIF()
    # Add NEK
    IF ( USE_EXT_NEK )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add NEK\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${NEK_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${NEK_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_NEK 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_NEK ) \n" )
    ENDIF()
    # Add MOAB
    IF ( USE_EXT_MOAB )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add MOAB\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${MOAB_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${MOAB_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_MOAB 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_MOAB ) \n" )
    ENDIF()
    # Add DENDRO
    IF ( USE_EXT_DENDRO )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add DENDRO\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${DENDRO_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${DENDRO_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_DENDRO 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_DENDRO ) \n" )
    ENDIF()
    # Add Netcdf
    IF ( USE_NETCDF )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Netcdf\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${NETCDF_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${NETCDF_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_NETCDF 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_NETCDF ) \n" )
    ENDIF()
    # Add DTK 
    IF ( USE_EXT_DTK )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_DTK 1 ) \n" )
    ENDIF()
    # Add Trilinos
    IF ( USE_EXT_TRILINOS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Trilinos\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${TRILINOS_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${TRILINOS_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_TRILINOS 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_UTILS   ${USE_TRILINOS_UTILS}   ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_TEUCHOS ${USE_TRILINOS_TEUCHOS} ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_VECTORS ${USE_TRILINOS_VECTORS} ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_SOLVERS ${USE_TRILINOS_SOLVERS} ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_THYRA   ${USE_TRILINOS_THYRA}   ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_NOX     ${USE_TRILINOS_NOX}     ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_STKMESH ${USE_TRILINOS_STKMESH} ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_TRILINOS_STRATIMIKOS ${USE_TRILINOS_STRATIMIKOS} ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_TRILINOS ) \n" )
        IF ( USE_TRILINOS_THYRA )
            FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_TRILINOS_THYRA ) \n" )
        ENDIF()
        IF ( USE_TRILINOS_NOX )
            FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_TRILINOS_NOX ) \n" )
        ENDIF()
        IF ( USE_TRILINOS_STKMESH )
            FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_TRILINOS_STKMESH ) \n" )
        ENDIF()
    ENDIF()
    # Add PETsc
    IF ( USE_EXT_PETSC )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add PETsc\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${PETSC_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${PETSC_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_PETSC 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_PETSC ) \n" )
    ENDIF()
    # Add Sundials
    IF ( USE_EXT_SUNDIALS )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Sundials\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${SUNDIALS_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${SUNDIALS_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_SUNDIALS 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_SUNDIALS ) \n" )
    ENDIF()
    # Add Silo
    IF ( USE_EXT_SILO )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add silo\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${SILO_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${SILO_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_SILO 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_SILO ) \n" )
    ENDIF()
    # Add Hypre
    IF ( USE_EXT_HYPRE )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add hypre\n" )
        # FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${HYPRE_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${HYPRE_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_HYPRE 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_HYPRE ) \n" )
    ENDIF()
    # Add X11
    IF ( USE_EXT_X11 )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add X11\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${X11_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${X11_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_X11 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_X11 ) \n" )
    ENDIF()
    # Add HDF5
    IF ( USE_EXT_HDF5 )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add HDF5\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${HDF5_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${HDF5_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_HDF5 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_HDF5 ) \n" )
    ENDIF()
    # Add TIMER
    IF ( USE_TIMER OR USE_EXT_TIMER )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Timer\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${TIMER_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${TIMER_LIBS} )\n" )
    ENDIF()
    # Add MPI
    IF ( USE_EXT_MPI )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add MPI\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${MPI_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS}  ${MPI_LINK_FLAGS} ${MPI_LIBRARIES} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_MPI 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_MPI ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(MPIEXEC ${MPIEXEC} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(MPIEXEC_NUMPROC_FLAG ${MPIEXEC_NUMPROC_FLAG} )\n" )
        IF ( USE_MPI_FOR_SERIAL_TESTS )
            FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_MPI_FOR_SERIAL_TESTS 1 )\n" )
            FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_EXT_MPI_FOR_SERIAL_TESTS 1 )\n" )
        ENDIF()
    ENDIF()
    # Add ZLIB
    IF ( USE_EXT_ZLIB )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add ZLIB\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${ZLIB_INCLUDE} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${ZLIB_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_ZLIB 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -D USE_EXT_ZLIB ) \n" )
    ENDIF()
    # Add LAPACK and BLAS
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add LAPACK/BLAS\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS}  ${LAPACK_LIBS} ${BLAS_LIBS} ${BLAS_LAPACK_LIBS} )\n" )
    # Add LDLIBS
    IF ( LDLIBS )
    	FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS}  ${LDLIBS} )\n" )
    ENDIF()
    # Add coverage
    IF ( ENABLE_GCOV )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add coverage flags\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${COVERAGE_LIBS} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( ENABLE_GCOV 1 ) \n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS( -fprofile-arcs -ftest-coverage ) \n" )
    ENDIF()
    # Add doxygen
    IF ( USE_EXT_DOXYGEN )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add doxygen flags\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( USE_EXT_DOXYGEN ${USE_EXT_DOXYGEN} )\n" )
        FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( DOXYGEN_MACROS \"${DOXYGEN_MACROS}\" )\n" )
    ENDIF()
    # Add misc flags
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add misc flags\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $" "{EXTERNAL_LIBS} ${SYSTEM_LIBS} )\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( TEST_MAX_PROCS ${TEST_MAX_PROCS} )\n" )
    FILE(APPEND ${AMP_INSTALL_DIR}/amp.cmake "\n" )
ENDFUNCTION()

