# Helper function to set source file properties to use device compiler when necessary
FUNCTION( SET_DEVICE_COMPILER )
    IF ( USE_DEVICE )
        IF ( USE_CUDA )
            set(DEVICE_LANG CUDA)
        ENDIF()
        IF ( USE_HIP )
            set(DEVICE_LANG HIP)
        ENDIF()
        FOREACH( tmp ${ARGV} )
            IF ( IS_DIRECTORY "${tmp}" )
                FILE( GLOB TMP "${tmp}/*.cc" "${tmp}/*.cpp" "${tmp}/*.cxx" "${tmp}/*.C" )
                SET_SOURCE_FILES_PROPERTIES( ${TMP} PROPERTIES LANGUAGE ${DEVICE_LANG} )
            ELSEIF ( IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}" )
                FILE( GLOB TMP "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cc"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cxx"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.C" )
                SET_SOURCE_FILES_PROPERTIES( ${TMP} PROPERTIES LANGUAGE ${DEVICE_LANG} )
            ELSE()
                SET_SOURCE_FILES_PROPERTIES( ${tmp} PROPERTIES LANGUAGE ${DEVICE_LANG} )
            ENDIF()
        ENDFOREACH()
    ENDIF()
ENDFUNCTION()


# Set the fortran module directory (and flags)
IF ( CMAKE_Fortran_COMPILER )
    SET( CMAKE_Fortran_MODULE_DIRECTORY ${${PROJ}_INSTALL_DIR}/include CACHE PATH "Fortran module install folder" )
    IF ( CMAKE_COMPILER_IS_GNUG77 OR (${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU") )
        SET(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -ffixed-line-length-none" )
    ENDIF()
ENDIF()


# Set the default matrix type
SET( MATRIX_TYPES "ManagedEpetraMatrix" "NativePetscMatrix" "CSRMatrix" "DenseSerialMatrix" "NULL" )
IF ( NOT DEFINED DEFAULT_MATRIX )
    SET( DEFAULT_MATRIX "CSRMatrix" )
ENDIF()
IF ( NOT "${DEFAULT_MATRIX}" IN_LIST MATRIX_TYPES )
    MESSAGE( FATAL_ERROR "DEFAULT_MATRIX=${DEFAULT_MATRIX} is not a valid matrix type" )
ENDIF()
IF ( "${DEFAULT_MATRIX}" STREQUAL "ManagedEpetraMatrix" AND NOT USE_TRILINOS )
    MESSAGE( FATAL_ERROR "Specified default matrix of ManagedEpetraMatrix, but compiled without Trilinos" )
ENDIF()
IF ( "${DEFAULT_MATRIX}" STREQUAL "NativePetscMatrix" AND NOT USE_PETSC )
    MESSAGE( FATAL_ERROR "Specified default matrix of NativePetscMatrix, but compiled without PETSc" )
ENDIF()


# Add the AMP folders and libraries
ADD_AMP_LIBRARY( applications )
ADD_AMP_LIBRARY( discretization )
ADD_AMP_LIBRARY( geometry )
ADD_AMP_LIBRARY( graphics )
ADD_AMP_LIBRARY( IO )
ADD_AMP_LIBRARY( matrices )
ADD_AMP_LIBRARY( materials )
ADD_AMP_LIBRARY( mesh)
ADD_AMP_LIBRARY( operators )
ADD_AMP_LIBRARY( solvers )
ADD_AMP_LIBRARY( time_integrators )
ADD_AMP_LIBRARY( utils )
ADD_AMP_LIBRARY( vectors )
INSTALL_PROJ_LIB()


