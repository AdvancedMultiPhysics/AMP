# Helper function to set source file properties to use nvcc wrapper when necessary
FUNCTION( SET_NVCC )
    IF ( USE_CUDA AND USE_KOKKOS )
        FOREACH( tmp ${ARGV} )
            IF ( IS_DIRECTORY "${tmp}" )
                FILE( GLOB TMP "${tmp}/*.cc" "${tmp}/*.cpp" "${tmp}/*.cxx" "${tmp}/*.C" )
                SET_SOURCE_FILES_PROPERTIES( ${TMP} PROPERTIES LANGUAGE CUDA )
            ELSEIF ( IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}" )
                FILE( GLOB TMP "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cc"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cpp"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.cxx"
                               "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/*.C" )
                SET_SOURCE_FILES_PROPERTIES( ${TMP} PROPERTIES LANGUAGE CUDA )
            ELSE()
                SET_SOURCE_FILES_PROPERTIES( ${tmp} PROPERTIES LANGUAGE CUDA )
            ENDIF()
        ENDFOREACH()
    ENDIF()
ENDFUNCTION()
  

# Set the fortran module directory
IF ( CMAKE_Fortran_COMPILER )
    SET( CMAKE_Fortran_MODULE_DIRECTORY ${${PROJ}_INSTALL_DIR}/include CACHE PATH "Fortran module install folder" )
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






