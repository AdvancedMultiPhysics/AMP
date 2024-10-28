# Set some CMake properties    
CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
CMAKE_POLICY( SET CMP0110 NEW )


# Print banner
MESSAGE ("===============")
MESSAGE ("Configuring AMP")
MESSAGE ("===============")


# Set the project name
SET( PROJ AMP )                 # Set the project name for CMake
SET( AMP_LIB amp )              # Set the final library name
SET( AMP_INC AMP )              # Set an optional subfolder for includes (e.g. include/name/...)
SET( AMP_MAJOR_VERSION 3 )      # Set the major version
SET( AMP_MINOR_VERSION 1 )      # Set the minor version
IF ( NOT ${PROJ}_INSTALL_DIR )
    SET( ${PROJ}_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}" )
ENDIF()


# Initialize the project (should disable langagues that the TPL builder will enable)
PROJECT( ${PROJ} LANGUAGES )


# Prevent users from building in place
IF ("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" )
    MESSAGE( FATAL_ERROR "Building code in place is a bad idea" )
ENDIF()


# Set source/install paths
SET( ${PROJ}_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src" )
SET( ${PROJ}_BUILD_DIR  "${CMAKE_CURRENT_BINARY_DIR}" )
INCLUDE_DIRECTORIES( "${${PROJ}_INSTALL_DIR}/include" )
SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} "${${PROJ}_INSTALL_DIR}/lib")


# Disable link-time optimization
SET( DISABLE_LTO TRUE)


# Build the TPLs if necessary
ADD_CUSTOM_TARGET( build-test )     # Deprecated, remove
ADD_CUSTOM_TARGET( build-examples ) # Deprecated, remove
ADD_CUSTOM_TARGET( check COMMAND make test  )
IF ( TPL_URL OR TPL_SOURCE_DIR )
    INCLUDE( cmake/configure_TPLs.cmake )
    CONFIGURE_TPL_BUILDER()
ENDIF()


# Load the TPL package
IF ( NOT ONLY_BUILD_DOCS )
    IF ( NOT TPL_DIRECTORY )
        MESSAGE( FATAL_ERROR "TPL_DIRECTORY must be specified")
    ENDIF()
    IF ( NOT DEFINED TPL_LIST_REQUIRED )
        SET( TPL_LIST_REQUIRED STACKTRACE )
    ENDIF()
    IF ( NOT DEFINED TPL_LIST_OPTIONAL )
	    SET( TPL_LIST_OPTIONAL CPPCHECK BOOST CABANA CATCH2 EIGEN FFTW HDF5 HYPRE KOKKOS LAPACK LAPACK_WRAPPERS LIBMESH MPI OGRE PETSC QT QWT RAJA SAMRAI SILO SUNDIALS THRUST TIMER TRILINOS UMPIRE X11 ZLIB GKLIB METIS PARMETIS )
    ENDIF()
    IF ( "${TPL_LIST_REQUIRED};${TPL_LIST_OPTIONAL}" MATCHES "LAPACK" AND NOT "${TPL_LIST_REQUIRED};${TPL_LIST_OPTIONAL}" MATCHES "LAPACK_WRAPPERS" )
        # If a user specified LAPACK support add optional support for LAPACK_WRAPPERS
        IF ( "${TPL_LIST_OPTIONAL}" STREQUAL "")
            SET( TPL_LIST_OPTIONAL "LAPACK_WRAPPERS" )
        ELSE()
            SET( TPL_LIST_OPTIONAL "${TPL_LIST_OPTIONAL};LAPACK_WRAPPERS" )
        ENDIF()
    ENDIF()
    STRING( REGEX REPLACE  ","  ";"  TPL_LIST_REQUIRED  "${TPL_LIST_REQUIRED}" )
    STRING( REGEX REPLACE  ","  ";"  TPL_LIST_OPTIONAL  "${TPL_LIST_OPTIONAL}" )
    FIND_PACKAGE( TPLs REQUIRED ${TPL_LIST_REQUIRED} OPTIONAL_COMPONENTS ${TPL_LIST_OPTIONAL} PATHS "${TPL_DIRECTORY}" )
    MESSAGE("TPL_LIST ${TPL_LIST}")
    MESSAGE("  TPLs Required: ${TPL_LIST_REQUIRED}")
    MESSAGE("  TPLs Supported: ${TPL_LIST_OPTIONAL}")
    MESSAGE("  TPLs Built: ${TPLs_LIST}")
    MESSAGE("  TPLs Included: ${TPLs_LIST_INCLUDED}")
    MESSAGE("  Include Paths: ${TPLs_INCLUDE_DIRS}")
    MESSAGE("  Libraries: ${TPLs_LIBRARIES}")
    MESSAGE("  CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
    MESSAGE("  CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
    MESSAGE("  Running with ${NUMBER_OF_GPUS} GPUs")
ELSE()
    INCLUDE( "cmake/docs_only.cmake" )
ENDIF()


#get_property(importTargets DIRECTORY "${CMAKE_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
#MESSAGE( FATAL_ERROR "${importTargets}" )


# Check the C++ version
IF ( "${CMAKE_CXX_STANDARD}" LESS "17" )
    MESSAGE( FATAL_ERROR "AMP requires C++17 or later, C++${CMAKE_CXX_STANDARD} defined" )
ENDIF()


# Include the TPL inclde paths and libraries
INCLUDE_DIRECTORIES( SYSTEM ${TPLs_INCLUDE_DIRS} )
SET( TPL_LIBS ${TPLs_LIBRARIES} )


# Set testing paramaters
SET( DROP_METHOD "http" )
SET( DROP_SITE "" )
SET( DROP_LOCATION "/CDash/submit.php?project=AMP" )
SET( TRIGGER_SITE "" )
SET( DROP_SITE_CDASH TRUE )
ENABLE_TESTING()
INCLUDE( CTest )


# Configure AMP (all external libraries must be configured first)
IF ( NOT ONLY_BUILD_DOCS )
    INCLUDE( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libraries.cmake" )
    CONFIGURE_AMP()
    CONFIGURE_LINE_COVERAGE()
ENDIF()


# Create the target for documentation
ADD_CUSTOM_TARGET( doc )
ADD_CUSTOM_TARGET( latex_docs )
CHECK_ENABLE_FLAG( USE_EXT_DOXYGEN 1 )
CHECK_ENABLE_FLAG( USE_LATEX 1 )
FILE( MAKE_DIRECTORY "${${PROJ}_INSTALL_DIR}/doc" )
IF ( USE_EXT_DOXYGEN )
    SET( DOXYFILE_LATEX NO )
    SET( DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/Doxyfile.in" )
    SET( DOXY_HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/html/header.html" )
    SET( DOXY_FOOTER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/html/footer.html" )
    SET( DOXY_LATEX_HEADER_FILE ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/header.tex )
    SET( DOXYFILE_OUTPUT_DIR "${${PROJ}_INSTALL_DIR}/doc" )
    SET( DOXYFILE_SRC_HTML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/doxygen/html" )
    SET( DOXYFILE_SOURCE_DIR "${${PROJ}_SOURCE_DIR}" )
    SET( DOXYFILE_EXTRA_SOURCES CACHE INTERNAL "doxyfile_extra_sources" FORCE )
    SET( REL_PACKAGE_HTML "" )
    SET( DOXYGEN_MACROS "${DOXYGEN_MACROS} AMP_USE_MPI AMP_USE_LAPACK AMP_USE_PETSC" )
    SET( DOXYGEN_MACROS "${DOXYGEN_MACROS} AMP_USE_LIBMESH AMP_USE_SUNDIALS AMP_USE_TRILINOS" )
    INCLUDE( ${CMAKE_CURRENT_SOURCE_DIR}/cmake/UseDoxygen.cmake )
    IF ( NOT DOXYGEN_FOUND )
        MESSAGE( WARNING "Doxygen not found, disabling documentation.  Set USE_EXT_DOXYGEN=0 to disable this message")
        SET( USE_EXT_DOXYGEN OFF )
    ELSE()
        MESSAGE("DOXYGEN_MACROS = ${DOXYGEN_MACROS}")
    ENDIF()
ENDIF()


# Create custom targets for distclean
IF ( NOT ONLY_BUILD_DOCS )
    ADD_DISTCLEAN( src ampdir AMP compile_commands.json AMP-Data.tar.gz tpl-build tpl-builder )
ENDIF()


# Write the version info
WRITE_REPO_VERSION()


# Create a release target
CREATE_RELEASE()


# Add a timer for the build
CHECK_ENABLE_FLAG( TIME_BUILD 0 )
IF ( TIME_BUILD )
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CMAKE_COMMAND} -E time")
    MESSAGE( STATUS "Adding build timers" )
ENDIF()


# Add cppcheck as part of build (still needs testing/development)
# IF ( CMAKE_CXX_CPPCHECK )
#     LIST( APPEND CMAKE_CXX_CPPCHECK
#             "--enable=warning"
#             "--inconclusive"
#             "--force"
#             "-iIO/test/test_HDF5.cpp"               # Causes cppcheck 2.6 to segfault
#             "-iutils/DelaunayTessellation.cpp"      # Very slow
#             "-ihost_defines.h"
#             "--suppressions-list=${CMAKE_CURRENT_SOURCE_DIR}/cppcheckSuppressionFile"
#     )
# ENDIF()
# MESSAGE( "CMAKE_CXX_CPPCHECK=${CMAKE_CXX_CPPCHECK}")


# Add the src directory
IF ( NOT ONLY_BUILD_DOCS )
    CHECK_ENABLE_FLAG( EXCLUDE_TESTS_FROM_ALL 1 )
    ADD_SUBDIRECTORY( src )
ENDIF()


# Add the cppcheck tests
IF ( NOT ONLY_BUILD_DOCS )
    SET( CPPCHECK_TIMEOUT 700 )
    SET( CPPCHECK_INCLUDE "${TIMER_INCLUDE}" "${${PROJ}_INSTALL_DIR}/include" )
    SET( CPPCHECK_IGNORE
        "IO/test/test_HDF5.cpp"                 # Causes cppcheck 2.6 to segfault
        "utils/DelaunayTessellation.cpp"        # Very slow
        "${CMAKE_BINARY_DIR}/src/graphics/getLines/moc_mainwindow.cpp" ) # Generated file
    SET( CPPCHECK_USE_JSON TRUE )
    FIND_PACKAGE( Cppcheck )
ENDIF()


# Add the cppclean tests
IF ( NOT ONLY_BUILD_DOCS )
    SET( CPPCLEAN_OPTIONS )
    SET( CPPCLEAN_EXCLUDE 
        libmeshGenerators.h meshGenerators.h meshTests.h
        DOFManager_tests.h test_Discretization.h
        Vector.h VectorDataIterator.h PetscVector.h NativePetscVector.inline.h
        Material.h
    )
    SET( CPPCLEAN_SUPPRESSIONS 
        ".inline.h' does not need to be #included"
        "tmpl.h' does not need to be #included"
        ".I' does not need to be #included"
        "'ProfilerApp.h' does not need to be #included"
        "'Utilities.h' does not need to be #included"
        "'UtilityMacros.h' does not need to be #included"
        "'utils/Counter.h' does not need to be #included"
        "'libmeshGenerators.h' does not need to be #included"
        "should #include header file '.*/src/.*'"
    )
    SET( CPPCLEAN_UNNECESSARY_INCLUDE 1 )
    SET( CPPCLEAN_EXTRA_INCLUDE 1 )
    SET( CPPCLEAN_SHOULD_INCLUDE 1 )
    SET( CPPCLEAN_INCLUDE_NOT_FOUND 1 )
    SET( CPPCLEAN_FUN_NOT_FOUND 0 )
    SET( CPPCLEAN_DECLARED 0 )
    SET( CPPCLEAN_STATIC 0 )
    SET( CPPCLEAN_UNUSED_VARIABLE 0 )
    SET( CPPCLEAN_UNKNOWN 1 )
    SET( CPPCLEAN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}" )
    FIND_PACKAGE( Cppclean )
ENDIF()


# Install the documentation
IF ( USE_EXT_DOXYGEN )
    ADD_DEPENDENCIES( doxygen latex_docs )
    ADD_DEPENDENCIES( doc latex_docs doxygen )
ENDIF()


# Generate a Package Configuration File
INCLUDE( CMakePackageConfigHelpers )
SET( INCLUDE_INSTALL_DIR  "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "Location of header files" )
SET( SYSCONFIG_INSTALL_DIR ${${PROJ}_INSTALL_DIR}/lib/cmake/AMP CACHE PATH "Location of configuration files" )
CONFIGURE_PACKAGE_CONFIG_FILE( cmake/AMP.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/AMPConfig.cmake
    INSTALL_DESTINATION "${${PROJ}_INSTALL_DIR}/lib/cmake/AMP"
    PATH_VARS INCLUDE_INSTALL_DIR SYSCONFIG_INSTALL_DIR )
WRITE_BASIC_PACKAGE_VERSION_FILE(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/AMPVersion.cmake
    VERSION "${${PROJ}_MAJOR_VERSION}.${${PROJ}_MINOR_VERSION}.${${PROJ}_BUILD_VERSION}"
    COMPATIBILITY SameMajorVersion )
INSTALL( FILES ${CMAKE_CURRENT_BINARY_DIR}/cmake/AMPConfig.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/cmake/AMPVersion.cmake
         DESTINATION ${${PROJ}_INSTALL_DIR}/lib/cmake/AMP )


