# Macro to convert a m4 file
# This command converts a file of the format "global_path/file.fm4"
# and convertes it to file.f90.  It also requires the path.  
MACRO (CONVERT_M4_FORTRAN IN LOCAL_PATH)
    STRING(REGEX REPLACE ${LOCAL_PATH} "" OUT ${IN} )
    STRING(REGEX REPLACE "/" "" OUT ${OUT} )
    STRING(REGEX REPLACE ".fm4" ".f90" OUT ${CMAKE_CURRENT_BINARY_DIR}/${OUT} )
    CONFIGURE_FILE ( ${IN} ${IN} COPYONLY )
    add_custom_command(
        OUTPUT ${OUT}
        COMMAND m4 --prefix-builtins -I${LOCAL_PATH} ${M4_OPTIONS} ${IN} > ${OUT}
        DEPENDS ${IN}
    )
    set_source_files_properties(${OUT} PROPERTIES GENERATED true)
    SET ( SOURCES ${SOURCES} "${OUT}" )
ENDMACRO ()


# Add a package to the test dependency list
MACRO (ADD_PACKAGE_TO_TEST_DEP_LIST PACKAGE)
  IF ( TEST_DEP_LIST )
    SET ( TEST_DEP_LIST ${PACKAGE} ${TEST_DEP_LIST} )
  ELSE()
    SET ( TEST_DEP_LIST ${PACKAGE} )
  ENDIF()
ENDMACRO ()


# Add a package to the AMP library
MACRO (ADD_AMP_LIBRARY PACKAGE)
  ADD_PACKAGE_TO_TEST_DEP_LIST ( ${PACKAGE} )
  INCLUDE_DIRECTORIES ( ${AMP_INSTALL_DIR}/include/${PACKAGE} )
  ADD_SUBDIRECTORY ( ${PACKAGE} )
ENDMACRO ()


# Add an AMP executable
MACRO (ADD_AMP_EXECUTABLE PACKAGE)
  ADD_SUBDIRECTORY ( ${PACKAGE} )
ENDMACRO ()


# Initialize a package
MACRO (BEGIN_PACKAGE_CONFIG PACKAGE)
  SET ( HEADERS "" )
  SET ( CXXSOURCES "" )
  SET ( CSOURCES "" )
  SET ( FSOURCES "" )
  SET ( M4FSOURCES "" )
  SET ( SOURCES "" )
  SET ( CURPACKAGE ${PACKAGE} )
ENDMACRO ()


# Find the source files
MACRO (FIND_FILES)
    # Find the C/C++ headers
    SET ( T_HEADERS "" )
    FILE ( GLOB T_HEADERS "*.h" "*.hh" "*.I" )
    # Find the C sources
    SET ( T_CSOURCES "" )
    FILE ( GLOB T_CSOURCES "*.c" )
    # Find the C++ sources
    SET ( T_CXXSOURCES "" )
    FILE ( GLOB T_CXXSOURCES "*.cc" "*.cpp" "*.cxx" "*.C" )
    # Find the Fortran sources
    SET ( T_FSOURCES "" )
    FILE ( GLOB T_FSOURCES "*.f" "*.f90" )
    # Find the m4 fortran source (and convert)
    SET ( T_M4FSOURCES "" )
    FILE ( GLOB T_M4FSOURCES "*.fm4" )
    FOREACH (m4file ${T_M4FSOURCES})
        CONVERT_M4_FORTRAN ( ${m4file} ${CMAKE_CURRENT_SOURCE_DIR} )
    ENDFOREACH ()
    # Add all found files to the current lists
    SET ( HEADERS ${HEADERS} ${T_HEADERS} )
    SET ( CXXSOURCES ${CXXSOURCES} ${T_CXXSOURCES} )
    SET ( CSOURCES ${CSOURCES} ${T_CSOURCES} )
    SET ( FSOURCES ${FSOURCES} ${T_FSOURCES} )
    SET ( M4FSOURCES ${M4FSOURCES} ${T_M4FSOURCES} )
    SET ( SOURCES ${SOURCES} ${T_CXXSOURCES} ${T_CSOURCES} ${T_FSOURCES} ${T_M4FSOURCES} )
ENDMACRO()


# Find the source files
MACRO (FIND_FILES_PATH IN_PATH)
    # Find the C/C++ headers
    SET ( T_HEADERS "" )
    FILE ( GLOB T_HEADERS "${IN_PATH}/*.h" "${IN_PATH}/*.hh" "${IN_PATH}/*.I" )
    # Find the C sources
    SET ( T_CSOURCES "" )
    FILE ( GLOB T_CSOURCES "${IN_PATH}/*.c" )
    # Find the C++ sources
    SET ( T_CXXSOURCES "" )
    FILE ( GLOB T_CXXSOURCES "${IN_PATH}/*.cc" "${IN_PATH}/*.cpp" "${IN_PATH}/*.cxx" )
    # Find the Fortran sources
    SET ( T_FSOURCES "" )
    FILE ( GLOB T_FSOURCES "${IN_PATH}/*.f" "${IN_PATH}/*.f90" )
    # Find the m4 fortran source (and convert)
    SET ( T_M4FSOURCES "" )
    FILE ( GLOB T_M4FSOURCES "${IN_PATH}/*.fm4" )
    FOREACH (m4file ${T_M4FSOURCES})
        CONVERT_M4_FORTRAN ( ${m4file} ${CMAKE_CURRENT_SOURCE_DIR}/${IN_PATH} )
    ENDFOREACH ()
    # Add all found files to the current lists
    SET ( HEADERS ${HEADERS} ${T_HEADERS} )
    SET ( CXXSOURCES ${CXXSOURCES} ${T_CXXSOURCES} )
    SET ( CSOURCES ${CSOURCES} ${T_CSOURCES} )
    SET ( FSOURCES ${FSOURCES} ${T_FSOURCES} )
    SET ( SOURCES ${SOURCES} ${T_CXXSOURCES} ${T_CSOURCES} ${T_FSOURCES} )
ENDMACRO()


# Add a subdirectory
MACRO ( ADD_PACKAGE_SUBDIRECTORY SUBDIR )
  SET ( FULLSUBDIR ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR} )
  FIND_FILES_PATH ( ${SUBDIR} )
  FILE ( GLOB HFILES RELATIVE ${FULLSUBDIR} ${SUBDIR}/*.h ${SUBDIR}/*.hh ${SUBDIR}/*.I )
  FOREACH (HFILE ${HFILES})
    CONFIGURE_FILE ( ${FULLSUBDIR}/${HFILE} ${AMP_INSTALL_DIR}/include/${CURPACKAGE}/${SUBDIR}/${HFILE} COPYONLY )
    INCLUDE_DIRECTORIES ( ${FULLSUBDIR} )
  ENDFOREACH ()
  ADD_SUBDIRECTORY ( ${SUBDIR} )
ENDMACRO ()


# Install a package
MACRO ( INSTALL_AMP_TARGET PACKAGE )
    # Find all files in the current directory
    FIND_FILES ()
    # Copy the header files to the include path
    FILE ( GLOB HFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.hh ${CMAKE_CURRENT_SOURCE_DIR}/*.I )
    FOREACH (HFILE ${HFILES})
        CONFIGURE_FILE ( ${CMAKE_CURRENT_SOURCE_DIR}/${HFILE} ${AMP_INSTALL_DIR}/include/${CURPACKAGE}/${HFILE} COPYONLY )
    ENDFOREACH ()
    # Add the library
    ADD_LIBRARY ( ${PACKAGE} ${SOURCES} )
    # Install the package
    INSTALL ( TARGETS ${PACKAGE} DESTINATION ${AMP_INSTALL_DIR}/lib )
    INSTALL ( FILES ${HEADERS} DESTINATION ${AMP_INSTALL_DIR}/include/${PACKAGE} )
ENDMACRO ()


# Macro to verify that a variable has been set
MACRO ( VERIFY_VARIABLE VARIABLE_NAME )
IF ( NOT ${VARIABLE_NAME} )
    MESSAGE ( FATAL_ERROR "PLease set: " ${VARIABLE_NAME} )
ENDIF ()
ENDMACRO ()


# Macro to verify that a path has been set
MACRO ( VERIFY_PATH PATH_NAME )
IF ( NOT EXISTS ${PATH_NAME} )
  MESSAGE ( FATAL_ERROR "Path does not exist: " ${PATH_NAME} )
ENDIF ()
ENDMACRO ()


# Macro to identify the compiler
MACRO ( SET_COMPILER )
  # SET the C/C++ compiler
  IF ( CMAKE_COMPILE_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
    SET( USING_GCC TRUE )
    MESSAGE("Using gcc")
  ELSEIF ( MSVC OR MSVC_IDE OR MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR CMAKE_COMPILER_2005 OR MSVC90 OR MSVC10 )
    SET( USING_MICROSOFT TRUE )
    MESSAGE("Using Microsoft")
  ELSEIF ( (${CMAKE_C_COMPILER_ID} MATCHES "Intel") OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Intel") ) 
    SET(USING_ICC TRUE)
    MESSAGE("Using icc")
  ELSE ()
    SET(USING_DEFAULT TRUE)
    MESSAGE("${CMAKE_C_COMPILER_ID}")
    MESSAGE("Unknown C/C++ compiler, default flags will be used")
  ENDIF()
  # SET the Fortran++ compiler
  IF ( CMAKE_COMPILE_IS_GFORTRAN OR (${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU") )
    SET( USING_GFORTRAN TRUE )
    MESSAGE("Using gfortran")
  ELSEIF ( (${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel") ) 
    SET(USING_IFORT TRUE)
    MESSAGE("Using ifort")
  ELSE ()
    SET(USING_DEFAULT TRUE)
    MESSAGE("${CMAKE_Fortran_COMPILER_ID}")
    MESSAGE("Unknown Fortran compiler, default flags will be used")
  ENDIF()
ENDMACRO ()


# Macro to set the proper warning level for AMP code
MACRO ( SET_WARNINGS )
  IF ( USING_GCC )
    # Add gcc specific compiler options
    #    -Wno-reorder:  warning: "" will be initialized after "" when initialized here
    SET(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -Wall ") 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
    # Disable warnings that I think are irrelavent (may need to be revisited)
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wno-reorder " )
    # Disable warnings that occur frequently, but should be fixed eventually
    SET(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -Wno-unused-variable" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wno-unused-variable" )
    # Add gcc specific flags
    SET(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} -ldl" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -ldl" )
  ELSEIF ( USING_MICROSOFT )
    # Add Microsoft specifc compiler options
    SET(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} /WALL" )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WALL" )
  ELSEIF ( USING_ICC )
    # Add Intel specifc compiler options
    #    111: statement is unreachable
    #         This occurs in LibMesh
    #    177: variable "" was declared but never referenced
    #    181: argument is incompatible with corresponding format string conversion
    #    304: access control not specified ("public" by default)
    #         This occurs in LibMesh
    #    383: value copied to temporary, reference to temporary used
    #         This is an irrelavent error
    #    444: destructor for base class "" is not virtual
    #         This can create memory leaks (and should be fixed)
    #         Unfortunatelly many of these come from LibMesh
    #    522: function "xxx" redeclared "inline" after being called
    #         We should fix this, but there are a lot of these
    #    593: variable "xxx" was set but never used
    #         This occurs commonly for error flags and internal variables that are helpful for debugging
    #    654: overloaded virtual function "" is only partially overridden in class " "
    #    869: parameter "xxx" was never referenced
    #         I believe this is bad practice and should be fixed, but it may require a broader discussion (it is built into the design of Operator)
    #    981: operands are evaluated in unspecified order
    #         This can occur when an implicit conversion take place in a function call 
    #   1011: missing return statement at end of non-void function
    #         This is bad practice
    #   1418: external function definition with no prior declaration
    #         This can happen if we don't include a header file (and maybe if there is an internal function?)
    #         Unfortunatelly many of these come from trilinos
    #   1419: external declaration in primary source file
    #         This occurs in a lot of the input processor, and needs to be revisited
    #   1572: floating-point equality and inequality comparisons are unreliable
    #         LibMesh warnings
    #   1599: declaration hides parameter 
    #         LibMesh warnings
    #   2259: non-pointer conversion from "int" to "unsigned char" may lose significant bits
    #         This is bad practice, use an explict coversion instead
    #         Unfortunatelly many of these come from LibMesh
    #   6843: A dummy argument with an explicit INTENT(OUT) declaration is not given an explicit value.
    #         This is fortran error in scalelib
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -Wall" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall" )
    # Disable warnings that I think are irrelavent (may need to be revisited)
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -wd383 -wd593 -wd981" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -wd383 -wd593 -wd981" )
    # Disable warnings that occur due to other packages (it would be nice to disable them for certain header files only)
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -wd111 -wd304 -wd304 -wd444 -wd1418 -wd1572 -wd1599 -wd2259" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -wd111 -wd304 -wd304 -wd444 -wd1418 -wd1572 -wd1599 -wd2259" )
    SET(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} -diag-disable 6843" )
    # Disable warnings that occur frequently, but should be fixed eventually
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -wd522 -wd869 -wd1419" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -wd522 -wd869 -wd1419" )
  ELSEIF ( USING_DEFAULT )
    # Add default compiler options
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -Wall")
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall")
  ENDIF ()
ENDMACRO ()


# Macro to add user compile flags
MACRO ( ADD_USER_FLAGS )
    SET(CMAKE_C_FLAGS   " ${CMAKE_C_FLAGS} ${CFLAGS} ${LDFLAGS}" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} ${LDFLAGS}" )
    SET(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} ${FFLAGS}" )
ENDMACRO ()


# Macro to set the flags for debug mode
MACRO ( SET_DEBUG_MACROS )
    SET_COMPILER ()
    ADD_USER_FLAGS()
    IF ( NOT DISABLE_GXX_DEBUG )
        SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -D_GLIBCXX_DEBUG" )
        SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG" )
        SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -D_GLIBCXX_DEBUG_PEDANTIC" )
        SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG_PEDANTIC" )
    ENDIF ()
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -DDEBUG -g -O0" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -DDEBUG -g -O0" )
    SET(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} -g -O0" )
    SET_WARNINGS()
ENDMACRO ()


# Macro to set the flags for optimized mode
MACRO ( SET_OPTIMIZED_MACROS )
    SET_COMPILER ()
    ADD_USER_FLAGS()
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -O2" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -O2" )
    SET(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} -O2" )
    SET_WARNINGS()
ENDMACRO ()


# Macro to copy a data file
MACRO ( COPY_TEST_DATA_FILE FILENAME )
    SET ( FILE_TO_COPY  ${CMAKE_CURRENT_SOURCE_DIR}/data/${FILENAME} )
    SET ( DESTINATION_NAME ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} )
    IF ( EXISTS ${FILE_TO_COPY} )
        CONFIGURE_FILE ( ${FILE_TO_COPY} ${DESTINATION_NAME} COPYONLY )
    ELSE()
        MESSAGE ( WARNING "Cannot find file: " ${FILE_TO_COPY} )
    ENDIF()
ENDMACRO ()


# Macro to copy a mesh file
MACRO (COPY_MESH_FILE MESHNAME)
    # Check the local data directory
    FILE ( GLOB MESHPATH ${CMAKE_CURRENT_SOURCE_DIR}/data/${MESHNAME} )
    # Check the AMP_DATA/meshes directory
    IF ( NOT MESHPATH )
        FILE ( GLOB_RECURSE MESHPATH ${AMP_DATA}/meshes/*/${MESHNAME} )
    ENDIF ()
    # Check the AMP_DATA/vvu directory
    IF ( NOT MESHPATH )
        FILE ( GLOB MESHPATH ${AMP_DATA}/vvu/meshes/${MESHNAME} )
    ENDIF ()
    # We have either found the mesh or failed
    IF ( NOT MESHPATH )
        MESSAGE ( WARNING "Cannot find mesh: " ${MESHNAME} )
    ELSE ()
        SET ( DESTINATION_NAME ${CMAKE_CURRENT_BINARY_DIR}/${MESHNAME} )
        CONFIGURE_FILE ( ${MESHPATH} ${DESTINATION_NAME} COPYONLY )
    ENDIF ()
ENDMACRO()


# Macro to add the dependencies and libraries to an executable
MACRO ( ADD_AMP_EXE_DEP EXEFILE )
    # Add the package dependencies
    IF ( AMP_TEST_LIB_EXISTS )
        ADD_DEPENDENCIES ( ${EXEFILE} ${PACKAGE_TEST_LIB} )
        TARGET_LINK_LIBRARIES ( ${EXEFILE} ${PACKAGE_TEST_LIB} )
    ENDIF ()
    # Add the executable to the dependencies of check and build-test
    ADD_DEPENDENCIES ( check ${EXEFILE} )
    ADD_DEPENDENCIES ( build-test ${EXEFILE} )
    # Add test dependencies
    #IF ( TEST_DEP_LIST )
    #    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${TEST_DEP_LIST} )
    #ENDIF()
    # Add the amp libraries
    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${AMP_LIBS} )
    # Add external libraries
    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${LIBMESH_LIBS} ${TRILINOS_LIBS} ${PETSC_LIBS} ${X11_LIBS} ${SILO_LIBS} ${HDF5_LIBS} ${HYPRE_LIBS} )
    IF ( ${USE_SUNDIALS} )
       TARGET_LINK_LIBRARIES ( ${EXEFILE} ${SUNDIALS_LIBS} )
    ENDIF  ()
    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${MPI_LINK_FLAGS} ${MPI_LIBRARIES})
    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${LAPACK_LIBS} ${BLAS_LIBS} )
    TARGET_LINK_LIBRARIES ( ${EXEFILE} ${COVERAGE_LIBS} ${LDLIBS} )
    TARGET_LINK_LIBRARIES ( ${EXEFILE} "-lz" )
ENDMACRO ()


# Add a executable
MACRO ( INSTALL_AMP_EXE EXE )
    FIND_FILES ()
    ADD_EXECUTABLE ( ${EXE} ${SOURCES} )
    ADD_AMP_EXE_DEP ( ${EXE} )
ENDMACRO()


MACRO ( ADD_FILES_TO_TEST_LIB FILENAMES )
  ADD_LIBRARY ( ${PACKAGE_TEST_LIB} ${FILENAMES} )
  IF ( TEST_DEP_LIST )
      TARGET_LINK_LIBRARIES ( ${PACKAGE_TEST_LIB} ${TEST_DEP_LIST} )
  ENDIF()
  SET ( AMP_TEST_LIB_EXISTS ${PACKAGE_TEST_LIB} )
ENDMACRO ()


# Macro to add a provisional test
MACRO ( ADD_AMP_PROVISIONAL_TEST EXEFILE )
    # Check if test has already been added
    get_target_property(tmp ${EXEFILE} LOCATION)
    IF ( NOT tmp )
        # The target has not been added
        SET ( CXXFILE ${EXEFILE}.cc )
        SET ( TESTS_SO_FAR ${TESTS_SO_FAR} ${EXEFILE} )
        ADD_EXECUTABLE ( ${EXEFILE} EXCLUDE_FROM_ALL ${CXXFILE} )
        ADD_AMP_EXE_DEP( ${EXEFILE} )
    ELSEIF ( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE}" )
        # The correct target has already been added
    ELSE()
        # We are trying to add 2 different tests with the same name
        MESSAGE ( "Existing test: ${tmp}" )
        MESSAGE ( "New test:      ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE}" )
        MESSAGE ( FATAL_ERROR "Trying to add 2 different tests with the same name" )
    ENDIF()
ENDMACRO ()

# Macro to create the test name
MACRO ( CREATE_TEST_NAME TEST ${ARGN} )
    IF ( PACKAGE )
        SET ( TESTNAME "${PACKAGE}::${TEST}" )
    ELSE()
        SET ( TESTNAME "${TEST}" )
    ENDIF()
    foreach ( tmp ${ARGN})
        SET ( TESTNAME "${TESTNAME}--${tmp}")
    endforeach()
    # STRING(REGEX REPLACE "--" "-" TESTNAME ${TESTNAME} )
ENDMACRO ()


# Add a executable as a test
MACRO ( ADD_AMP_TEST EXEFILE ${ARGN} )
    ADD_AMP_PROVISIONAL_TEST ( ${EXEFILE} )
    CREATE_TEST_NAME( ${EXEFILE} ${ARGN} )
    IF ( USE_MPI_FOR_SERIAL_TESTS )
        ADD_TEST ( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
    ELSE()
        ADD_TEST ( ${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
    ENDIF()
    SET_TESTS_PROPERTIES ( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION ".*FAILED.*" )
ENDMACRO ()

# Add a executable as a weekly test
MACRO ( ADD_AMP_WEEKLY_TEST EXEFILE PROCS ${ARGN} )
    ADD_AMP_PROVISIONAL_TEST ( ${EXEFILE} )
    IF ( ${PROCS} STREQUAL "1" )
        CREATE_TEST_NAME( "${EXEFILE}_WEEKLY" ${ARGN} )
        IF ( USE_MPI_FOR_SERIAL_TESTS )
            ADD_TEST ( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
        ELSE()
            ADD_TEST ( ${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
        ENDIF()
    ELSEIF ( USE_MPI )
        CREATE_TEST_NAME( "${EXEFILE}_${PROCS}procs_WEEKLY" ${ARGN} )
        ADD_TEST ( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
    ENDIF()
    SET_TESTS_PROPERTIES ( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION ".*FAILED.*" )
ENDMACRO ()

# Add a executable as a parallel test
MACRO ( ADD_AMP_TEST_PARALLEL EXEFILE PROCS ${ARGN} )
    ADD_AMP_PROVISIONAL_TEST ( ${EXEFILE} )
    IF ( USE_MPI )
        CREATE_TEST_NAME( "${EXEFILE}_${PROCS}procs" ${ARGN} )
        ADD_TEST ( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
        SET_TESTS_PROPERTIES ( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION ".*FAILED.*" )
    ENDIF()
ENDMACRO ()

# Add a executable as a parallel 1, 2, 4 processor test
MACRO ( ADD_AMP_TEST_1_2_4 EXENAME ${ARGN} )
    ADD_AMP_TEST ( ${EXENAME} ${ARGN} )
    ADD_AMP_TEST_PARALLEL ( ${EXENAME} 2 ${ARGN} )
    ADD_AMP_TEST_PARALLEL ( ${EXENAME} 4 ${ARGN} )
ENDMACRO ()


# Macro to check if a flag is enabled
MACRO ( CHECK_ENABLE_FLAG FLAG DEFAULT )
    IF ( NOT DEFINED ${FLAG} )
        SET ( ${FLAG} ${DEFAULT} )
    ELSEIF ( ( ${${FLAG}} STREQUAL "false" ) OR ( ${${FLAG}} STREQUAL "0" ) )
        SET ( ${FLAG} 0 )
    ELSEIF ( ( ${${FLAG}} STREQUAL "true" ) OR ( ${${FLAG}} STREQUAL "1" ) )
        SET ( ${FLAG} 1 )
    ELSE()
        MESSAGE ( "Bad value for ${FLAG}; use true or false" )
        MESSAGE ( ${USE_ORIGEN} )
    ENDIF ()
ENDMACRO ()


# add custom target distclean
# cleans and removes cmake generated files etc.
MACRO ( ADD_DISTCLEAN )
IF (UNIX)
  ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
  SET(DISTCLEANED
    cmake.depends
    cmake.check_depends
    CMakeCache.txt
    CMakeFiles
    cmake.check_cache
    *.cmake
    compile.log
    Doxyfile
    Makefile
    core core.*
    src
    ampdir
    DartConfiguration.tcl
    Testing
    install_manifest.txt
  )
  ADD_CUSTOM_COMMAND(
    DEPENDS clean
    COMMENT "distribution clean"
    COMMAND rm
    ARGS    -Rf CMakeTmp ${DISTCLEANED}
    TARGET  distclean
  )
ENDIF(UNIX)
ENDMACRO ()


# Save the necessary cmake variables to a file for applications to load
# Note: we need to save the external libraries in the same order as AMP for consistency
MACRO ( SAVE_CMAKE_FLAGS )
    # Write the header (comments)
    file(WRITE  ${AMP_INSTALL_DIR}/amp.cmake "# This is a automatically generate file to include AMP within another application\n\n" )
    # Write the compilers and compile flags
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the compilers and compile flags\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_C_COMPILER ${CMAKE_C_COMPILER})\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER})\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_Fortran_COMPILER ${CMAKE_Fortran_COMPILER})\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_C_FLAGS \"${CMAKE_C_FLAGS}\")\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS}\")\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(CMAKE_Fortran_FLAGS \"${CMAKE_Fortran_FLAGS}\")\n" )
    # Write the AMP_DATA and AMP_SOURCE paths
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the AMP data and source directories\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_DATA ${AMP_DATA})\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_SOURCE ${AMP_SOURCE_DIR})\n" )
    # Create the AMP libraries and include paths
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Set the AMP libraries\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(AMP_LIBS ${AMP_LIBS})\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES ( ${AMP_TRUNK}/external/boost/include )\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES ( ${AMP_INSTALL_DIR}/include )\n" )
    IF ( USE_AMP_UTILS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_UTILS ) \n" )
    ENDIF()
    IF ( USE_AMP_MESH )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_MESH ) \n" )
    ENDIF()
    IF ( USE_AMP_DISCRETIZATION )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_DISCRETIZATION ) \n" )
    ENDIF()
    IF ( USE_AMP_VECTORS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_VECTORS ) \n" )
    ENDIF()
    IF ( USE_AMP_MATRICIES )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_MATRICIES ) \n" )
    ENDIF()
    IF ( USE_AMP_MATERIALS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_MATERIALS ) \n" )
    ENDIF()
    IF ( USE_AMP_OPERATORS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_OPERATORS ) \n" )
    ENDIF()
    IF ( USE_AMP_SOLVERS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_SOLVERS ) \n" )
    ENDIF()
    IF ( USE_AMP_TIME_INTEGRATORS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_AMP_TIME_INTEGRATORS ) \n" )
    ENDIF()
    # Create the external libraries and include paths in the order they are linked in AMP
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS )\n" )
    # Add boost
    IF ( USE_BOOST )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add boost\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${BOOST_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_BOOST 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_BOOST ) \n" )
    ENDIF()
    # Add Libmesh
    IF ( USE_LIBMESH )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Libmesh\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${LIBMESH_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${LIBMESH_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -DLIBMESH_ENABLE_PARMESH )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_LIBMESH 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_LIBMESH ) \n" )
    ENDIF()
    # Add Trilinos
    IF ( USE_TRILINOS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Trilinos\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${TRILINOS_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${TRILINOS_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_TRILINOS 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_TRILINOS ) \n" )
    ENDIF()
    # Add PETsc
    IF ( USE_PETSC )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add PETsc\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${PETSC_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${PETSC_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_PETSC 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_PETSC ) \n" )
    ENDIF()
    # Add Sundials
    IF ( USE_SUNDIALS )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add Sundials\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${SUNDIALS_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${SUNDIALS_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_SUNDIALS 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_SUNDIALS ) \n" )
    ENDIF()
    # Add Silo
    IF ( USE_SILO )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add silo\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${SILO_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${SILO_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_SILO 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_SILO ) \n" )
    ENDIF()
    # Add Hypre
    IF ( USE_HYPRE )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add hypre\n" )
        # file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${HYPRE_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${HYPRE_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_HYPRE 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_HYPRE ) \n" )
    ENDIF()
    # Add X11
    IF ( USE_X11 )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add X11\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${X11_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${X11_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_X11 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_X11 ) \n" )
    ENDIF()
    # Add HDF5
    IF ( USE_HDF5 )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add HDF5\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${HDF5_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${HDF5_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_HDF5 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_HDF5 ) \n" )
    ENDIF()
    # Add MPI
    IF ( USE_MPI )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add MPI\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "INCLUDE_DIRECTORIES( ${MPI_INCLUDE} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS}  ${MPI_LINK_FLAGS} ${MPI_LIBRARIES} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( USE_MPI 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -D USE_MPI ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(MPIEXEC ${MPIEXEC} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(MPIEXEC_NUMPROC_FLAG ${MPIEXEC_NUMPROC_FLAG} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_MPI_FOR_SERIAL_TESTS ${USE_MPI_FOR_SERIAL_TESTS} )\n" )
    ELSE()
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET(USE_MPI_FOR_SERIAL_TESTS 0 )\n" )
    ENDIF()
    # Add LAPACK and BLAS
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add LAPACK/BLAS\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS}  ${LAPACK_LIBS} ${BLAS_LIBS} )\n" )
    # Add coverage
    IF ( ENABLE_GCOV )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add coverage flags\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} ${COVERAGE_LIBS} )\n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET ( ENABLE_GCOV 1 ) \n" )
        file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "ADD_DEFINITIONS ( -fprofile-arcs -ftest-coverage ) \n" )
    ENDIF ()
    # Add misc flags
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "# Add misc flags\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "SET( EXTERNAL_LIBS $""{EXTERNAL_LIBS} \"-lz\" )\n" )
    file(APPEND ${AMP_INSTALL_DIR}/amp.cmake "\n" )
ENDMACRO ()


