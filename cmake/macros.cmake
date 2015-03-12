INCLUDE(CheckCSourceCompiles)
IF ( NOT TEST_FAIL_REGULAR_EXPRESSION )
    # Note: we cannot check for "handles are still allocated" due to PETSc.  See static variable
    #   Petsc_Reduction_keyval on line 234 of comb.c
    #SET( TEST_FAIL_REGULAR_EXPRESSION "(FAILED)|(leaked context IDs detected)|(handles are still allocated)" )
    SET( TEST_FAIL_REGULAR_EXPRESSION "(FAILED)" )
ENDIF()


# Check that the PROJ and ${PROJ}_INSTALL_DIR variables are set 
# These variables are used to generate the ADD_PROJ_TEST macros
IF ( NOT PROJ )
    MESSAGE(FATAL_ERROR "PROJ must be set before including macros.cmake")
ENDIF()
IF ( NOT ${PROJ}_INSTALL_DIR )
    MESSAGE(FATAL_ERROR "${PROJ}_INSTALL_DIR must be set before including macros.cmake")
ENDIF()


# Add some default targets if they do not exist
IF ( NOT TARGET copy-${PROJ}-Data )
    ADD_CUSTOM_TARGET( copy-${PROJ}-Data ALL )
ENDIF()
IF ( NOT TARGET copy-${PROJ}-include )
    ADD_CUSTOM_TARGET ( copy-${PROJ}-include ALL )
ENDIF()


# Macro to set a global variable
MACRO(GLOBAL_SET VARNAME)
  SET(${VARNAME} ${ARGN} CACHE INTERNAL "")
ENDMACRO()


# Macro to print all variables
MACRO( PRINT_ALL_VARIABLES )
    GET_CMAKE_PROPERTY(_variableNames VARIABLES)
    FOREACH(_variableName ${_variableNames})
        message(STATUS "${_variableName}=${${_variableName}}")
    ENDFOREACH()
ENDMACRO()


# CMake assert
MACRO(ASSERT test comment)
    IF (NOT ${test})
        MESSSAGE(FATAL_ERROR "Assertion failed: ${comment}")
    ENDIF(NOT ${test})
ENDMACRO(ASSERT)


# Macro to convert a m4 file
# This command converts a file of the format "global_path/file.m4"
# and convertes it to file.F.  It also requires the path.  
MACRO( CONVERT_M4_FORTRAN IN LOCAL_PATH OUT_PATH )
    STRING(REGEX REPLACE ${LOCAL_PATH} "" OUT ${IN} )
    STRING(REGEX REPLACE "/" "" OUT ${OUT} )
    STRING(REGEX REPLACE "(.fm4)|(.m4)" ".F" OUT "${CMAKE_CURRENT_BINARY_DIR}/${OUT_PATH}/${OUT}" )
    IF ( NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${OUT_PATH}" )
        FILE(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${OUT_PATH}" )    
    ENDIF()
    CONFIGURE_FILE ( ${IN} ${IN} COPYONLY )
    IF ("${CMAKE_GENERATOR}" STREQUAL "Xcode")
        STRING(REGEX REPLACE ".F" ".o" OUT2 "${OUT}" )
        STRING(REGEX REPLACE ";" " " COMPILE_CMD "${CMAKE_Fortran_COMPILER} -c ${OUT} ${CMAKE_Fortran_FLAGS} -o ${OUT2}")
        STRING(REGEX REPLACE "\\\\" "" COMPILE_CMD "${COMPILE_CMD}")
        MESSAGE("COMPILE_CMD = ${COMPILE_CMD}")
        SET( COMPILE_CMD ${COMPILE_CMD} )
        add_custom_command(
            OUTPUT ${OUT2}
            COMMAND m4 -I${LOCAL_PATH} -I${SAMRAI_FORTDIR} ${M4DIRS} ${IN} > ${OUT}
            COMMAND ${COMPILE_CMD}
            DEPENDS ${IN}
            )
        set_source_files_properties(${OUT2} PROPERTIES GENERATED true)
        SET( SOURCES ${SOURCES} "${OUT2}" )
     ELSE()
        add_custom_command(
            OUTPUT ${OUT}
            COMMAND m4 -I${LOCAL_PATH} -I${SAMRAI_FORTDIR} ${M4DIRS} ${M4_OPTIONS} ${IN} > ${OUT}
            DEPENDS ${IN}
            )
         set_source_files_properties(${OUT} PROPERTIES GENERATED true)
         SET( SOURCES ${SOURCES} "${OUT}" )
     ENDIF()
ENDMACRO()


# Add a package to the project's library
MACRO( ADD_${PROJ}_LIBRARY PACKAGE )
    ADD_SUBDIRECTORY( ${PACKAGE} )
ENDMACRO()


# Add a project executable
MACRO( ADD_${PROJ}_EXECUTABLE EXEFILE )
    ADD_PROJ_PROVISIONAL_TEST( ${EXEFILE} )
    INSTALL( TARGETS ${EXEFILE} DESTINATION ${${PROJ}_INSTALL_DIR}/bin )
ENDMACRO()


# Initialize a package
MACRO (BEGIN_PACKAGE_CONFIG PACKAGE)
    SET( HEADERS "" )
    SET( CXXSOURCES "" )
    SET( CSOURCES "" )
    SET( FSOURCES "" )
    SET( M4FSOURCES "" )
    SET( SOURCES "" )
    SET( CURPACKAGE ${PACKAGE} )
ENDMACRO ()


# Find the source files
MACRO (FIND_FILES)
    # Find the C/C++ headers
    SET( T_HEADERS "" )
    FILE( GLOB T_HEADERS "*.h" "*.H" "*.hh" "*.hpp" "*.I" )
    # Find the CUDA sources
    SET( T_CUDASOURCES "" )
    FILE( GLOB T_CUDASOURCES "*.cu" )
    # Find the C sources
    SET( T_CSOURCES "" )
    FILE( GLOB T_CSOURCES "*.c" )
    # Find the C++ sources
    SET( T_CXXSOURCES "" )
    FILE( GLOB T_CXXSOURCES "*.cc" "*.cpp" "*.cxx" "*.C" )
    # Find the Fortran sources
    SET( T_FSOURCES "" )
    FILE( GLOB T_FSOURCES "*.f" "*.f90" "*.F" "*.F90" )
    # Find the m4 fortran source (and convert)
    SET( T_M4FSOURCES "" )
    FILE( GLOB T_M4FSOURCES "*.m4" "*.fm4" )
    FOREACH( m4file ${T_M4FSOURCES} )
        CONVERT_M4_FORTRAN( ${m4file} ${CMAKE_CURRENT_SOURCE_DIR} "" )
    ENDFOREACH()
    # Add all found files to the current lists
    SET( HEADERS ${HEADERS} ${T_HEADERS} )
    SET( CXXSOURCES ${CXXSOURCES} ${T_CXXSOURCES} )
    SET( CUDASOURCES ${CUDASOURCES} ${T_CUDASOURCES} )
    SET( CSOURCES ${CSOURCES} ${T_CSOURCES} )
    SET( FSOURCES ${FSOURCES} ${T_FSOURCES} )
    SET( M4FSOURCES ${M4FSOURCES} ${T_M4FSOURCES} )
    SET( SOURCES ${SOURCES} ${T_CXXSOURCES} ${T_CSOURCES} ${T_FSOURCES} ${T_M4FSOURCES} )
ENDMACRO()


# Find the source files
MACRO( FIND_FILES_PATH IN_PATH )
    # Find the C/C++ headers
    SET( T_HEADERS "" )
    FILE( GLOB T_HEADERS "${IN_PATH}/*.h" "${IN_PATH}/*.H" "${IN_PATH}/*.hh" "${IN_PATH}/*.hpp" "${IN_PATH}/*.I" )
    # Find the CUDA sources
    SET( T_CUDASOURCES "" )
    FILE( GLOB T_CUDASOURCES "${IN_PATH}/*.cu" )
    # Find the C sources
    SET( T_CSOURCES "" )
    FILE( GLOB T_CSOURCES "${IN_PATH}/*.c" )
    # Find the C++ sources
    SET( T_CXXSOURCES "" )
    FILE( GLOB T_CXXSOURCES "${IN_PATH}/*.cc" "${IN_PATH}/*.cpp" "${IN_PATH}/*.cxx" "${IN_PATH}/*.C" )
    # Find the Fortran sources
    SET( T_FSOURCES "" )
    FILE( GLOB T_FSOURCES "${IN_PATH}/*.f" "${IN_PATH}/*.f90" "${IN_PATH}/*.F" "${IN_PATH}/*.F90" )
    # Find the m4 fortran source (and convert)
    SET( T_M4FSOURCES "" )
    FILE( GLOB T_M4FSOURCES "${IN_PATH}/*.m4" "${IN_PATH}/*.fm4" )
    FOREACH( m4file ${T_M4FSOURCES} )
        CONVERT_M4_FORTRAN( ${m4file} ${CMAKE_CURRENT_SOURCE_DIR}/${IN_PATH} ${IN_PATH} )
    ENDFOREACH ()
    # Add all found files to the current lists
    SET( HEADERS ${HEADERS} ${T_HEADERS} )
    SET( CXXSOURCES ${CXXSOURCES} ${T_CXXSOURCES} )
    SET( CUDASOURCES ${CUDASOURCES} ${T_CUDASOURCES} )
    SET( CSOURCES ${CSOURCES} ${T_CSOURCES} )
    SET( FSOURCES ${FSOURCES} ${T_FSOURCES} )
    SET( SOURCES ${SOURCES} ${T_CXXSOURCES} ${T_CSOURCES} ${T_FSOURCES} )
ENDMACRO()


# Add a subdirectory
MACRO( ADD_PACKAGE_SUBDIRECTORY SUBDIR )
    FIND_FILES_PATH( ${SUBDIR} )
ENDMACRO()


# Install a package
MACRO( INSTALL_${PROJ}_TARGET PACKAGE )
    # Find all files in the current directory
    FIND_FILES()
    # Copy the header files to the include path
    FILE( GLOB HFILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${HEADERS} )
    STRING(REGEX REPLACE "${${PROJ}_SOURCE_DIR}/" "" COPY_TARGET "copy-${PROJ}-${CMAKE_CURRENT_SOURCE_DIR}-include" )
    STRING(REGEX REPLACE "/" "-" COPY_TARGET ${COPY_TARGET} )
    IF( NOT TARGET ${COPY_TARGET} )
        ADD_CUSTOM_TARGET( ${COPY_TARGET} ALL )
        ADD_DEPENDENCIES( copy-${PROJ}-include ${COPY_TARGET} )
    ENDIF()
    FOREACH( HFILE ${HFILES} )
        SET( SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${HFILE}" )
        SET( DST_FILE "${${PROJ}_INSTALL_DIR}/include/${CURPACKAGE}/${HFILE}" )
        ADD_CUSTOM_COMMAND(TARGET ${COPY_TARGET} 
            PRE_BUILD 
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC_FILE}" "${DST_FILE}"
            DEPENDS "${SRC_FILE}"
        )
    ENDFOREACH()
    # Add the library and install the package
    IF ( NOT ONLY_BUILD_DOCS AND ( SOURCES OR CUDASOURCES ) )
        IF( USE_CUDA )
            CUDA_COMPILE( CUBINS ${CUDASOURCES} )
        ENDIF()
        ADD_LIBRARY( ${PACKAGE} ${LIB_TYPE} ${SOURCES} ${CUBINS} )
        IF ( TARGET write_repo_version )
            ADD_DEPENDENCIES( ${PACKAGE} write_repo_version )
        ENDIF()
        ADD_DEPENDENCIES ( ${PACKAGE} copy-${PROJ}-include )
        INSTALL( TARGETS ${PACKAGE} DESTINATION ${${PROJ}_INSTALL_DIR}/lib )
    ELSE()
        ADD_CUSTOM_TARGET( ${PACKAGE} ALL )
    ENDIF()
    INSTALL( FILES ${HEADERS} DESTINATION "${${PROJ}_INSTALL_DIR}/include/${PACKAGE}" )
    # Clear the sources
    SET( HEADERS "" )
    SET( CSOURCES "" )
    SET( CXXSOURCES "" )
ENDMACRO()


# Macro to verify that a variable has been set
MACRO( VERIFY_VARIABLE VARIABLE_NAME )
    IF ( NOT ${VARIABLE_NAME} )
        MESSAGE( FATAL_ERROR "PLease set: " ${VARIABLE_NAME} )
    ENDIF()
ENDMACRO()


# Macro to verify that a path has been set
MACRO( VERIFY_PATH PATH_NAME )
    IF ("${PATH_NAME}" STREQUAL "")
        MESSAGE ( FATAL_ERROR "Path is not set: ${PATH_NAME}" )
    ENDIF()
    IF ( NOT EXISTS "${PATH_NAME}" )
        MESSAGE( FATAL_ERROR "Path does not exist: ${PATH_NAME}" )
    ENDIF()
ENDMACRO()


# Macro to tell cmake to use static libraries
MACRO( SET_STATIC_FLAGS )
    # Remove extra library links
    set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
    set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
    set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
    set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
    set(CMAKE_SHARED_LINKER_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
    # Add the static flag if necessary
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static") # Add static flag
    SET(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -static ") 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static ")
    SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "-static")                # Add static flag
    SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "-static")              # Add static flag
ENDMACRO()


# Macro to identify the compiler
MACRO( SET_COMPILER )
    # SET the C/C++ compiler
    IF ( CMAKE_C_COMPILER_WORKS OR CMAKE_C_COMPILER_WORKS )
        IF( CMAKE_COMPILE_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
            SET( USING_GCC TRUE )
            ADD_DEFINITIONS( "-D USING_GCC" )
            MESSAGE("Using gcc")
        ELSEIF( MSVC OR MSVC_IDE OR MSVC60 OR MSVC70 OR MSVC71 OR MSVC80 OR CMAKE_COMPILER_2005 OR MSVC90 OR MSVC10 )
            IF( NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
                MESSAGE( FATAL_ERROR "Using microsoft compilers on non-windows system?" )
            ENDIF()
            SET( USING_MSVC TRUE )
            ADD_DEFINITIONS( "-D USING_MSVC" )
            MESSAGE("Using Microsoft")
        ELSEIF( (${CMAKE_C_COMPILER_ID} MATCHES "Intel") OR (${CMAKE_CXX_COMPILER_ID} MATCHES "Intel") ) 
            SET(USING_ICC TRUE)
            ADD_DEFINITIONS( "-D USING_ICC" )
            MESSAGE("Using icc")
        ELSEIF( ${CMAKE_C_COMPILER_ID} MATCHES "PGI")
            SET(USING_PGCC TRUE)
            ADD_DEFINITIONS( "-D USING_PGCC" )
            MESSAGE("Using pgCC")
        ELSEIF( (${CMAKE_C_COMPILER_ID} MATCHES "CRAY") OR (${CMAKE_C_COMPILER_ID} MATCHES "Cray") )
            SET(USING_CRAY TRUE)
            ADD_DEFINITIONS( "-D USING_CRAY" )
            MESSAGE("Using Cray")
        ELSEIF( (${CMAKE_C_COMPILER_ID} MATCHES "CLANG") OR (${CMAKE_C_COMPILER_ID} MATCHES "Clang") )
            SET(USING_CLANG TRUE)
            ADD_DEFINITIONS( "-D USING_CLANG" )
            MESSAGE("Using Clang")
        ELSE()
            SET(USING_DEFAULT TRUE)
            MESSAGE("${CMAKE_C_COMPILER_ID}")
            MESSAGE("Unknown C/C++ compiler, default flags will be used")
        ENDIF()
    ENDIF()
    # SET the Fortran++ compiler
    IF ( CMAKE_Fortran_COMPILER_WORKS )
        IF( CMAKE_COMPILE_IS_GFORTRAN OR (${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU") )
            SET( USING_GFORTRAN TRUE )
            MESSAGE("Using gfortran")
        ELSEIF ( (${CMAKE_Fortran_COMPILER_ID} MATCHES "Intel") ) 
            SET(USING_IFORT TRUE)
            MESSAGE("Using ifort")
        ELSEIF ( ${CMAKE_Fortran_COMPILER_ID} MATCHES "PGI")
            SET(USING_PGF90 TRUE)
            MESSAGE("Using pgf90")
        ELSE()
            SET(USING_DEFAULT TRUE)
            MESSAGE("${CMAKE_Fortran_COMPILER_ID}")
            MESSAGE("Unknown Fortran compiler, default flags will be used")
        ENDIF()
    ENDIF()
ENDMACRO()


# Macro to set the proper warnings
MACRO( SET_WARNINGS )
  IF ( USING_GCC )
    # Add gcc specific compiler options
    SET(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -Wall -Wextra") 
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Woverloaded-virtual")
  ELSEIF ( USING_MICROSOFT )
    # Add Microsoft specifc compiler options
    SET(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} /D _SCL_SECURE_NO_WARNINGS /D _CRT_SECURE_NO_WARNINGS /D _ITERATOR_DEBUG_LEVEL=0" )
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D _SCL_SECURE_NO_WARNINGS /D _CRT_SECURE_NO_WARNINGS /D _ITERATOR_DEBUG_LEVEL=0" )
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
    #   1572: floating-point equality and inequality comparisons are unreliable
    #         LibMesh warnings
    #   1599: declaration hides parameter 
    #         LibMesh warnings
    #   2259: non-pointer conversion from "int" to "unsigned char" may lose significant bits
    #         This is bad practice, use an explict coversion instead
    #         Unfortunatelly many of these come from LibMesh
    #   6843: A dummy argument with an explicit INTENT(OUT) declaration is not given an explicit value.
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -Wall" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall" )
  ELSEIF ( USING_CRAY )
    # Add default compiler options
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")
  ELSEIF ( USING_PGCC )
    # Add default compiler options
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -lpthread")
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -lpthread")
  ELSEIF ( USING_CLANG )
    # Add default compiler options
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS} -Wall")
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -Wall")
  ELSEIF ( USING_DEFAULT )
    # Add default compiler options
    SET(CMAKE_C_FLAGS     " ${CMAKE_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS}")
  ENDIF()
ENDMACRO ()


# Macro to add user compile flags
MACRO( ADD_USER_FLAGS )
    SET(CMAKE_C_FLAGS   " ${CMAKE_C_FLAGS} ${CFLAGS} ${CFLAGS_EXTRA}" )
    SET(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CXXFLAGS} ${CXXFLAGS_EXTRA}" )
    SET(CMAKE_Fortran_FLAGS " ${CMAKE_Fortran_FLAGS} ${FFLAGS} ${FFLAGS_EXTRA}" )
ENDMACRO()


# Macro to set the flags for debug mode
MACRO( SET_COMPILER_FLAGS )
    # Initilaize the compiler
    SET_COMPILER()
    # Set the default flags for each build type
    IF ( USING_MSVC )
        SET(CMAKE_C_FLAGS_DEBUG       "-D_DEBUG /DEBUG /Od /EHsc /MDd /Zi /Z7" )
        SET(CMAKE_C_FLAGS_RELEASE     "/O2 /EHsc /MD"                      )
        SET(CMAKE_CXX_FLAGS_DEBUG     "-D_DEBUG /DEBUG /Od /EHsc /MDd /Zi /Z7" )
        SET(CMAKE_CXX_FLAGS_RELEASE   "/O2 /EHsc /MD"                      )
        SET(CMAKE_Fortran_FLAGS_DEBUG ""                                   )
        SET(CMAKE_Fortran_FLAGS_RELEASE ""                                 )
    ELSE()
        SET(CMAKE_C_FLAGS_DEBUG       "-g -D_DEBUG -O0" )
        SET(CMAKE_C_FLAGS_RELEASE     "-O2"             )
        SET(CMAKE_CXX_FLAGS_DEBUG     "-g -D_DEBUG -O0" )
        SET(CMAKE_CXX_FLAGS_RELEASE   "-O2"             )
        SET(CMAKE_Fortran_FLAGS_DEBUG "-g -O0"          )
        SET(CMAKE_Fortran_FLAGS_RELEASE "-O2"           )
    ENDIF()
    IF ( NOT DISABLE_GXX_DEBUG )
        SET(CMAKE_C_FLAGS_DEBUG   " ${CMAKE_C_FLAGS_DEBUG}   -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" )
        SET(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" )
    ENDIF()
    # Set the compiler flags to use
    IF ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
        SET(CMAKE_C_FLAGS       ${CMAKE_C_FLAGS_DEBUG}       )
        SET(CMAKE_CXX_FLAGS     ${CMAKE_CXX_FLAGS_DEBUG}     )
        SET(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS_DEBUG} )
    ELSEIF ( ${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
        SET(CMAKE_C_FLAGS       ${CMAKE_C_FLAGS_RELEASE}       )
        SET(CMAKE_CXX_FLAGS     ${CMAKE_CXX_FLAGS_RELEASE}     )
        SET(CMAKE_Fortran_FLAGS ${CMAKE_Fortran_FLAGS_RELEASE} )
    ELSEIF ( ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" OR ${CMAKE_BUILD_TYPE} STREQUAL "RELWITHDEBINFO")
        SET(CMAKE_C_FLAGS       "-g ${CMAKE_C_FLAGS_RELEASE}"       )
        SET(CMAKE_CXX_FLAGS     "-g ${CMAKE_CXX_FLAGS_RELEASE}"     )
        SET(CMAKE_Fortran_FLAGS "-g ${CMAKE_Fortran_FLAGS_RELEASE}" )
    ELSE()
        MESSAGE(FATAL_ERROR "Unknown value for CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
    ENDIF()
    # Add the user flags
    ADD_USER_FLAGS()
    # Set the warnings to use
    SET_WARNINGS()
ENDMACRO()


# Macro to copy data file at build time
MACRO( COPY_DATA_FILE SRC_FILE DST_FILE )
    STRING(REGEX REPLACE "${${PROJ}_SOURCE_DIR}/" "" COPY_TARGET "copy-${PROJ}-${CMAKE_CURRENT_SOURCE_DIR}" )
    STRING(REGEX REPLACE "/" "-" COPY_TARGET ${COPY_TARGET} )
    IF ( NOT TARGET ${COPY_TARGET} )
        ADD_CUSTOM_TARGET( ${COPY_TARGET} ALL )
        ADD_DEPENDENCIES( copy-${PROJ}-Data ${COPY_TARGET} )
    ENDIF()
    ADD_CUSTOM_COMMAND( TARGET ${COPY_TARGET} 
        PRE_BUILD 
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC_FILE}" "${DST_FILE}"
        DEPENDS "${SRC_FILE}"
    )
ENDMACRO()


# Macro to copy a data or input file
MACRO ( COPY_TEST_DATA_FILE FILENAME ${ARGN} )
    SET( FILE_TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}" )
    IF ( NOT EXISTS "${FILE_TO_COPY}" )
        SET( FILE_TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/data/${FILENAME}" )
    ENDIF()
    IF ( NOT EXISTS "${FILE_TO_COPY}" )
        SET( FILE_TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/inputs/${FILENAME}" )
    ENDIF()
    FOREACH( tmp ${ARGN} )
        IF ( NOT EXISTS "${FILE_TO_COPY}" )
            SET( FILE_TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/${tmp}/${FILENAME}" )
        ENDIF()
    ENDFOREACH()
    SET( DESTINATION_NAME "${CMAKE_CURRENT_BINARY_DIR}/${FILENAME}" )
    IF ( EXISTS "${FILE_TO_COPY}" )
        COPY_DATA_FILE( ${FILE_TO_COPY} ${DESTINATION_NAME} )
    ELSE()
        MESSAGE( WARNING "Cannot find file: " ${FILE_TO_COPY} )
    ENDIF()
ENDMACRO ()


# Macro to copy a data file
MACRO ( RENAME_TEST_DATA_FILE SRC DST )
    SET( FILE_TO_COPY  ${CMAKE_CURRENT_SOURCE_DIR}/data/${SRC} )
    SET( DESTINATION_NAME ${CMAKE_CURRENT_BINARY_DIR}/${DST} )
    IF ( EXISTS ${FILE_TO_COPY} )
        COPY_DATA_FILE( ${FILE_TO_COPY} ${DESTINATION_NAME} )
    ELSE()
        MESSAGE( WARNING "Cannot find file: ${FILE_TO_COPY}" )
    ENDIF()
ENDMACRO ()


# Macro to copy a data file
FUNCTION( COPY_EXAMPLE_DATA_FILE FILENAME )
    SET( FILE_TO_COPY  ${CMAKE_CURRENT_SOURCE_DIR}/data/${FILENAME} )
    SET( DESTINATION1 ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} )
    SET( DESTINATION2 ${EXAMPLE_INSTALL_DIR}/${FILENAME} )
    IF ( EXISTS ${FILE_TO_COPY} )
        COPY_DATA_FILE( ${FILE_TO_COPY} ${DESTINATION1} )
        COPY_DATA_FILE( ${FILE_TO_COPY} ${DESTINATION2} )
    ELSE()
        MESSAGE( WARNING "Cannot find file: " ${FILE_TO_COPY} )
    ENDIF()
ENDFUNCTION()


# Macro to copy a mesh file
MACRO( COPY_MESH_FILE MESHNAME )
    # Check the local data directory
    FILE( GLOB MESHPATH "${CMAKE_CURRENT_SOURCE_DIR}/data/${MESHNAME}" )
    # Check the AMP_DATA directory
    IF ( NOT MESHPATH )
        FILE( GLOB MESHPATH "${AMP_DATA}/${MESHNAME}" )
    ENDIF()
    # Check the AMP_DATA/vvu directory
    IF ( NOT MESHPATH )
        FILE( GLOB MESHPATH "${AMP_DATA}/vvu/meshes/${MESHNAME}" )
    ENDIF()
    # Check the AMP_DATA/meshes directory
    IF ( NOT MESHPATH )
        FILE( GLOB MESHPATH "${AMP_DATA}/meshes/TestMeshes/${MESHNAME}" )
    ENDIF()
    IF ( NOT MESHPATH )
        FILE( GLOB_RECURSE MESHPATH "${AMP_DATA}/meshes/*/${MESHNAME}" )
    ENDIF()
    # We have either found the mesh or failed
    IF ( NOT MESHPATH )
        MESSAGE ( WARNING "Cannot find mesh: " ${MESHNAME} )
    ELSE ()
        SET( MESHPATH2 )
        FOREACH( tmp ${MESHPATH} )
            SET( MESHPATH2 "${tmp}" )
        ENDFOREACH()
        STRING(REGEX REPLACE "//${MESHNAME}" "" MESHPATH "${MESHPATH2}" )
        STRING(REGEX REPLACE "${MESHNAME}" "" MESHPATH "${MESHPATH}" )
        COPY_DATA_FILE( "${MESHPATH}/${MESHNAME}" "${CMAKE_CURRENT_BINARY_DIR}/${MESHNAME}" )
    ENDIF()
ENDMACRO()


# Macro to add the dependencies and libraries to an executable
MACRO( ADD_PROJ_EXE_DEP EXE )
    # Add the package dependencies
    IF( ${PROJ}_TEST_LIB_EXISTS )
        ADD_DEPENDENCIES ( ${EXE} ${PACKAGE_TEST_LIB} )
        TARGET_LINK_LIBRARIES ( ${EXE} ${PACKAGE_TEST_LIB} )
    ENDIF()
    # Add the executable to the dependencies of check and build-test
    ADD_DEPENDENCIES( check ${EXE} )
    ADD_DEPENDENCIES( build-test ${EXE} )
    # Add the file copy targets to the dependency list
    IF ( TARGET copy-${PROJ}-Data )
        ADD_DEPENDENCIES( ${EXE} copy-${PROJ}-Data )
    ENDIF()
    # Add the project libraries
    TARGET_LINK_LIBRARIES( ${EXE} ${${PROJ}_LIBS} ${${PROJ}_LIBS} )
    TARGET_LINK_LIBRARIES( ${EXE} ${${PROJECT_NAME}_LIBRARIES} )
    # Add external libraries
    SET_TARGET_PROPERTIES( ${EXE} PROPERTIES LINK_FLAGS "${LDFLAGS}" )
    SET_TARGET_PROPERTIES( ${EXE} PROPERTIES LINK_FLAGS "${LDFLAGS_EXTRA}" )
    TARGET_LINK_LIBRARIES( ${EXE} ${NEK_LIBS} ${MOAB_LIBS} ${DENDRO_LIBS} ${NETCDF_LIBS} )
    TARGET_LINK_LIBRARIES( ${EXE} ${LIBMESH_LIBS} ${TRILINOS_LIBS} ${PETSC_LIBS} ${HYPRE_LIBS} )
    TARGET_LINK_LIBRARIES( ${EXE} ${SILO_LIBS} ${HDF5_LIBS} ${TIMER_LIBS} ${X11_LIBS} )
    IF ( ${USE_EXT_SUNDIALS} )
        TARGET_LINK_LIBRARIES( ${EXE} ${SUNDIALS_LIBS} )
    ENDIF()
    TARGET_LINK_LIBRARIES( ${EXE} ${MPI_LINK_FLAGS} ${MPI_LIBRARIES} )
    TARGET_LINK_LIBRARIES( ${EXE} ${LAPACK_LIBS} ${BLAS_LIBS} ${BLAS_LAPACK_LIBS} ${ZLIB_LIBS} )
    TARGET_LINK_LIBRARIES( ${EXE} ${COVERAGE_LIBS} ${LDLIBS_EXTRA} )
    TARGET_LINK_LIBRARIES( ${EXE} ${SYSTEM_LIBS} ${SYSTEM_LDFLAGS} )
ENDMACRO()


MACRO( ADD_FILES_TO_TEST_LIB FILENAMES )
    ADD_LIBRARY( ${PACKAGE_TEST_LIB} ${FILENAMES} )
    IF ( TEST_DEP_LIST )
        TARGET_LINK_LIBRARIES( ${PACKAGE_TEST_LIB} ${TEST_DEP_LIST} )
    ENDIF()
    SET( ${PROJ}_TEST_LIB_EXISTS ${PACKAGE_TEST_LIB} )
ENDMACRO()


# Check if we want to keep the test
FUNCTION( KEEP_TEST RESULT )
    SET( ${RESULT} 1 PARENT_SCOPE )
    IF ( NOT ${PACKAGE_NAME}_ENABLE_TESTS )
        SET( ${RESULT} 0 PARENT_SCOPE )
    ENDIF()
    IF ( ONLY_BUILD_DOCS )
        SET( ${RESULT} 0 PARENT_SCOPE )
    ENDIF()
ENDFUNCTION()


# Add a provisional test
FUNCTION( ADD_PROJ_PROVISIONAL_TEST EXEFILE )
    # Check if we actually want to add the test
    KEEP_TEST( RESULT )
    IF ( NOT RESULT )
        RETURN()
    ENDIF()
    # Check if test has already been added
    SET( tmp )
    IF ( TARGET ${EXEFILE} )
        GET_TARGET_PROPERTY(tmp ${EXEFILE} LOCATION)
        STRING(REGEX REPLACE "//" "/" tmp "${tmp}" )        
    ENDIF()
    IF ( NOT tmp )
        # The target has not been added
        SET( CXXFILE ${EXEFILE} )
        SET( TESTS_SO_FAR ${TESTS_SO_FAR} ${EXEFILE} )
        # Check if we want to add the test to all
        IF ( NOT EXCLUDE_TESTS_FROM_ALL )
            ADD_EXECUTABLE( ${EXEFILE} ${CXXFILE} )
        ELSE()
            ADD_EXECUTABLE( ${EXEFILE} EXCLUDE_FROM_ALL ${CXXFILE} )
        ENDIF()
        ADD_PROJ_EXE_DEP( ${EXEFILE} )
    ELSEIF( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE}" )
        # The correct target has already been added
    ELSEIF( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE}.exe" )
        # The correct target has already been added
    ELSEIF( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/$(Configuration)/${EXEFILE}.exe" )
        # The correct target has already been added
    ELSEIF( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/$(CONFIGURATION)$(EFFECTIVE_PLATFORM_NAME)/${EXEFILE}" )
        # The correct target has already been added
    ELSEIF( ${tmp} STREQUAL "${CMAKE_CURRENT_BINARY_DIR}/$(OutDir)/${EXEFILE}.exe" )
        # The correct target has already been added
    ELSE()
        # We are trying to add 2 different tests with the same name
        MESSAGE( "Existing test: ${tmp}" )
        MESSAGE( "New test:      ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE}" )
        MESSAGE( FATAL_ERROR "Trying to add 2 different tests with the same name" )
    ENDIF()
ENDFUNCTION()
FUNCTION( ADD_${PROJ}_PROVISIONAL_TEST EXEFILE )
    ADD_PROJ_PROVISIONAL_TEST( ${EXEFILE} )
ENDFUNCTION()


# Macro to create the test name
MACRO( CREATE_TEST_NAME TEST ${ARGN} )
    IF ( PACKAGE )
        SET( TESTNAME "${PACKAGE}::${TEST}" )
    ELSE()
        SET( TESTNAME "${TEST}" )
    ENDIF()
    FOREACH( tmp ${ARGN})
        SET( TESTNAME "${TESTNAME}--${tmp}")
    endforeach()
    # STRING(REGEX REPLACE "--" "-" TESTNAME ${TESTNAME} )
ENDMACRO()


# Function to add the resource locks to an executable
FUNCTION( ADD_RESOURCE_LOCK TESTNAME EXEFILE ${ARGN} )
    IF ( NOT ARGN )
        SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES RESOURCE_LOCK ${EXEFILE} )
    ELSE()
        FOREACH( tmp ${ARGN} )
            SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES RESOURCE_LOCK ${tmp} )
        ENDFOREACH()
    ENDIF()
ENDFUNCTION()


# Add a executable as a test
FUNCTION( ADD_${PROJ}_TEST EXEFILE ${ARGN} )
    # Check if we actually want to add the test
    KEEP_TEST( RESULT )
    IF ( NOT RESULT )
        RETURN()
    ENDIF()
    # Add the provisional test
    ADD_PROJ_PROVISIONAL_TEST ( ${EXEFILE} )
    CREATE_TEST_NAME( ${EXEFILE} ${ARGN} )
    GET_TARGET_PROPERTY(EXE ${EXEFILE} LOCATION)
    STRING(REGEX REPLACE "\\$\\(Configuration\\)" "${CONFIGURATION}" EXE "${EXE}" )
    IF ( USE_EXT_MPI_FOR_SERIAL_TESTS )
        ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${EXE} ${ARGN} )
    ELSE()
        ADD_TEST( ${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
    ENDIF()
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" PROCESSORS 1 )
    ADD_RESOURCE_LOCK( ${TESTNAME} ${EXEFILE} ${ARGN} )
ENDFUNCTION()


# Add a executable as a weekly test
FUNCTION( ADD_${PROJ}_WEEKLY_TEST EXEFILE PROCS ${ARGN} )
    # Check if we actually want to add the test
    KEEP_TEST( RESULT )
    IF ( NOT RESULT )
        RETURN()
    ENDIF()
    # Add the provisional test
    ADD_PROJ_PROVISIONAL_TEST ( ${EXEFILE} )
    GET_TARGET_PROPERTY(EXE ${EXEFILE} LOCATION)
    STRING(REGEX REPLACE "\\$\\(Configuration\\)" "${CONFIGURATION}" EXE "${EXE}" )
    IF( ${PROCS} STREQUAL "1" )
        CREATE_TEST_NAME( "${EXEFILE}_WEEKLY" ${ARGN} )
    ELSEIF( USE_EXT_MPI AND NOT (${PROCS} GREATER ${TEST_MAX_PROCS}) )
        CREATE_TEST_NAME( "${EXEFILE}_${PROCS}procs_WEEKLY" ${ARGN} )
    ENDIF()
    IF ( ${PROCS} GREATER ${TEST_MAX_PROCS} )
        MESSAGE("Disabling test ${TESTNAME} (exceeds maximum number of processors ${TEST_MAX_PROCS})")
    ELSEIF( ${PROCS} STREQUAL "1" )
        CREATE_TEST_NAME( "${EXEFILE}_WEEKLY" ${ARGN} )
        IF ( USE_MPI_FOR_SERIAL_TESTS )
            ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${EXE} ${ARGN} )
        ELSE()
            ADD_TEST( ${TESTNAME} ${CMAKE_CURRENT_BINARY_DIR}/${EXEFILE} ${ARGN} )
        ENDIF()
        SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" PROCESSORS 1 )
        ADD_RESOURCE_LOCK( ${TESTNAME} ${EXEFILE} ${ARGN} )
    ELSEIF( USE_EXT_MPI AND NOT (${PROCS} GREATER ${TEST_MAX_PROCS}) )
        CREATE_TEST_NAME( "${EXEFILE}_${PROCS}procs_WEEKLY" ${ARGN} )
        ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${EXE} ${ARGN} )
        SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" PROCESSORS ${PROCS} )
        ADD_RESOURCE_LOCK( ${TESTNAME} ${EXEFILE} ${ARGN} )
    ENDIF()
ENDFUNCTION()


# Add a executable as a parallel test
FUNCTION( ADD_${PROJ}_TEST_PARALLEL EXEFILE PROCS ${ARGN} )
    # Check if we actually want to add the test
    KEEP_TEST( RESULT )
    IF ( NOT RESULT )
        RETURN()
    ENDIF()
    # Add the provisional test
    ADD_PROJ_PROVISIONAL_TEST( ${EXEFILE} )
    GET_TARGET_PROPERTY(EXE ${EXEFILE} LOCATION)
    STRING(REGEX REPLACE "\\$\\(Configuration\\)" "${CONFIGURATION}" EXE "${EXE}" )
    IF ( USE_EXT_MPI AND NOT (${PROCS} GREATER ${TEST_MAX_PROCS}) )
        CREATE_TEST_NAME( "${EXEFILE}_${PROCS}procs" ${ARGN} )
        ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${EXE} ${ARGN} )
        SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" PROCESSORS ${PROCS} )
        ADD_RESOURCE_LOCK( ${TESTNAME} ${EXEFILE} ${ARGN} )
    ENDIF()
ENDFUNCTION()


# Add a executable as an example
FUNCTION( ADD_${PROJ}_EXAMPLE EXEFILE PROCS ${ARGN} )
    # Add the file to the example doxygen file
    SET( VALUE 0 )
    FOREACH(_variableName ${EXAMPLE_LIST})
        IF ( "${_variableName}" STREQUAL "${EXEFILE}" )
            SET( VALUE 1 )
        ENDIF()
    ENDFOREACH()
    IF ( NOT ${VALUE} )
        FILE(APPEND ${EXAMPLE_INSTALL_DIR}/examples.h "* \\ref ${EXEFILE} \"${EXEFILE}\"\n" )
        SET( EXAMPLE_LIST ${EXAMPLE_LIST} ${EXEFILE} CACHE INTERNAL "example_list" FORCE )
    ENDIF()
    # Check if we actually want to add the test
    IF ( ONLY_BUILD_DOCS )
        RETURN()
    ENDIF()
    # Add the provisional test
    ADD_PROJ_PROVISIONAL_TEST( ${EXEFILE} )
    GET_TARGET_PROPERTY(EXE ${EXEFILE} LOCATION)
    STRING(REGEX REPLACE "\\$\\(Configuration\\)" "${CONFIGURATION}" EXE "${EXE}" )
    ADD_DEPENDENCIES( build-examples ${EXEFILE} )
    GET_TARGET_PROPERTY(EXE2 ${EXEFILE} LOCATION)
    ADD_CUSTOM_COMMAND( TARGET ${EXEFILE} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${EXE2} "${EXAMPLE_INSTALL_DIR}/${EXEFILE}"
    )
    IF( ${PROCS} STREQUAL "1" AND (NOT USE_EXT_MPI_FOR_SERIAL_TESTS) )
        CREATE_TEST_NAME( "example--${EXEFILE}" ${ARGN} )
        ADD_TEST( ${TESTNAME} ${EXE} ${ARGN} )
    ELSEIF ( USE_EXT_MPI AND NOT (${PROCS} GREATER ${TEST_MAX_PROCS}) )
        CREATE_TEST_NAME( "example--${EXEFILE}_${PROCS}procs" ${ARGN} )
        ADD_TEST( ${TESTNAME} ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${PROCS} ${EXE} ${ARGN} )
    ENDIF()
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES FAIL_REGULAR_EXPRESSION "${TEST_FAIL_REGULAR_EXPRESSION}" PROCESSORS ${PROCS} )
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES RESOURCE_LOCK ${EXEFILE} )
ENDFUNCTION()


# Begin configure for the examples for a package
MACRO( BEGIN_EXAMPLE_CONFIG PACKAGE )
    # Set example install dir
    SET( EXAMPLE_INSTALL_DIR ${AMP_INSTALL_DIR}/examples/${PACKAGE} )
    # Create list of examples
    SET( EXAMPLE_LIST "dummy" CACHE INTERNAL "example_list" FORCE )
    # Create doxygen input file for examples
    SET( DOXYFILE_EXTRA_SOURCES ${DOXYFILE_EXTRA_SOURCES} ${EXAMPLE_INSTALL_DIR} CACHE INTERNAL "doxyfile_extra_sources") 
    FILE(WRITE  ${EXAMPLE_INSTALL_DIR}/examples.h "// Include file for doxygen providing the examples for ${PACKAGE}\n")
    FILE(APPEND ${EXAMPLE_INSTALL_DIR}/examples.h "/*! \\page Examples_${PACKAGE}\n" )
ENDMACRO()

# Install the examples
MACRO( INSTALL_${PROJ}_EXAMPLE PACKAGE )
    FILE(APPEND ${EXAMPLE_INSTALL_DIR}/examples.h "*/\n" )
    SET( EXAMPLE_INSTALL_DIR "" )
ENDMACRO()


# Macro to check if a flag is enabled
MACRO( CHECK_ENABLE_FLAG FLAG DEFAULT )
    IF( NOT DEFINED ${FLAG} )
        SET( ${FLAG} ${DEFAULT} )
    ELSEIF( ${FLAG}  STREQUAL "" )
        SET( ${FLAG} ${DEFAULT} )
    ELSEIF( ( ${${FLAG}} STREQUAL "false" ) OR ( ${${FLAG}} STREQUAL "0" ) OR ( ${${FLAG}} STREQUAL "OFF" ) )
        SET( ${FLAG} 0 )
    ELSEIF( ( ${${FLAG}} STREQUAL "true" ) OR ( ${${FLAG}} STREQUAL "1" ) OR ( ${${FLAG}} STREQUAL "ON" ) )
        SET( ${FLAG} 1 )
    ELSE()
        MESSAGE( "Bad value for ${FLAG} (${${FLAG}}); use true or false" )
    ENDIF()
ENDMACRO()





# Macro to change the classification of a package
MACRO( SET_PACKAGE_CLASSIFICATION  PACKAGE_LIST  PACKAGE_NAME  CLASS )
    LIST(FIND ${PACKAGE_LIST} ${PACKAGE_NAME} PACKAGE_NAME_IDX)
    IF (PACKAGE_NAME_IDX EQUAL -1)
        MESSAGE(FATAL_ERROR "Package ${PACKAGE_NAME} not found in list of packages!")
    ELSE()
        MATH(EXPR PACKAGE_CLASSIFICATION_IDX "${PACKAGE_NAME_IDX}+2")
        LIST(INSERT ${PACKAGE_LIST} ${PACKAGE_CLASSIFICATION_IDX} ${CLASS})
        MATH(EXPR PACKAGE_CLASSIFICATION_IDX "${PACKAGE_CLASSIFICATION_IDX} + 1")
        LIST(REMOVE_AT ${PACKAGE_LIST} ${PACKAGE_CLASSIFICATION_IDX})
    ENDIF()
ENDMACRO()


# Macro to "disable" a package on the given platform (this mearly changes it to experimental)
MACRO( PACKAGE_DISABLE_ON_PLATFORMS  PACKAGE_LIST  PACKAGE_NAME )
    FOREACH(HOSTTYPE ${ARGN})
        IF (${PROJECT_NAME}_HOSTTYPE STREQUAL ${HOSTTYPE})
            SET_PACKAGE_CLASSIFICATION(${PACKAGE_LIST} ${PACKAGE_NAME} EX)
            IF (${PROJECT_NAME}_ENABLE_${PACKAGE_NAME})
                MESSAGE(
                  "\n***"
                  "\n*** WARNING: User has set ${PROJECT_NAME}_ENABLE_${PACKAGE_NAME}=ON but the"
                  "\n*** package ${PACKAGE_NAME} is not supported on this platform type '${HOSTTYPE}'!"
                  "\n***\n"
               )
            ENDIF()
        ENDIF()
    ENDFOREACH()
ENDMACRO()


# Append a list to a file
FUNCTION( APPEND_LIST FILENAME VARS PREFIX POSTFIX )
    FOREACH( tmp ${VARS} )
        FILE( APPEND "${FILENAME}" "${PREFIX}" )
        FILE( APPEND "${FILENAME}" "${tmp}" )
        FILE( APPEND "${FILENAME}" "${POSTFIX}" )
    ENDFOREACH ()
ENDFUNCTION()


# add custom target distclean
# cleans and removes cmake generated files etc.
MACRO( ADD_DISTCLEAN ${ARGN} )
    SET(DISTCLEANED
        cmake.depends
        cmake.check_depends
        CMakeCache.txt
        CMakeFiles
        CMakeTmp
        cmake.check_cache
        *.cmake
        compile.log
        Doxyfile
        Makefile
        core core.*
        DartConfiguration.tcl
        install_manifest.txt
        Testing
        include
        doc
        docs
        latex_docs
        lib
        Makefile.config
        install_manifest.txt
        test
        matlab
        mex
        tmp
        #tmp#
        bin
        cmake
        ${ARGN}
    )
    ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
    IF (UNIX)
        ADD_CUSTOM_COMMAND(
            DEPENDS clean
            COMMENT "distribution clean"
            COMMAND rm
            ARGS    -Rf ${DISTCLEANED}
            TARGET  distclean
        )
    ELSE()
        SET( DISTCLEANED
            ${DISTCLEANED}
            *.vcxproj*
            ipch
            x64
            Debug
        )
        SET( DISTCLEAN_FILE "${CMAKE_CURRENT_BINARY_DIR}/distclean.bat" )
        FILE( WRITE  "${DISTCLEAN_FILE}" "del /s /q /f " )
        APPEND_LIST( "${DISTCLEAN_FILE}" "${DISTCLEANED}" " " " " )
        FILE( APPEND "${DISTCLEAN_FILE}" "\n" )
        APPEND_LIST( "${DISTCLEAN_FILE}" "${DISTCLEANED}" "for /d %%x in ("   ") do rd /s /q \"%%x\"\n" )
        ADD_CUSTOM_COMMAND(
            DEPENDS clean
            COMMENT "distribution clean"
            COMMAND distclean.bat & del /s/q/f distclean.bat
            TARGET  distclean
        )
    ENDIF()
ENDMACRO()




# Add an external subdirectory
MACRO( ADD_EXTERNAL_PACKAGE_SUBDIRECTORY SUBDIR_NAME SUBDIR_PATH )
    VERIFY_PATH ( ${SUBDIR_PATH} )
    FIND_FILES_PATH ( ${SUBDIR_PATH} )
    FILE( GLOB HFILES RELATIVE ${SUBDIR_PATH} ${SUBDIR_PATH}/*.h ${SUBDIR_PATH}/*.hh ${SUBDIR_PATH}/*.I )
    FOREACH (HFILE ${HFILES})
        CONFIGURE_FILE( ${SUBDIR_PATH}/${HFILE} ${${PROJ}_INSTALL_DIR}/include/${CURPACKAGE}/${HFILE} COPYONLY )
    ENDFOREACH ()
    ADD_SUBDIRECTORY ( ${SUBDIR_PATH} ${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR_NAME} )
ENDMACRO()


# Print the current repo version and create target to write to a file
SET( WriteRepoVersionCmakeFile "${CMAKE_CURRENT_LIST_DIR}/WriteRepoVersion.cmake" )
FUNCTION( WRITE_REPO_VERSION FILENAME )
    SET( CMD ${CMAKE_COMMAND} -Dfilename="${FILENAME}" -Dsrc_dir="${${PROJ}_SOURCE_DIR}" 
             -Dtmp_file="${CMAKE_CURRENT_BINARY_DIR}/tmp/version.h" -DPROJ=${PROJ} 
             -P "${WriteRepoVersionCmakeFile}" )
    EXECUTE_PROCESS( COMMAND ${CMD} )
    ADD_CUSTOM_TARGET( write_repo_version  COMMENT "Write repo version"  COMMAND ${CMD} )
ENDFUNCTION()


