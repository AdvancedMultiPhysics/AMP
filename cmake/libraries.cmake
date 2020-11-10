# Macro to find and configure DENDRO
MACRO ( CONFIGURE_DENDRO )
    # Determine if we want to use 
    CHECK_ENABLE_FLAG( USE_EXT_DENDRO 0 )
    IF ( USE_EXT_DENDRO )
        IF ( DENDRO_DIRECTORY )
            VERIFY_PATH ( ${DENDRO_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${DENDRO_DIRECTORY}/include )
            SET( DENDRO_INCLUDE ${DENDRO_DIRECTORY}/include )
            FIND_LIBRARY ( DENDRO_BIN_LIB   NAMES BinOps PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_OCT_LIB   NAMES Oct    PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_PAR_LIB   NAMES Par    PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_POINT_LIB NAMES Point  PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_TEST_LIB  NAMES Test   PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            IF ( (NOT DENDRO_BIN_LIB) OR (NOT DENDRO_OCT_LIB) OR (NOT DENDRO_PAR_LIB) OR
                (NOT DENDRO_POINT_LIB) OR (NOT DENDRO_TEST_LIB) )
                MESSAGE( ${DENDRO_BIN_LIB} )
                MESSAGE( ${DENDRO_OCT_LIB} )
                MESSAGE( ${DENDRO_PAR_LIB} )
                MESSAGE( ${DENDRO_POINT_LIB} )
                MESSAGE( ${DENDRO_TEST_LIB} )
                MESSAGE( FATAL_ERROR "DENDRO libraries not found in ${DENDRO_DIRECTORY}/lib" )
            ENDIF ()
            # Add the libraries in the appropriate order
            SET( DENDRO_LIBS
                ${DENDRO_OCT_LIB}
                ${DENDRO_PAR_LIB}
                ${DENDRO_POINT_LIB}
                ${DENDRO_TEST_LIB}
                ${DENDRO_BIN_LIB}
             )
        ELSE()
            MESSAGE( FATAL_ERROR "Default search for DENDRO is not supported.  Use -D DENDRO_DIRECTORY=" )
        ENDIF()
        ADD_DEFINITIONS( -DUSE_EXT_DENDRO )  
        MESSAGE( "Using DENDRO" )
        MESSAGE( "   ${DENDRO_LIBS}" )
    ENDIF()
ENDMACRO()


# Macro to find and configure MOAB
MACRO ( CONFIGURE_MOAB )
    # Determine if we want to use MOAB
    CHECK_ENABLE_FLAG( USE_EXT_MOAB 0 )
    IF ( USE_EXT_MOAB )
        # Check if we specified the MOAB directory
        IF ( MOAB_DIRECTORY )
            VERIFY_PATH ( ${MOAB_DIRECTORY} )
            # Include the MOAB directories
            SET( MOAB_INCLUDE ${MOAB_DIRECTORY}/include )
            SET( IMESH_INCLUDE ${MOAB_DIRECTORY}/lib )
            # Find the MOAB libaries
            SET( MOAB_PATH_LIB ${MOAB_DIRECTORY}/lib )
            VERIFY_PATH ( ${MOAB_PATH_LIB} )
            FIND_LIBRARY ( MOAB_MESH_LIB     NAMES MOAB      PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_iMESH_LIB    NAMES iMesh     PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_COUPLER_LIB  NAMES mbcoupler PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            IF ( NOT MOAB_MESH_LIB )
                MESSAGE( FATAL_ERROR "MOAB library (MOAB) not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_iMESH_LIB )
                MESSAGE( FATAL_ERROR "iMesh library ${MOAB_iMESH_LIB}  not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_COUPLER_LIB )
                MESSAGE( FATAL_ERROR "MBCoupler library ${MOAB_COUPLER_LIB}  not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE( FATAL_ERROR "Default search for MOAB is not supported.  Use -D MOAB_DIRECTORY=" )
        ENDIF()
        # Check if we specified the cgm directory
        IF ( CGM_DIRECTORY )
            VERIFY_PATH ( ${CGM_DIRECTORY} )
            # Include the CGM directories
            SET( MOAB_INCLUDE ${MOAB_INCLUDE} ${CGM_DIRECTORY}/include )
            # Find the CGM libaries
            SET( CGM_PATH_LIB ${CGM_DIRECTORY}/lib )
            VERIFY_PATH ( ${CGM_PATH_LIB} )
            FIND_LIBRARY ( MOAB_CGM_LIB     NAMES cgm      PATHS ${CGM_PATH_LIB}        NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_iGEOM_LIB   NAMES iGeom    PATHS ${CGM_PATH_LIB}        NO_DEFAULT_PATH )
            IF ( NOT MOAB_CGM_LIB )
                MESSAGE( FATAL_ERROR "CGM library ${MOAB_CGM_LIB}  not found in ${CGM_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_iGEOM_LIB )
                MESSAGE( FATAL_ERROR "iGEOM library ${MOAB_iGEOM_LIB}  not found in ${CGM_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE( FATAL_ERROR "Default search for cgm is not supported.  Use -D CGM_DIRECTORY=" )
        ENDIF()
        # Check if we specified the Cubit directory
        IF ( CUBIT_DIRECTORY )
            VERIFY_PATH ( ${CUBIT_DIRECTORY} )
            # Include the CUBIT directories
            # SET( MOAB_INCLUDE ${MOAB_INCLUDE} ${CUBIT_DIRECTORY}/include )
            # Find the CGM libaries
            SET( CUBIT_PATH_LIB ${CUBIT_DIRECTORY} )
            VERIFY_PATH ( ${CGM_PATH_LIB} )
            FIND_LIBRARY ( MOAB_CUBIT_LIB     NAMES cubiti19      PATHS ${CUBIT_PATH_LIB}        NO_DEFAULT_PATH )
            IF ( NOT MOAB_CUBIT_LIB )
                MESSAGE( FATAL_ERROR "CUBIT librarys not found in ${CUBIT_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE( FATAL_ERROR "Default search for cubit is not supported.  Use -D CUBIT_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        INCLUDE_DIRECTORIES ( ${MOAB_INCLUDE} )
        SET( MOAB_LIBS
            ${MOAB_COUPLER_LIB}
            ${MOAB_iMESH_LIB}
            ${MOAB_MESH_LIB}
            ${MOAB_CGM_LIB}
            ${MOAB_iGEOM_LIB}
            ${MOAB_CUBIT_LIB}
        )
        ADD_DEFINITIONS( -DUSE_EXT_MOAB )  
        MESSAGE( "Using MOAB" )
        MESSAGE( "   ${MOAB_LIBS}" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the Eigen package
MACRO ( CONFIGURE_EIGEN_LIBRARIES )
    # Determine if we want to use Eigen
    CHECK_ENABLE_FLAG( USE_EXT_EIGEN 0 )
    IF ( USE_EXT_EIGEN )
        IF ( EIGEN3_INCLUDE_DIR )
            INCLUDE_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
            MESSAGE( "Using Eigen" )
        ELSE()
            MESSAGE( FATAL_ERROR "Default search for eigen is not yet supported.  Use -D EIGEN3_INCLUDE_DIR" )
        ENDIF()
        ADD_DEFINITIONS( -DUSE_EXT_EIGEN )  
    ENDIF()
ENDMACRO ()


# Macro to find and configure the Eigen package
MACRO ( CONFIGURE_ARMADILLO_LIBRARIES )
    # Determine if we want to use Eigen
    CHECK_ENABLE_FLAG( USE_EXT_ARMADILLO 0 )
    IF ( USE_EXT_ARMADILLO )
        FIND_PACKAGE(Armadillo REQUIRED)
        ADD_DEFINITIONS( -DUSE_EXT_ARMADILLO )  
    ENDIF()
ENDMACRO ()


# Macro to configure system-specific libraries and flags
MACRO ( CONFIGURE_SYSTEM )
    IDENTIFY_COMPILER()
    # Remove extra library links
    CHECK_ENABLE_FLAG( USE_STATIC 0 )
    IF ( USE_STATIC )
        SET_STATIC_FLAGS()
    ENDIF()
    # Add system dependent flags
    MESSAGE("System is: ${CMAKE_SYSTEM_NAME}")
    IF ( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
        # Windows specific system libraries
        SET( SYSTEM_PATHS "C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64" 
                          "C:/Program Files (x86)/Microsoft Visual Studio 8/VC/PlatformSDK/Lib/AMD64" 
                          "C:/Program Files (x86)/Microsoft Visual Studio 12.0/Common7/Packages/Debugger/X64" )
        FIND_LIBRARY( PSAPI_LIB    NAMES Psapi    PATHS ${SYSTEM_PATHS}  NO_DEFAULT_PATH )
        FIND_LIBRARY( DBGHELP_LIB  NAMES DbgHelp  PATHS ${SYSTEM_PATHS}  NO_DEFAULT_PATH )
        FIND_LIBRARY( DBGHELP_LIB  NAMES DbgHelp )
        IF ( PSAPI_LIB ) 
            ADD_DEFINITIONS( -DPSAPI )
            SET( SYSTEM_LIBS ${PSAPI_LIB} )
        ENDIF()
        IF ( DBGHELP_LIB ) 
            ADD_DEFINITIONS( -DDBGHELP )
            SET( SYSTEM_LIBS ${DBGHELP_LIB} )
        ELSE()
            MESSAGE( WARNING "Did not find DbgHelp, stack trace will not be availible" )
        ENDIF()
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
        # Linux specific system libraries
        SET( SYSTEM_LIBS "-ldl -lpthread" )
        CONFIGURE_ZLIB()
        IF ( NOT USE_STATIC )
            SET( SYSTEM_LIBS "${SYSTEM_LIBS} -rdynamic" )   # Needed for backtrace to print function names
        ENDIF()
        IF ( USING_GCC )
            SET( SYSTEM_LIBS "${SYSTEM_LIBS} -lgfortran" )   # Needed for backtrace to print function names
        ENDIF()
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" )
        # Max specific system libraries
        SET( SYSTEM_LIBS "-ldl -lpthread" )
        CONFIGURE_ZLIB()
        IF ( USING_GCC )
            SET( SYSTEM_LIBS "${SYSTEM_LIBS} -lgfortran" )
        ENDIF()
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Generic" )
        # Generic system libraries
    ELSE()
        MESSAGE( FATAL_ERROR "OS not detected" )
    ENDIF()
    MESSAGE("System libs: ${SYSTEM_LIBS}")
ENDMACRO ()


# Macro to configure AMP-specific options
MACRO ( CONFIGURE_AMP )
    # Add the AMP install directory
    INCLUDE_DIRECTORIES ( ${AMP_INSTALL_DIR}/include )
    # Set the data directory for AMP (needed to find the meshes)
    IF ( AMP_DATA OR AMP_DATA_URL )
        IF ( AMP_DATA_URL )
            MESSAGE( STATUS "Downloading AMP Data - ${AMP_DATA_URL}" )
            GET_FILENAME_COMPONENT( AMP_DATA "${AMP_DATA_URL}" NAME)
            SET( AMP_DATA "${CMAKE_CURRENT_BINARY_DIR}/${AMP_DATA}" )
            FILE( DOWNLOAD "${AMP_DATA_URL}" "${AMP_DATA}" )
        ENDIF()
        IF ( "${AMP_DATA}" STREQUAL "" )
            UNSET( AMP_DATA )
        ENDIF()
        IF ( IS_DIRECTORY "${AMP_DATA}" )
            # AMP_DATA is a directory
        ELSEIF ( EXISTS "${AMP_DATA}" )
            # AMP_DATA is a file, try to unpack it
            EXECUTE_PROCESS(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${AMP_DATA}"
                WORKING_DIRECTORY "${AMP_INSTALL_DIR}"
            )
            IF ( EXISTS "${AMP_INSTALL_DIR}/AMP-Data" )
                SET( AMP_DATA "${AMP_INSTALL_DIR}/AMP-Data" )
            ELSE()
                MESSAGE(FATAL_ERROR "Error unpacking tar file ${AMP_DATA}")
            ENDIF()
        ENDIF()
    ENDIF()
    IF ( AMP_DATA )
        ADD_DEFINITIONS( -DUSE_AMP_DATA )
    ELSEIF ( NOT ONLY_BUILD_DOCS AND NOT AMP_DATA )
        MESSAGE( WARNING "AMP_DATA is not set, some tests will be disabled" )
    ENDIF()
    # Fix LDFLAGS if it is a CMake list
    STRING(REPLACE ";" " " LDFLAGS "${LDFLAGS}")
    # Check the user configure flags
    CHECK_ENABLE_FLAG( USE_AMP_UTILS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_MESH 1 )
    CHECK_ENABLE_FLAG( USE_AMP_DISCRETIZATION 1 )
    CHECK_ENABLE_FLAG( USE_AMP_VECTORS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_MATRICES 1 )
    CHECK_ENABLE_FLAG( USE_AMP_MATERIALS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_OPERATORS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_SOLVERS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_TIME_INTEGRATORS 1 )
    CHECK_ENABLE_FLAG( USE_AMP_GRAPHICS 1 )
    # Check and disable packages based on dependencies
    IF ( NOT BUILD_ONLY_DOCS )
        # Check if we are using utils
        IF ( NOT USE_AMP_UTILS )
            MESSAGE( FATAL_ERROR "AMP Utils must be used" )
        ENDIF()
        # Check if we are using ampmesh
        IF ( NOT USE_AMP_MESH )
            MESSAGE( "Disabling AMP Mesh" )
        ENDIF()
        # Check if we are using discretization
        IF ( NOT USE_AMP_MESH )
            SET( USE_AMP_DISCRETIZATION 0 )
        ENDIF()
        IF ( NOT USE_AMP_DISCRETIZATION )
            MESSAGE( "Disabling AMP Descritization" )
        ENDIF()
        # Check if we are using vectors
        IF ( NOT USE_AMP_VECTORS )
            MESSAGE( "Disabling AMP Vectors" )
        ENDIF()
        # Check if we are using matrices
        IF ( NOT USE_AMP_MATRICES )
            MESSAGE( "Disabling AMP Matrices" )
        ENDIF()
        # Check if we are using materials
        IF ( NOT USE_AMP_VECTORS )
            SET( USE_AMP_MATERIALS 0 )
        ENDIF()
        IF ( NOT USE_AMP_MATRICES )
            MESSAGE( "Disabling AMP Materials" )
        ENDIF()
        # Check if we are using operators
        IF ( (NOT USE_AMP_MESH) OR (NOT USE_AMP_VECTORS) OR (NOT USE_AMP_MATRICES) )
            SET( USE_AMP_OPERATORS 0 )
        ENDIF()
        IF ( NOT USE_AMP_OPERATORS )
            MESSAGE( "Disabling AMP Operators" )
        ENDIF()
        # Check if we are using solvers
        IF ( (NOT USE_AMP_OPERATORS) )
            SET( USE_AMP_SOLVERS 0 )
        ENDIF()
        IF ( NOT USE_AMP_SOLVERS )
            MESSAGE( "Disabling AMP Solvers" )
        ENDIF()
        # Check if we are using time_integrators
        IF ( (NOT USE_AMP_SOLVERS) )
            SET( USE_AMP_TIME_INTEGRATORS 0 )
        ENDIF()
        IF ( NOT USE_AMP_TIME_INTEGRATORS )
            MESSAGE( "Disabling AMP Time Integrators" )
        ENDIF()
    ENDIF()
    # Add documentation folders and define variables
    IF ( USE_AMP_TIME_INTEGRATORS )
        ADD_DEFINITIONS( -DUSE_AMP_TIME_INTEGRATORS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/time_integrators\"" )
    ENDIF()
    IF ( USE_AMP_SOLVERS )
        ADD_DEFINITIONS( -DUSE_AMP_SOLVERS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/solvers\"" )
    ENDIF()
    IF ( USE_AMP_OPERATORS )
        ADD_DEFINITIONS( -DUSE_AMP_OPERATORS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/operators\"" )
    ENDIF()
    IF ( USE_AMP_MATERIALS )
        ADD_DEFINITIONS( -DUSE_AMP_MATERIALS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/materials\"" )
    ENDIF()
    IF ( USE_AMP_MATRICES )
        ADD_DEFINITIONS( -DUSE_AMP_MATRICES )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/matrices\"" )
    ENDIF()
    IF ( USE_AMP_VECTORS )
        ADD_DEFINITIONS( -DUSE_AMP_VECTORS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/vectors\"" )
    ENDIF()
    IF ( USE_AMP_DISCRETIZATION )
        ADD_DEFINITIONS( -DUSE_AMP_DISCRETIZATION )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/discretization\"" )
    ENDIF()
    IF ( USE_AMP_MESH )
        ADD_DEFINITIONS( -DUSE_AMP_MESH )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/ampmesh\"" )
    ENDIF()
    IF ( USE_AMP_UTILS )
        ADD_DEFINITIONS( -DUSE_AMP_UTILS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/utils\"" )
    ENDIF()
    IF ( USE_AMP_GRAPHICS )
        ADD_DEFINITIONS( -DUSE_AMP_GRAPHICS )
        SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/graphics\"" )
    ENDIF()
ENDMACRO()



