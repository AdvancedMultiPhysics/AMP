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
    IF ( DEFINED USE_AMP_UTILS )
        MESSAGE( WARNING "Setting USE_AMP_UTILS/USE_AMP_MESH/... is deprecated" )
    ENDIF()
    # Add documentation folders and define variables
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/discretization\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/IO\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/materials\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/matrices\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/mesh\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/operators\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/solvers\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/time_integrators\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/utils\"" )
    SET( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/vectors\"" )
ENDMACRO()



