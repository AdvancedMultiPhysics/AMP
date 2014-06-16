INCLUDE(TribitsTplDeclareLibraries)
INCLUDE( ${AMP_SOURCE_DIR}/cmake/macros.cmake )
INCLUDE( ${AMP_SOURCE_DIR}/cmake/FindLibmesh.cmake )

# Get LIBMESH_DIRECTORY
IF ( LIBMESH_DIR )
    SET( LIBMESH_DIRECTORY ${LIBMESH_DIR} )
ELSEIF( TPL_LIBMESH_DIR )
    SET( LIBMESH_DIRECTORY ${TPL_LIBMESH_DIR} )
ELSE()
    MESSAGE(FATAL_ERROR "Could not find Libmesh.  Please manually set LIBMESH_DIR or TPL_LIBMESH_DIR to point to the Libmesh installation directory" )
ENDIF()
VERIFY_PATH ( ${LIBMESH_DIRECTORY} )

# Get LIBMESH_HOSTTYPE and LIBMESH_COMPILE_TYPE
IF ( LIBMESH_HOSTTYPE )
    SET( LIBMESH_HOSTTYPE ${LIBMESH_HOSTTYPE} )
ELSEIF ( TPL_LIBMESH_HOSTTYPE )
    SET( LIBMESH_HOSTTYPE ${TPL_LIBMESH_HOSTTYPE} )
ELSEIF ( NOT LIBMESH_ARCH )
    MESSAGE(FATAL_ERROR "LIBMESH_HOSTTYPE is not set.  Please manually set LIBMESH_HOSTTYPE or TPL_LIBMESH_HOSTTYPE" )
ENDIF()
IF ( LIBMESH_COMPILE_TYPE )
    SET( LIBMESH_COMPILE_TYPE ${LIBMESH_COMPILE_TYPE} )
ELSEIF ( TPL_LIBMESH_COMPILE_TYPE )
    SET( LIBMESH_COMPILE_TYPE ${TPL_LIBMESH_COMPILE_TYPE} )
ELSEIF ( NOT LIBMESH_ARCH )
    MESSAGE(FATAL_ERROR "LIBMESH_COMPILE_TYPE is not set.  Please manually set LIBMESH_COMPILE_TYPE or TPL_LIBMESH_COMPILE_TYPE" )
ENDIF()

# Find the libmesh includes
LIBMESH_SET_INCLUDES( ${LIBMESH_DIRECTORY} )

# Find the libmesh libraries
LIBMESH_SET_LIBRARIES( ${LIBMESH_DIRECTORY} )
MESSAGE ( "Using libmesh" )
MESSAGE ( "   "  ${LIBMESH_LIBS} )

# Add the tribits flags
SET( TPL_ENABLE_LIBMESH ON )
SET( TPL_LIBMESH_INCLUDE_DIRS ${LIBMESH_INCLUDE} )
SET( TPL_LIBMESH_LIBRARY_DIRS "" )
SET( TPL_LIBMESH_LIBRARIES ${LIBMESH_LIBS} )
SET( ${LIBMESH_LIBS} )

# Add the definitions
SET( USE_EXT_LIBMESH 1 )
