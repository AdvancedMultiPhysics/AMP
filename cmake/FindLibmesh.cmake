FUNCTION ( LIBMESH_SET_INCLUDES  LIBMESH_DIRECTORY )
    VERIFY_PATH ( ${LIBMESH_DIRECTORY} )
    # Include the libmesh directories
    SET ( LIBMESH_INCLUDE )
    SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/ )
    SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} PARENT_SCOPE )
ENDFUNCTION()


FUNCTION ( LIBMESH_SET_LIBRARIES  LIBMESH_DIRECTORY )
    VERIFY_PATH ( ${LIBMESH_DIRECTORY} )
    # Find the libmesh libaries
    SET ( LIBMESH_PATH_LIB ${LIBMESH_DIRECTORY}/lib )
    SET ( LIBMESH_CONTRIB_PATH_LIB ${LIBMESH_DIRECTORY}/lib )
    VERIFY_PATH ( ${LIBMESH_PATH_LIB} )
    VERIFY_PATH ( ${LIBMESH_CONTRIB_PATH_LIB} )
    FIND_LIBRARY ( LIBMESH_MESH_LIB     NAMES mesh_dbg      PATHS ${LIBMESH_PATH_LIB}          NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_EXODUSII_LIB NAMES exodusii  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_LASPACK_LIB  NAMES laspack   PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_METIS_LIB    NAMES metis     PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_NEMESIS_LIB  NAMES nemesis   PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
    FIND_LIBRARY ( LIBMESH_NETCDF_LIB   NAMES netcdf    PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
    IF ( USE_EXT_MPI ) 
#        FIND_LIBRARY ( LIBMESH_PARMETIS_LIB NAMES parmetis  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
    ENDIF()
#    FIND_LIBRARY ( LIBMESH_SFCURVES_LIB NAMES sfcurves  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_GMV_LIB      NAMES gmv       PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_GZSTREAM_LIB NAMES gzstream  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_TETGEN_LIB   NAMES tetgen    PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    FIND_LIBRARY ( LIBMESH_TRIANGLE_LIB NAMES triangle  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
#    IF ( ${CMAKE_SYSTEM} MATCHES ^Linux.* )
#        IF ( ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64 )
#            SET ( LIBMESH_TECIO_LIB  ${LIBMESH_DIRECTORY}/contrib/tecplot/lib/x86_64-unknown-linux-gnu/tecio.a )
#        ENDIF ()
#    ENDIF ()
    IF ( NOT LIBMESH_MESH_LIB )
        MESSAGE ( FATAL_ERROR "Libmesh library (mesh) not found in ${LIBMESH_PATH_LIB}" )
    ENDIF ()
#    IF ( (NOT LIBMESH_EXODUSII_LIB) OR (NOT LIBMESH_GMV_LIB) OR (NOT LIBMESH_GZSTREAM_LIB) OR
#         (NOT LIBMESH_LASPACK_LIB) OR 
#         (NOT LIBMESH_NEMESIS_LIB) OR (NOT LIBMESH_NETCDF_LIB) OR (NOT LIBMESH_METIS_LIB) OR 
#         (NOT LIBMESH_SFCURVES_LIB) OR (NOT LIBMESH_TETGEN_LIB) OR (NOT LIBMESH_TRIANGLE_LIB) )
#        MESSAGE ( ${LIBMESH_EXODUSII_LIB} )
#        MESSAGE ( ${LIBMESH_LASPACK_LIB} )
#        MESSAGE ( ${LIBMESH_NEMESIS_LIB} )
#        MESSAGE ( ${LIBMESH_NETCDF_LIB} )
#        MESSAGE ( ${LIBMESH_METIS_LIB} )
#        MESSAGE ( ${LIBMESH_SFCURVES_LIB} )
#        MESSAGE ( ${LIBMESH_GMV_LIB} )
#        MESSAGE ( ${LIBMESH_GZSTREAM_LIB} )
#        MESSAGE ( ${LIBMESH_TETGEN_LIB} )
#        MESSAGE ( ${LIBMESH_TRIANGLE_LIB} )
#        MESSAGE ( FATAL_ERROR "Libmesh contribution libraries not found in ${LIBMESH_PATH_LIB}" )
#    ENDIF ()
#    IF ( USE_EXT_MPI AND (NOT LIBMESH_PARMETIS_LIB) )
#        MESSAGE ( ${LIBMESH_PARMETIS_LIB} )
#        MESSAGE ( FATAL_ERROR "Libmesh contribution libraries not found in ${LIBMESH_PATH_LIB}" )
#    ENDIF ()
    # Add the libraries in the appropriate order
    SET ( LIBMESH_LIBS
        ${LIBMESH_MESH_LIB}
        ${LIBMESH_EXODUSII_LIB}
        ${LIBMESH_LASPACK_LIB}
        ${LIBMESH_NEMESIS_LIB}
        ${LIBMESH_NETCDF_LIB}
    )
    IF ( USE_EXT_MPI OR TPL_ENABLE_MPI ) 
        SET ( LIBMESH_LIBS ${LIBMESH_LIBS} ${LIBMESH_PARMETIS_LIB} )
    ENDIF()
    SET ( LIBMESH_LIBS
        ${LIBMESH_LIBS}
        ${LIBMESH_METIS_LIB}
        ${LIBMESH_SFCURVES_LIB}
        ${LIBMESH_GMV_LIB}
        ${LIBMESH_GZSTREAM_LIB}
        ${LIBMESH_TETGEN_LIB}
        ${LIBMESH_TRIANGLE_LIB} 
    )
    IF ( LIBMESH_TECIO_LIB )
        SET ( LIBMESH_LIBS ${LIBMESH_LIBS} ${LIBMESH_TECIO_LIB} )
    ENDIF ()
    SET ( LIBMESH_LIBS ${LIBMESH_LIBS} PARENT_SCOPE )
ENDFUNCTION()


