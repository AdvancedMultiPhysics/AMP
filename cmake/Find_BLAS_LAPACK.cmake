# Check that the PROJ and ${PROJ}_INSTALL/SOURCE_DIR variables are set 
IF ( NOT PROJ )
    MESSAGE(FATAL_ERROR "PROJ must be set before including macros.cmake")
ENDIF()
IF ( NOT ${PROJ}_INSTALL_DIR )
    MESSAGE(FATAL_ERROR "${PROJ}_INSTALL_DIR must be set before including macros.cmake")
ENDIF()
IF ( NOT ${PROJ}_SOURCE_DIR )
    MESSAGE(FATAL_ERROR "${PROJ}_INSTALL_DIR must be set before including macros.cmake")
ENDIF()
SET( BlasLapackCMakeDir "${${PROJ}_SOURCE_DIR}/cmake/BlasLapack" )
SET( BlasLapackInstallDir "${${PROJ}_INSTALL_DIR}/include" )
INCLUDE( "${BlasLapackCMakeDir}/FindBLAS.cmake" )
INCLUDE( "${BlasLapackCMakeDir}/FindLAPACK.cmake" )


# Macro to configure BLAS and LAPACK libraries
FUNCTION( CONFIGURE_BLAS_AND_LAPACK )
    CHECK_ENABLE_FLAG( USE_ACML 0 )
    CHECK_ENABLE_FLAG( USE_MKL  0 )
    CHECK_ENABLE_FLAG( USE_MATLAB 0 )
    CHECK_ENABLE_FLAG( USE_MATLAB_LAPACK 0 )
    IF ( NOT APPLE )
        SET( USE_VECLIB 0 )
    ELSE()
        CHECK_ENABLE_FLAG( USE_VECLIB  1 )
    ENDIF()
    CHECK_ENABLE_FLAG( USE_BLAS   1 )
    CHECK_ENABLE_FLAG( USE_LAPACK 1 )
    # Write a file with the necessary includes for blas/lapack routines
    SET( BLAS_LAPACK_HEADER "${CMAKE_CURRENT_BINARY_DIR}/tmp/blas_lapack.h" )
    FILE(WRITE ${BLAS_LAPACK_HEADER} "// This is a automatically generated file to include blas/lapack headers\n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#ifndef INCLUDE_BLAS_LAPACK\n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#define INCLUDE_BLAS_LAPACK\n" )
    IF( USE_MATLAB AND USE_MATLAB_LAPACK ) 
        # Matlab requires using their lapack/blas
        SET( USE_BLAS 1 )
        SET( USE_LAPACK 1 )
        SET( USE_MATLAB 1 PARENT_SCOPE )
        SET( USE_MATLAB_LAPACK ${USE_MATLAB_LAPACK} PARENT_SCOPE )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_BLAS\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_LAPACK\n" )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_MATLAB_LAPACK\n" )
        FIND_LIBRARY( BLAS_LIBS   NAMES mwblas          PATHS "${MATLAB_EXTERN}"  NO_DEFAULT_PATH )
        FIND_LIBRARY( BLAS_LIBS   NAMES libmwblas.dll   PATHS "${MATLAB_EXTERN}"  NO_DEFAULT_PATH )
        FIND_LIBRARY( LAPACK_LIBS NAMES mwlapack        PATHS "${MATLAB_EXTERN}"  NO_DEFAULT_PATH )
        FIND_LIBRARY( LAPACK_LIBS NAMES libmwlapack.dll PATHS "${MATLAB_EXTERN}"  NO_DEFAULT_PATH )
        IF ( (NOT BLAS_LIBS) OR (NOT LAPACK_LIBS) )
            MESSAGE("${BLAS_LIBS}")
            MESSAGE("${LAPACK_LIBS}")
            MESSAGE(FATAL_ERROR "Could not find MATLAB blas/lapack libraries in '${MATLAB_EXTERN}'")
        ENDIF()
        SET( BLAS_LAPACK_LIBS ${BLAS_LIBS} ${LAPACK_LIBS} PARENT_SCOPE)
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#include \"${MATLAB_DIRECTORY}/extern/include/tmwtypes.h\"\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#include \"${MATLAB_DIRECTORY}/extern/include/blas.h\"\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#include \"${MATLAB_DIRECTORY}/extern/include/lapack.h\"\n" )
    ELSEIF( USE_ACML ) 
        CONFIGURE_ACML()
        SET( USE_BLAS 1 )
        SET( USE_LAPACK 1 )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_BLAS\n" )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_LAPACK\n" )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_ACML\n" )
        SET( BLAS_LAPACK_LIBS ${ACML_LIBS} PARENT_SCOPE)
    ELSEIF( USE_MKL ) 
        CONFIGURE_MKL()
        SET( USE_BLAS 1 )
        SET( USE_LAPACK 1 )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_BLAS\n" )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_LAPACK\n" )
        FILE(APPEND ${BLAS_LAPACK_HEADER} "#define USE_MKL\n" )
        SET( BLAS_LAPACK_LIBS ${MKL_LIBS} ${MKL_LIBS} PARENT_SCOPE)
    ELSEIF( USE_ATLAS ) 
        CONFIGURE_ATLAS()
        SET( USE_BLAS 1 )
        SET( USE_LAPACK 1 )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_BLAS\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_LAPACK\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_ATLAS\n" )
        SET( BLAS_LAPACK_LIBS ${ATLAS_LIBS} PARENT_SCOPE)
    ELSEIF( USE_VECLIB ) 
        SET( USE_BLAS 1 )
        SET( USE_LAPACK 1 )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_BLAS\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_LAPACK\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#define USE_VECLIB\n" )
        FILE(APPEND "${BLAS_LAPACK_HEADER}" "#include <Accelerate/Accelerate.h>\n" )
        SET( BLAS_LAPACK_LIBS "-framework Accelerate" PARENT_SCOPE)
        MESSAGE( "Using Accelerate" )
    ELSE()
        IF ( DEFINED USE_EXT_BLAS )
            SET( USE_BLAS ${USE_EXT_BLAS} )
        ENDIF()
        IF ( DEFINED USE_EXT_LAPACK )
            SET( USE_LAPACK ${USE_EXT_LAPACK} )
        ENDIF()
        IF ( USE_BLAS OR USE_LAPACK )
            CONFIGURE_BLAS()
            CONFIGURE_LAPACK()
            IF ( (NOT BLAS_LIBS) OR (NOT LAPACK_LIBS) )
                MESSAGE(FATAL_ERROR "Blas or Lapack libraries not found")
            ENDIF()
            SET( BLAS_LAPACK_LIBS ${BLAS_LIBS} ${LAPACK_LIBS} PARENT_SCOPE)
            CONFIGURE_FILE( "${BlasLapackCMakeDir}/fortran_calls.h" "${BlasLapackInstallDir}/fortran_calls.h" COPYONLY )
            FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${BlasLapackInstallDir}/fortran_calls.h\"\n" )
        ENDIF()
    ENDIF()
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#endif\n" )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${BLAS_LAPACK_HEADER}" "${${PROJ}_INSTALL_DIR}/include/blas_lapack.h" )
    SET( USE_BLAS ${USE_BLAS} PARENT_SCOPE)
    SET( USE_LAPACK ${USE_LAPACK} PARENT_SCOPE)
ENDFUNCTION()


# Macro to configure ACML
MACRO( CONFIGURE_ACML )
    IF ( NOT ACML_DIRECTORY )
        MESSAGE(FATAL_ERROR "Default search for ACML not supported, set ACML_DIRECTORY" )
    ENDIF()
    VERIFY_PATH( ${ACML_DIRECTORY} )
    VERIFY_PATH( ${ACML_DIRECTORY}/include )
    VERIFY_PATH( ${ACML_DIRECTORY}/lib )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${ACML_DIRECTORY}/include/acml.h\"\n" )
    FIND_LIBRARY( ACML_LIBS NAMES libacml.a PATHS ${ACML_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ACML_LIBS NAMES libacml_dll.lib PATHS ${ACML_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ACML_LIBS NAMES acml      PATHS ${ACML_DIRECTORY}/lib  NO_DEFAULT_PATH )
    MESSAGE( "Using acml" )
    MESSAGE( "   ${ACML_LIBS}" )
ENDMACRO()


# Macro to configure MKL
MACRO( CONFIGURE_MKL )
    IF ( NOT MKL_DIRECTORY )
        MESSAGE(FATAL_ERROR "Default search for MKL not supported, set MKL_DIRECTORY" )
    ENDIF()
    VERIFY_PATH( ${MKL_DIRECTORY} )
    VERIFY_PATH( ${MKL_DIRECTORY}/include )
    VERIFY_PATH( ${MKL_DIRECTORY}/lib )
    SET( MKL_LIB_PATH ${MKL_DIRECTORY}/lib )
    IF ( EXISTS ${MKL_DIRECTORY}/lib/intel64 )
        SET( MKL_LIB_PATH ${MKL_DIRECTORY}/lib/intel64 )
    ENDIF()
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${MKL_DIRECTORY}/include/mkl_blas.h\"\n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${MKL_DIRECTORY}/include/mkl_lapack.h\"\n" )
    FIND_LIBRARY( MKL_SEQ     NAMES libmkl_sequential.a     PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_SEQ     NAMES mkl_sequential          PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_CORE    NAMES libmkl_core.a           PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_CORE    NAMES mkl_core                PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_GF      NAMES libmkl_gf_lp64.a        PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_GF      NAMES mkl_gf_lp64             PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_LP      NAMES mkl_intel_lp64          PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_BLAS    NAMES libmkl_blas95_lp64.a    PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_BLAS    NAMES mkl_blas95_lp64         PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_LAPACK  NAMES libmkl_lapack95_lp64.a  PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    FIND_LIBRARY( MKL_LAPACK  NAMES mkl_lapack95_lp64       PATHS ${MKL_LIB_PATH}  NO_DEFAULT_PATH )
    SET( MKL_LIBS  )
    IF ( MKL_GF )
        SET( MKL_LIBS ${MKL_LIBS} ${MKL_GF})
    ENDIF()
    IF ( MKL_LP )
        SET( MKL_LIBS ${MKL_LIBS} ${MKL_LP})
    ENDIF()
    SET( MKL_LIBS ${MKL_LIBS} ${MKL_LAPACK} ${MKL_BLAS} ${MKL_SEQ} ${MKL_CORE} )
    MESSAGE( "Using mkl" )
    MESSAGE( "   ${MKL_LIBS}" )
ENDMACRO()


# Macro to configure ATLAS
MACRO( CONFIGURE_ATLAS )
    IF ( NOT ATLAS_DIRECTORY )
        MESSAGE(FATAL_ERROR "Default search for ATLAS not supported, set ATLAS_DIRECTORY" )
    ENDIF()
    VERIFY_PATH( ${ATLAS_DIRECTORY} )
    VERIFY_PATH( ${ATLAS_DIRECTORY}/include )
    VERIFY_PATH( ${ATLAS_DIRECTORY}/lib )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "extern \"C\"{ \n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${ATLAS_DIRECTORY}/include/cblas.h\"\n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "#include \"${ATLAS_DIRECTORY}/include/clapack.h\"\n" )
    FILE(APPEND ${BLAS_LAPACK_HEADER} "}\n" )
    FIND_LIBRARY( ATLAS_LAPACK  NAMES liblapack.a  PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_LAPACK  NAMES lapack       PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_F77BLAS NAMES libf77blas.a PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_F77BLAS NAMES f77blas      PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_CBLAS   NAMES libcblas.a   PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_CBLAS   NAMES cblas        PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_ATLAS   NAMES libatlas.a   PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY( ATLAS_ATLAS   NAMES atlas        PATHS ${ATLAS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    SET( ATLAS_LIBS ${ATLAS_LAPACK} ${ATLAS_F77BLAS} ${ATLAS_CBLAS} ${ATLAS_ATLAS} )
    MESSAGE( "Using atlas" )
    MESSAGE( "   ${ATLAS_LIBS}" )
ENDMACRO()


# Macro to configure the BLAS
MACRO( CONFIGURE_BLAS )
    # Determine if we want to use BLAS
    CHECK_ENABLE_FLAG( USE_BLAS 1 )
    IF ( USE_BLAS )
        IF ( BLAS_LIBRARIES )
            # The user is specifying the blas command directly
        ELSEIF ( BLAS_DIRECTORY )
            # The user is specifying the blas directory
            IF ( BLAS_LIB )
                # The user is specifying both the blas directory and the blas library
                FIND_LIBRARY ( BLAS_LIBRARIES NAMES ${BLAS_LIB} PATHS ${BLAS_DIRECTORY}  NO_DEFAULT_PATH )
                IF ( NOT BLAS_LIBRARIES )
                    MESSAGE ( FATAL_ERROR "BLAS library not found in ${BLAS_DIRECTORY}" )
                ENDIF()
            ELSE()
                # The user did not specify the library serach for a blas library
                FIND_LIBRARY ( BLAS_LIBRARIES NAMES blas PATHS ${BLAS_DIRECTORY}  NO_DEFAULT_PATH )
                IF ( NOT BLAS_LIBRARIES )
                    MESSAGE ( FATAL_ERROR "BLAS library not found in ${BLAS_DIRECTORY}" )
                ENDIF()
            ENDIF()
        ELSEIF ( BLAS_LIB )
            # The user is specifying the blas library (search for the file)
            FIND_LIBRARY ( BLAS_LIBRARIES NAMES ${BLAS_LIB} )
            IF ( NOT BLAS_LIBRARIES )
                MESSAGE ( FATAL_ERROR "BLAS library not found" )
            ENDIF()
        ELSE ()
            # The user did not include BLAS directly, perform a search
            INCLUDE ( FindBLAS )
            IF ( NOT BLAS_FOUND )
                MESSAGE ( FATAL_ERROR "BLAS not found.  Try setting BLAS_DIRECTORY or BLAS_LIB" )
            ENDIF()
        ENDIF()
        SET ( BLAS_LIBS ${BLAS_LIBRARIES} )
        MESSAGE ( "Using blas" )
        MESSAGE ( "   ${BLAS_LIBS}" )
    ENDIF()
ENDMACRO ()


# Macro to configure the LAPACK
MACRO( CONFIGURE_LAPACK )
    # Determine if we want to use LAPACK
    IF ( USE_LAPACK )
        IF ( LAPACK_LIBRARIES )
            # The user is specifying the lapack command directly
        ELSEIF ( LAPACK_DIRECTORY )
            # The user is specifying the lapack directory
            IF ( LAPACK_LIB )
                # The user is specifying both the lapack directory and the lapack library
                FIND_LIBRARY ( LAPACK_LIBRARIES NAMES ${LAPACK_LIB} PATHS ${LAPACK_DIRECTORY}  NO_DEFAULT_PATH )
                IF ( NOT LAPACK_LIBRARIES )
                    MESSAGE ( FATAL_ERROR "LAPACK library not found in ${LAPACK_DIRECTORY}" )
                ENDIF()
            ELSE()
                # The user did not specify the library serach for a lapack library
                FIND_LIBRARY ( LAPACK_LIBRARIES NAMES lapack PATHS ${LAPACK_DIRECTORY}  NO_DEFAULT_PATH )
                IF ( NOT LAPACK_LIBRARIES )
                    MESSAGE ( FATAL_ERROR "LAPACK library not found in ${LAPACK_DIRECTORY}" )
                ENDIF()
            ENDIF()
        ELSEIF ( LAPACK_LIB )
            # The user is specifying the lapack library (search for the file)
            FIND_LIBRARY ( LAPACK_LIBRARIES NAMES ${LAPACK_LIB} )
            IF ( NOT LAPACK_LIBRARIES )
                MESSAGE ( FATAL_ERROR "LAPACK library not found" )
            ENDIF()
        ELSE ()
            # The user did not include lapack directly, perform a search
            INCLUDE ( FindLAPACK )
            IF ( NOT LAPACK_FOUND )
                MESSAGE ( FATAL_ERROR "LAPACK not found.  Try setting LAPACK_DIRECTORY or LAPACK_LIB" )
            ENDIF()
        ENDIF()
        SET ( LAPACK_LIBS ${LAPACK_LIBRARIES} )
        MESSAGE ( "Using lapack" )
        MESSAGE ( "   ${LAPACK_LIBS}" )
    ENDIF()
ENDMACRO ()

