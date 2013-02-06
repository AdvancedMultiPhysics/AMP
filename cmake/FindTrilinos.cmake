# Function to find the trilinos version
FUNCTION ( TRILINOS_GET_VERSION )
    if (EXISTS "${TRILINOS_DIRECTORY}/include/Trilinos_version.h")
        FILE (STRINGS "${TRILINOS_DIRECTORY}/include/Trilinos_version.h" vstrings REGEX "#define TRILINOS_MAJOR_MINOR_VERSION ")
        foreach (line ${vstrings})
            STRING (REGEX REPLACE " +" ";" fields ${line}) # break line into three fields (the first is always "#define")
            LIST (GET fields 1 var)
            LIST (GET fields 2 val)
            SET (${var} ${val})         # Also in local scope so we have access below
        endforeach ()
        SET (TRILINOS_VERSION ${TRILINOS_MAJOR_MINOR_VERSION} PARENT_SCOPE)
    else ()
        MESSAGE (FATAL_ERROR "${TRILINOS_DIRECTORY}/include/Trilinos_version.h does not exist")
    endif ()
ENDFUNCTION ()


# Function to get/set the subpackages desired
MACRO ( TRILINOS_SET_SUBPACKAGES )
    # Set required libraries
    SET( USE_TRILINOS_VECTORS 1 )
    SET( USE_TRILINOS_TEUCHOS 1 )
    SET( USE_TRILINOS_UTILS 1 )
    SET( USE_TRILINOS_SOLVERS 1 )
    # Determine if we want to use thyra
    CHECK_ENABLE_FLAG( USE_TRILINOS_THYRA 0 )
    CHECK_ENABLE_FLAG( USE_TRILINOS_NOX 0 )
    IF ( USE_TRILINOS_THYRA OR USE_TRILINOS_NOX )
        SET( USE_TRILINOS_THYRA 1 )
        SET( USE_TRILINOS_NOX 1 )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_THYRA" )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_NOX" )
        MESSAGE ( "Using thyra and NOX" )
    ENDIF()
    # Check if we need stratimikos
    CHECK_ENABLE_FLAG( USE_TRILINOS_STRATIMIKOS 0 )
    # Determine if we want to use stkmesh
    CHECK_ENABLE_FLAG(USE_TRILINOS_STKMESH 0 )
    IF ( USE_TRILINOS_STKMESH )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_STKMESH" )  
        MESSAGE ( "Using stkmesh" )
    ENDIF()
ENDMACRO()


# Function to find the trilinos libraries
FUNCTION ( TRILINOS_SET_LIBRARIES )
    IF ( USE_UTILS_UTILS )
        FIND_TRILINOS_UTILS_LIBS()
    ELSE()
        SET( TRILINOS_UTILS_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_TEUCHOS )
        FIND_TRILINOS_TEUCHOS_LIBS()
    ELSE()
        SET( TRILINOS_TEUCHOS_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_VECTORS )
        FIND_TRILINOS_VEC_LIBS()
    ELSE()
        SET( TRILINOS_VEC_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_SOLVERS )
        FIND_TRILINOS_SOLVER_LIBS()
    ELSE()
        SET( TRILINOS_SOLVER_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_NOX )
        FIND_TRILINOS_NOX_LIBS()
    ELSE()
        SET( TRILINOS_NOX_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_STRATIMIKOS )
        FIND_TRILINOS_STRATIMIKOS_LIBS()
    ELSE()
        SET( TRILINOS_STRATIMIKOS_LIBS )
    ENDIF()
    IF ( USE_TRILINOS_STKMESH )
        FIND_TRILINOS_STKMESH_LIBS()
    ELSE()
        SET( TRILINOS_STKMESH_LIBS )
    ENDIF()
    # Get the complete list of libraries
    SET ( TRILINOS_LIBS
        ${TRILINOS_STRATIMIKOS_LIBS}
        ${TRILINOS_SOLVER_LIBS}
        ${TRILINOS_NOX_LIBS}
        ${TRILINOS_VEC_LIBS}
        ${TRILINOS_TEUCHOS_LIBS}
        ${TRILINOS_UTILS_LIBS}
        PARENT_SCOPE
    )
ENDFUNCTION ()


# Function to find the utility libraries
FUNCTION ( FIND_TRILINOS_TEUCHOS_LIBS )
    FIND_LIBRARY ( TRILINOS_TRIUTILIT_LIB   NAMES triutils    PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_GALERI_LIB      NAMES galeri      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    IF ( (NOT TRILINOS_TRIUTILIT_LIB) OR (NOT TRILINOS_GALERI_LIB) )
        MESSAGE ( ${TRILINOS_TRIUTILIT_LIB} )
        MESSAGE ( ${TRILINOS_GALERI_LIB} )
        MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
    ELSE()
        SET ( TRILINOS_UTILS_LIBS
            ${TRILINOS_TRIUTILIT_LIB}
            ${TRILINOS_GALERI_LIB}
            PARENT_SCOPE
        )
    ENDIF()
ENDFUNCTION()


# Function to find the TEUCHOS libraries
FUNCTION ( FIND_TRILINOS_TEUCHOS_LIBS )
    IF ( ${TRILINOS_VERSION} LESS 110100 )
        FIND_LIBRARY ( TRILINOS_TEUCHOS_LIB     NAMES teuchos     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        IF ( NOT TRILINOS_TEUCHOS_LIB )
            MESSAGE ( ${TRILINOS_TEUCHOS_LIB} )
            MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
        ENDIF()
        SET ( TRILINOS_TEUCHOS_LIBS ${TRILINOS_TEUCHOS_LIB} PARENT_SCOPE )
    ELSEIF( ${TRILINOS_VERSION} LESS 120000 )
        FIND_LIBRARY ( TRILINOS_TEUCHOSREMAINDER_LIBS  NAMES teuchosremainder      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_TEUCHOSNUMERICS_LIBS   NAMES teuchosnumerics       PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_TEUCHOSCOMM_LIBS       NAMES teuchoscomm           PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_TEUCHOSPARAMETER_LIBS  NAMES teuchosparameterlist  PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_TEUCHOSSCORE_LIBS      NAMES teuchoscore           PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        IF ( (NOT TRILINOS_TEUCHOSREMAINDER_LIBS) OR (NOT TRILINOS_TEUCHOSNUMERICS_LIBS) OR (NOT TRILINOS_TEUCHOSCOMM_LIBS) OR 
             (NOT TRILINOS_TEUCHOSPARAMETER_LIBS) OR (NOT TRILINOS_TEUCHOSSCORE_LIBS)  )
            MESSAGE ( ${TRILINOS_TEUCHOSREMAINDER_LIBS} )
            MESSAGE ( ${TRILINOS_TEUCHOSNUMERICS_LIBS} )
            MESSAGE ( ${TRILINOS_TEUCHOSCOMM_LIBS} )
            MESSAGE ( ${TRILINOS_TEUCHOSPARAMETER_LIBS} )
            MESSAGE ( ${TRILINOS_TEUCHOSSCORE_LIBS} )
            MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
        ELSE()
            SET ( TRILINOS_TEUCHOS_LIBS
                ${TRILINOS_TEUCHOSREMAINDER_LIBS}
                ${TRILINOS_TEUCHOSNUMERICS_LIBS}
                ${TRILINOS_TEUCHOSCOMM_LIBS}
                ${TRILINOS_TEUCHOSPARAMETER_LIBS}
                ${TRILINOS_TEUCHOSSCORE_LIBS}
                PARENT_SCOPE
            )
        ENDIF()
    ENDIF()
ENDFUNCTION()


# Function to find the vector libraries
FUNCTION ( FIND_TRILINOS_VEC_LIBS )
    IF ( ${TRILINOS_VERSION} LESS 110100 )
        FIND_LIBRARY ( TRILINOS_KOKKOS_LIBS     NAMES kokkos      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    ELSEIF( ${TRILINOS_VERSION} LESS 120000 )
        FIND_LIBRARY ( TRILINOS_KOKKOSDISTTSQR_LIB NAMES kokkosdisttsqr  PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_KOKKOSNODETSQR_LIB NAMES kokkosnodetsqr  PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_KOKKOSLINALG_LIB   NAMES kokkoslinalg    PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_KOKKOSNODEAPI_LIB  NAMES kokkosnodeapi   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_KOKKOS_LIB         NAMES kokkos          PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_TPI_LIB            NAMES tpi             PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        IF ( (NOT TRILINOS_KOKKOSDISTTSQR_LIB ) OR (NOT TRILINOS_KOKKOSNODETSQR_LIB ) OR (NOT TRILINOS_KOKKOSLINALG_LIB ) OR (NOT TRILINOS_KOKKOSNODEAPI_LIB) 
            OR (NOT TRILINOS_KOKKOS_LIB) OR (NOT TRILINOS_TPI_LIB) )                      
            MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
        ELSE()
            SET ( TRILINOS_KOKKOS_LIBS
               ${TRILINOS_KOKKOSDISTTSQR_LIB}
               ${TRILINOS_KOKKOSNODETSQR_LIB}
               ${TRILINOS_KOKKOSLINALG_LIB}
               ${TRILINOS_KOKKOSNODEAPI_LIB}
               ${TRILINOS_KOKKOS_LIB}
               ${TRILINOS_TPI_LIB}
            )
        ENDIF()
    ENDIF()
    FIND_LIBRARY ( TRILINOS_EPETRA_LIB      NAMES epetra      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_TPETRA_LIB      NAMES tpetra      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_EPETRAEXT_LIB   NAMES epetraext   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_RTOP_LIB        NAMES rtop        PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    IF ( (NOT TRILINOS_EPETRA_LIB) OR (NOT TRILINOS_EPETRAEXT_LIB) OR (NOT TRILINOS_KOKKOS_LIBS) OR (NOT TRILINOS_RTOP_LIB) )
        MESSAGE ( ${TRILINOS_EPETRA_LIB} )
        MESSAGE ( ${TRILINOS_EPETRAEXT_LIB} )
        MESSAGE ( ${TRILINOS_KOKKOS_LIBS} )
        MESSAGE ( ${TRILINOS_RTOP_LIB} )
        MESSAGE ( ${TRILINOS_TPETRA_LIB} )
        MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
    ELSE()
        SET ( TRILINOS_VEC_LIBS
            ${TRILINOS_TPETRA_LIB}
            ${TRILINOS_RTOP_LIB}
            ${TRILINOS_KOKKOS_LIBS}
            ${TRILINOS_EPETRAEXT_LIB}
            ${TRILINOS_EPETRA_LIB}
            PARENT_SCOPE
        )
    ENDIF()
ENDFUNCTION()


# Function to find the solver libraries
FUNCTION ( FIND_TRILINOS_SOLVER_LIBS )
    FIND_LIBRARY ( TRILINOS_AZTECOO_LIB     NAMES aztecoo     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_ML_LIB          NAMES ml          PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_BELOS_LIB       NAMES belos       PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IFPACK_LIB      NAMES ifpack      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_ZOLTAN_LIB      NAMES zoltan      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_AMESOS_LIB      NAMES amesos      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_LOCA_LIB        NAMES loca        PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_MOERTEL_LIB     NAMES moertel     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    IF ( (NOT TRILINOS_AZTECOO_LIB) OR (NOT TRILINOS_ML_LIB) OR (NOT TRILINOS_IFPACK_LIB) OR 
         (NOT TRILINOS_ZOLTAN_LIB)  OR  (NOT TRILINOS_AMESOS_LIB) OR (NOT TRILINOS_LOCA_LIB) OR 
         (NOT TRILINOS_MOERTEL_LIB) OR (NOT TRILINOS_BELOS_LIB) )
        MESSAGE ( ${TRILINOS_AZTECOO_LIB} )
        MESSAGE ( ${TRILINOS_BELOS_LIB} )
        MESSAGE ( ${TRILINOS_ML_LIB} )
        MESSAGE ( ${TRILINOS_IFPACK_LIB} )
        MESSAGE ( ${TRILINOS_ZOLTAN_LIB} )
        MESSAGE ( ${TRILINOS_AMESOS_LIB} )
        MESSAGE ( ${TRILINOS_LOCA_LIB} )
        MESSAGE ( ${TRILINOS_MOERTEL_LIB} )
        MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
    ELSE()
        SET ( TRILINOS_SOLVER_LIBS
            ${TRILINOS_ML_LIB}
            ${TRILINOS_BELOS_LIB}
            ${TRILINOS_AZTECOO_LIB}
            ${TRILINOS_IFPACK_LIB}
            ${TRILINOS_ZOLTAN_LIB}
            ${TRILINOS_AMESOS_LIB}
            ${TRILINOS_LOCA_LIB}
            ${TRILINOS_MOERTEL_LIB}
            PARENT_SCOPE
        )
    ENDIF()
ENDFUNCTION ()


# Function to find the stratimikos libraries
FUNCTION ( FIND_TRILINOS_STRATIMIKOS_LIBS )
    # Get the libs for stratimikos
    FIND_LIBRARY ( TRILINOS_STRATIMIKOS_LIB         NAMES stratimikos         PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STRATIMIKOSAMESOS_LIB   NAMES stratimikosamesos   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STRATIMIKOSAZTECOO_LIB  NAMES stratimikosaztecoo  PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STRATIMIKOSBELOS_LIB    NAMES stratimikosbelos    PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STRATIMIKOSIFPACK_LIB   NAMES stratimikosifpack   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STRATIMIKOSML_LIB       NAMES stratimikosml       PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    SET ( TRILINOS_STRATIMIKOS_LIBS
        ${TRILINOS_STRATIMIKOS_LIB}
        ${TRILINOS_STRATIMIKOSAMESOS_LIB}
        ${TRILINOS_STRATIMIKOSAZTECOO_LIB}
        ${TRILINOS_STRATIMIKOSBELOS_LIB}
        ${TRILINOS_STRATIMIKOSIFPACK_LIB}
        ${TRILINOS_STRATIMIKOSML_LIB}
        PARENT_SCOPE
    )
ENDFUNCTION ()


# Function to find the thyra/nox libraries
FUNCTION ( FIND_TRILINOS_NOX_LIBS )
    IF ( ${TRILINOS_VERSION} LESS 101000  )
        FIND_LIBRARY ( TRILINOS_THYRACORE_LIB   NAMES thyra   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        FIND_LIBRARY ( TRILINOS_NOXTHYRA_LIB    NAMES noxthyra   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    ELSEIF ( ${TRILINOS_VERSION} LESS 120000  )
        FIND_LIBRARY ( TRILINOS_THYRACORE_LIB   NAMES thyracore   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
        SET( TRILINOS_NOXTHYRA_LIB )
    ENDIF()
    FIND_LIBRARY ( TRILINOS_THYRAEPETRA_LIB NAMES thyraepetra PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_NOX_LIB         NAMES nox         PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
    IF ( (NOT TRILINOS_THYRACORE_LIB) OR (NOT TRILINOS_THYRAEPETRA_LIB) OR (NOT TRILINOS_NOX_LIB) )
        MESSAGE ( ${TRILINOS_THYRACORE_LIB} )
        MESSAGE ( ${TRILINOS_THYRAEPETRA_LIB} )
        MESSAGE ( ${TRILINOS_NOX_LIB} )
        MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
    ELSE()
        SET ( TRILINOS_NOX_LIBS
            ${TRILINOS_THYRAEPETRA_LIB}
            ${TRILINOS_THYRACORE_LIB}
            ${TRILINOS_NOX_LIB}
            ${TRILINOS_NOXTHYRA_LIB}
            PARENT_SCOPE
        )
    ENDIF()
ENDFUNCTION ()


# Function to find the stkmesh libraries
FUNCTION ( FIND_TRILINOS_STKMESH_LIBS )
    FIND_LIBRARY ( TRILINOS_SHARDS_LIB    NAMES shards    PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_INTREPID_LIB  NAMES intrepid  PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_IO_LIB    NAMES stk_io    PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_IO_UTIL_LIB NAMES stk_io_util PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_UTIL_UTIL_LIB NAMES stk_util_util PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_UTIL_PARALLEL_LIB NAMES stk_util_parallel PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_UTIL_ENV_LIB NAMES stk_util_env PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_MESH_BASE_LIB NAMES stk_mesh_base PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_STK_MESH_FEM_LIB NAMES stk_mesh_fem PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOSS_LIB NAMES Ioss PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOGN_LIB NAMES Iogn PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOHB_LIB NAMES Iohb PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOPG_LIB NAMES Iopg PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOTR_LIB NAMES Iotr PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IONIT_LIB NAMES Ionit PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_IOEX_LIB NAMES Ioex PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_EXODUS_LIB NAMES exodus PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    FIND_LIBRARY ( TRILINOS_PAMGEN_LIB NAMES pamgen PATHS ${TRILINOS_DIRECTORY}/lib  NO_DEFAULT_PATH )
    IF ( (NOT TRILINOS_SHARDS_LIB) OR 
         (NOT TRILINOS_INTREPID_LIB) OR 
         (NOT TRILINOS_STK_IO_LIB) OR 
         (NOT TRILINOS_STK_IO_UTIL_LIB) OR  
         (NOT TRILINOS_STK_UTIL_UTIL_LIB) OR
         (NOT TRILINOS_STK_UTIL_PARALLEL_LIB) OR 
         (NOT TRILINOS_STK_UTIL_ENV_LIB) OR
         (NOT TRILINOS_STK_MESH_BASE_LIB) OR 
         (NOT TRILINOS_STK_MESH_FEM_LIB) OR 
         (NOT TRILINOS_IOSS_LIB) OR 
         (NOT TRILINOS_IOGN_LIB)  OR 
         (NOT TRILINOS_IOHB_LIB) OR
         (NOT TRILINOS_IOPG_LIB) OR
         (NOT TRILINOS_IOTR_LIB) OR
         (NOT TRILINOS_IONIT_LIB) OR 
         (NOT TRILINOS_IOEX_LIB) OR 
         (NOT TRILINOS_EXODUS_LIB) OR
         (NOT TRILINOS_PAMGEN_LIB) )
        MESSAGE ( ${TRILINOS_SHARDS_LIB} )
        MESSAGE ( ${TRILINOS_INTREPID_LIB} )
        MESSAGE ( ${TRILINOS_STK_IO_LIB} )
        MESSAGE ( ${TRILINOS_STK_IO_UTIL_LIB} )
        MESSAGE ( ${TRILINOS_STK_UTIL_UTIL_LIB} )
        MESSAGE ( ${TRILINOS_STK_UTIL_PARALLEL_LIB} )
        MESSAGE ( ${TRILINOS_STK_UTIL_ENV_LIB} )
        MESSAGE ( ${TRILINOS_STK_MESH_BASE_LIB} )
        MESSAGE ( ${TRILINOS_STK_MESH_FEM_LIB} )
        MESSAGE ( ${TRILINOS_IOSS_LIB} )
        MESSAGE ( ${TRILINOS_IOGN_LIB} )
        MESSAGE ( ${TRILINOS_IOHB_LIB} )
        MESSAGE ( ${TRILINOS_IOPG_LIB} )
        MESSAGE ( ${TRILINOS_IOTR_LIB} )
        MESSAGE ( ${TRILINOS_IONIT_LIB} )
        MESSAGE ( ${TRILINOS_IOEX_LIB} )
        MESSAGE ( ${TRILINOS_EXODUS_LIB} )
        MESSAGE ( ${TRILINOS_TEUCHOS_LIB} )
        MESSAGE ( ${TRILINOS_MOERTEL_LIB} )
        MESSAGE ( ${TRILINOS_PAMGEN_LIB} )
        MESSAGE ( FATAL_ERROR "Trilinos libraries (stkmesh) not found in ${TRILINOS_DIRECTORY}/lib" )
    ENDIF()
    SET ( TRILINOS_STKMESH_LIBS 
        ${TRILINOS_SHARDS_LIB}
        ${TRILINOS_INTREPID_LIB}
        ${TRILINOS_STK_IO_LIB}
        ${TRILINOS_STK_IO_UTIL_LIB}
        ${TRILINOS_STK_UTIL_UTIL_LIB}
        ${TRILINOS_STK_UTIL_PARALLEL_LIB}
        ${TRILINOS_STK_UTIL_ENV_LIB}
        ${TRILINOS_STK_MESH_FEM_LIB}
        ${TRILINOS_STK_MESH_BASE_LIB}
        ${TRILINOS_IOSS_LIB}
        ${TRILINOS_IOGN_LIB}
        ${TRILINOS_IOHB_LIB}
        ${TRILINOS_IOPG_LIB}
        ${TRILINOS_IOTR_LIB}
        ${TRILINOS_IONIT_LIB}
        ${TRILINOS_IOEX_LIB}
        ${TRILINOS_SHARDS_LIB}
        ${TRILINOS_INTREPID_LIB}
        ${TRILINOS_STK_IO_LIB}
        ${TRILINOS_STK_IO_UTIL_LIB}
        ${TRILINOS_STK_UTIL_UTIL_LIB}
        ${TRILINOS_STK_UTIL_PARALLEL_LIB}
        ${TRILINOS_STK_UTIL_ENV_LIB}
        ${TRILINOS_STK_MESH_FEM_LIB}
        ${TRILINOS_STK_MESH_BASE_LIB}
        ${TRILINOS_IOSS_LIB}
        ${TRILINOS_IOGN_LIB}
        ${TRILINOS_IOHB_LIB}
        ${TRILINOS_IOPG_LIB}
        ${TRILINOS_IOTR_LIB}
        ${TRILINOS_IONIT_LIB}
        ${TRILINOS_IOEX_LIB}
        ${TRILINOS_EXODUS_LIB}
        ${TRILINOS_PAMGEN_LIB}
        ${TRILINOS_TEUCHOS_LIB}
        ${TRILINOS_MOERTEL_LIB}
        PARENT_SCOPE
    )
ENDFUNCTION ()


