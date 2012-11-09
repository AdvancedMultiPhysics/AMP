INCLUDE ( FindPetsc )

MACRO ( CONFIGURE_LINE_COVERAGE )
    SET ( COVERAGE_LIBS )
    IF ( USE_EXT_GCOV )
        SET ( COVERAGE_LIBS ${COVERAGE_LIBS} -lgcov )
    ENDIF ()
    IF ( ENABLE_GCOV )
        ADD_DEFINITIONS ( -fprofile-arcs -ftest-coverage )
        SET ( COVERAGE_LIBS ${COVERAGE_LIBS} -fprofile-arcs )
    ENDIF ()
ENDMACRO ()


MACRO ( CONFIGURE_TIMERS )
  CHECK_INCLUDE_FILE ( sys/times.h HAVE_SYS_TIMES_H )
  CHECK_INCLUDE_FILE ( windows.h HAVE_WINDOWS_H )
ENDMACRO ()


# Macro to find and configure boost (we only need the headers)
MACRO ( CONFIGURE_BOOST )
    # Determine if we want to use boost
    CHECK_ENABLE_FLAG(USE_EXT_BOOST 1 )
    IF ( USE_EXT_BOOST )
        # Check if we specified the boost directory
        IF ( BOOST_DIRECTORY )
            VERIFY_PATH ( ${BOOST_DIRECTORY} )
            VERIFY_PATH ( ${BOOST_DIRECTORY}/include )
            SET ( BOOST_INCLUDE ${BOOST_DIRECTORY}/include )
        ELSE()
            # Check the default path for boost
            VERIFY_PATH ( ${AMP_SOURCE_DIR}/../external/boost/include )
            SET ( BOOST_INCLUDE ${AMP_SOURCE_DIR}/../external/boost/include )
        ENDIF()
        INCLUDE_DIRECTORIES ( ${BOOST_INCLUDE} )
        ADD_DEFINITIONS ( "-D USE_EXT_BOOST" )
        MESSAGE ( "Using boost" )
    ELSE()
        MESSAGE( FATAL_ERROR "boost headers are necessary for AMP" )
    ENDIF()
ENDMACRO()


# Macro to find and configure netcdf
MACRO ( CONFIGURE_NETCDF )
    CHECK_ENABLE_FLAG(USE_EXT_NETCDF 0 )
    IF ( USE_EXT_NETCDF )
        IF ( NETCDF_DIRECTORY )
            VERIFY_PATH ( ${NETCDF_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${NETCDF_DIRECTORY}/include )
            SET ( NETCDF_INCLUDE ${NETCDF_DIRECTORY}/include )
            FIND_LIBRARY ( NETCDF_NETCDF_LIB    NAMES netcdf    PATHS ${NETCDF_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( NETCDF_HDF5_LIB      NAMES hdf5      PATHS ${NETCDF_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( NETCDF_HL_LIB        NAMES hl        PATHS ${NETCDF_DIRECTORY}/lib  NO_DEFAULT_PATH )
            IF ( (NOT NETCDF_NETCDF_LIB) OR (NOT NETCDF_HDF5_LIB) OR (NOT NETCDF_HL_LIB)  )
                MESSAGE ( ${NETCDF_NETCDF_LIB} )
                MESSAGE ( ${NETCDF_HDF5_LIB} )
                MESSAGE ( ${NETCDF_HL_LIB} )
            ENDIF()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for netcdf is not yet supported.  Use -D NETCDF_DIRECTORY=" )
        ENDIF()
        SET ( NETCDF_LIBS
            ${NETCDF_NETCDF_LIB} 
            ${NETCDF_HDF5_LIB} 
            ${NETCDF_HL_LIB} 
            ${NETCDF_NETCDF_LIB} 
            ${NETCDF_HDF5_LIB} 
            ${NETCDF_HL_LIB} 
        )
        ADD_DEFINITIONS ( "-D USE_NETCDF" )
        MESSAGE ( "Using netcdf" )
        MESSAGE ( "   " ${NETCDF_LIBS} )
    ENDIF()
ENDMACRO()


# Macro to find and configure the trilinos libraries
MACRO ( CONFIGURE_TRILINOS_LIBRARIES )
    # Determine if we want to use trilinos
    CHECK_ENABLE_FLAG(USE_EXT_TRILINOS 1 )
    IF ( USE_EXT_TRILINOS )
        # Check if we specified the trilinos directory
        IF ( TRILINOS_DIRECTORY )
            VERIFY_PATH ( ${TRILINOS_DIRECTORY} )
            MESSAGE ( " Trilinos Directory " ${TRILINOS_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${TRILINOS_DIRECTORY}/include )
            SET ( TRILINOS_INCLUDE ${TRILINOS_DIRECTORY}/include )
            FIND_LIBRARY ( TRILINOS_EPETRA_LIB      NAMES epetra      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_EPETRAEXT_LIB   NAMES epetraext   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_TRIUTILIT_LIB   NAMES triutils    PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_AZTECOO_LIB     NAMES aztecoo     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_GALERI_LIB      NAMES galeri      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_ML_LIB          NAMES ml          PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_IFPACK_LIB      NAMES ifpack      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_ZOLTAN_LIB      NAMES zoltan      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_AMESOS_LIB      NAMES amesos      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_LOCA_LIB        NAMES loca        PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_NOX_LIB         NAMES nox         PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_TEUCHOS_LIB     NAMES teuchos     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_MOERTEL_LIB     NAMES moertel     PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_TPETRA_LIB      NAMES tpetra      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_KOKKOS_LIB      NAMES kokkos      PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_THYRAEPETRA_LIB NAMES thyraepetra PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_THYRACORE_LIB   NAMES thyracore   PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            FIND_LIBRARY ( TRILINOS_RTOP_LIB        NAMES rtop        PATHS ${TRILINOS_DIRECTORY}/lib   NO_DEFAULT_PATH )
            IF ( (NOT TRILINOS_EPETRA_LIB) OR 
                 (NOT TRILINOS_EPETRAEXT_LIB) OR 
                 (NOT TRILINOS_TRIUTILIT_LIB) OR 
                 (NOT TRILINOS_AZTECOO_LIB) OR 
                 (NOT TRILINOS_ML_LIB) OR 
                 (NOT TRILINOS_IFPACK_LIB) OR 
                 (NOT TRILINOS_ZOLTAN_LIB) OR 
                 (NOT TRILINOS_AMESOS_LIB) OR 
                 (NOT TRILINOS_LOCA_LIB) OR
                 (NOT TRILINOS_NOX_LIB) OR 
                 (NOT TRILINOS_TEUCHOS_LIB) OR 
                 (NOT TRILINOS_MOERTEL_LIB) OR 
                 (NOT TRILINOS_TPETRA_LIB) OR
                 (NOT TRILINOS_KOKKOS_LIB) OR
                 (NOT TRILINOS_THYRAEPETRA_LIB) OR
                 (NOT TRILINOS_THYRACORE_LIB) OR
                 (NOT TRILINOS_RTOP_LIB) )
                MESSAGE ( ${TRILINOS_EPETRA_LIB} )
                MESSAGE ( ${TRILINOS_EPETRAEXT_LIB} )
                MESSAGE ( ${TRILINOS_TRIUTILIT_LIB} )
                MESSAGE ( ${TRILINOS_TPETRA_LIB} )
                MESSAGE ( ${TRILINOS_AZTECOO_LIB} )
                MESSAGE ( ${TRILINOS_ML_LIB} )
                MESSAGE ( ${TRILINOS_GALERI_LIB} )
                MESSAGE ( ${TRILINOS_IFPACK_LIB} )
                MESSAGE ( ${TRILINOS_ZOLTAN_LIB} )
                MESSAGE ( ${TRILINOS_AMESOS_LIB} )
                MESSAGE ( ${TRILINOS_LOCA_LIB} )
                MESSAGE ( ${TRILINOS_NOX_LIB} )
                MESSAGE ( ${TRILINOS_KOKKOS_LIB} )
                MESSAGE ( ${TRILINOS_THYRAEPETRA_LIB} )
                MESSAGE ( ${TRILINOS_THYRACORE_LIB} )
                MESSAGE ( ${TRILINOS_RTOP_LIB} )
                MESSAGE ( FATAL_ERROR "Trilinos libraries not found in ${TRILINOS_DIRECTORY}/lib" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for trilinos is not yet supported.  Use -D TRILINOS_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        SET ( TRILINOS_LIBS
            ${TRILINOS_THYRAEPETRA_LIB}
            ${TRILINOS_THYRACORE_LIB}
            ${TRILINOS_RTOP_LIB}
            ${TRILINOS_ML_LIB}
            ${TRILINOS_GALERI_LIB}
            ${TRILINOS_TRIUTILIT_LIB}
            ${TRILINOS_IFPACK_LIB}
            ${TRILINOS_AZTECOO_LIB}
            ${TRILINOS_AMESOS_LIB}
            ${TRILINOS_EPETRAEXT_LIB}
            ${TRILINOS_EPETRA_LIB}
            ${TRILINOS_TPETRA_LIB}
            ${TRILINOS_ZOLTAN_LIB}
            ${TRILINOS_LOCA_LIB}
            ${TRILINOS_NOX_LIB}
            ${TRILINOS_KOKKOS_LIB}
            ${TRILINOS_TEUCHOS_LIB}
            ${TRILINOS_MOERTEL_LIB}
        )
        IF ( USE_EXT_GCOV )
            SET ( TRILINOS_LIBS ${TRILINOS_LIBS} -lgcov )
        ENDIF ()
        ADD_DEFINITIONS ( "-D USE_EXT_TRILINOS" )
        MESSAGE ( "Using trilinos" )
        MESSAGE ( "   " ${TRILINOS_LIBS} )
    ENDIF()
ENDMACRO ()



# Macro to find and configure the stkmesh libraries
MACRO ( CONFIGURE_STKMESH )
    # Determine if we want to use stkmesh
    CHECK_ENABLE_FLAG(USE_EXT_STKMESH 0 )
    IF ( USE_EXT_STKMESH )
        IF ( NOT USE_EXT_TRILINOS )
            MESSAGE( FATAL_ERROR "stkmesh requires trilinos" )
        ENDIF()
        ADD_DEFINITIONS ( "-D USE_EXT_STKMESH" )  
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
        SET ( TRILINOS_LIBS ${TRILINOS_LIBS}
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
        )
        MESSAGE ( "Using stkmesh" )
    ENDIF()
ENDMACRO ()



# Macro to find and configure the silo libraries
MACRO ( CONFIGURE_SILO )
    # Determine if we want to use silo
    CHECK_ENABLE_FLAG(USE_EXT_SILO 1 )
    IF ( USE_EXT_SILO )
        # Check if we specified the silo directory
        IF ( SILO_DIRECTORY )
            VERIFY_PATH ( ${SILO_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${SILO_DIRECTORY}/include )
            SET ( SILO_INCLUDE ${SILO_DIRECTORY}/include )
            FIND_LIBRARY ( SILO_LIB  NAMES siloh5  PATHS ${SILO_DIRECTORY}/lib  NO_DEFAULT_PATH )
        ELSE()
            MESSAGE ( "Default search for silo is not yet supported")
            MESSAGE ( "Use -D SILO_DIRECTORY=" FATAL_ERROR)
        ENDIF()
        SET ( SILO_LIBS
            ${SILO_LIB}
        )
        ADD_DEFINITIONS ( "-D USE_EXT_SILO" )  
        MESSAGE ( "Using silo" )
    ENDIF ()
ENDMACRO ()


# Macro to find and configure the hdf5 libraries
MACRO ( CONFIGURE_HDF5 )
    # Determine if we want to use hdf5
    CHECK_ENABLE_FLAG(USE_EXT_HDF5 1 )
    IF ( USE_EXT_HDF5 )
        # Check if we specified the silo directory
        IF ( HDF5_DIRECTORY )
            VERIFY_PATH ( ${HDF5_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${HDF5_DIRECTORY}/include )
            SET ( HDF5_INCLUDE ${HDF5_DIRECTORY}/include )
            FIND_LIBRARY ( HDF5_LIB    NAMES hdf5    PATHS ${HDF5_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HDF5_HL_LIB NAMES hdf5_hl PATHS ${HDF5_DIRECTORY}/lib  NO_DEFAULT_PATH )
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for hdf5 is not yet supported.  Use -D HDF5_DIRECTORY=" )
        ENDIF()
        SET ( HDF5_LIBS
            ${HDF5_HL_LIB}
            ${HDF5_LIB}
        )
        ADD_DEFINITIONS ( "-D USE_EXT_HDF5" )  
        MESSAGE ( "Using hdf5" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the X11 libraries
MACRO ( CONFIGURE_X11_LIBRARIES )
    # Determine if we want to use X11
    CHECK_ENABLE_FLAG(USE_EXT_X11 1 )
    IF ( USE_EXT_X11 )
        # Check if we specified the silo directory
        IF ( X11_DIRECTORY )
            VERIFY_PATH ( ${X11_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${X11_DIRECTORY}/include )
            SET ( X11_INCLUDE ${X11_DIRECTORY}/include )
            FIND_LIBRARY ( X11_SM_LIB  NAMES SM  PATHS ${X11_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( X11_ICE_LIB NAMES ICE PATHS ${X11_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( X11_X11_LIB NAMES X11 PATHS ${X11_DIRECTORY}/lib  NO_DEFAULT_PATH )
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for X11 is not yet supported.  Use -D X11_DIRECTORY=" )
        ENDIF()
        SET ( X11_LIBS
            ${X11_SM_LIB}
            ${X11_ICE_LIB}
            ${X11_X11_LIB} 
        )
        ADD_DEFINITIONS ( "-D USE_EXT_X11" )  
        MESSAGE ( "Using X11" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the MPI libraries
MACRO ( CONFIGURE_MPI )
    # Determine if we want to use MPI
    CHECK_ENABLE_FLAG(USE_EXT_MPI 1 )
    IF ( USE_EXT_MPI )
        # Check if we specified the MPI directory
        IF ( MPI_DIRECTORY )
            # Check the provided MPI directory for include files and the mpi executable
            VERIFY_PATH ( ${MPI_DIRECTORY} )
            SET ( MPI_INCLUDE_PATH ${MPI_DIRECTORY}/include )
            VERIFY_PATH ( ${MPI_INCLUDE_PATH} )
            IF ( NOT EXISTS ${MPI_INCLUDE_PATH}/mpi.h )
                MESSAGE ( FATAL_ERROR "mpi.h not found in ${MPI_INCLUDE_PATH}/include" )
            ENDIF ()
            INCLUDE_DIRECTORIES ( ${MPI_INCLUDE_PATH} )
            SET ( MPI_INCLUDE ${MPI_INCLUDE_PATH} )
            IF ( MPIEXEC ) 
                # User specified the MPI command directly, use as is
            ELSEIF ( MPIEXEC_CMD )
                # User specified the name of the MPI executable
                SET ( MPIEXEC ${MPI_DIRECTORY}/bin/${MPIEXEC_CMD} )
                IF ( NOT EXISTS ${MPIEXEC} )
                    MESSAGE ( FATAL_ERROR "${MPIEXEC_CMD} not found in ${MPI_DIRECTORY}/bin" )
                ENDIF ()
            ELSE ()
                # Search for the MPI executable in the current directory
                FIND_PROGRAM ( MPIEXEC  NAMES mpiexec mpirun lamexec  PATHS ${MPI_DIRECTORY}/bin  NO_DEFAULT_PATH )
                IF ( NOT MPIEXEC )
                    MESSAGE ( FATAL_ERROR "Could not locate mpi executable" )
                ENDIF()
            ENDIF ()
            # Set MPI flags
            IF ( NOT MPIEXEC_NUMPROC_FLAG )
                SET( MPIEXEC_NUMPROC_FLAG "-np" )
            ENDIF()
        ELSE()
            # Perform the default search for MPI
            INCLUDE ( FindMPI )
            IF ( NOT MPI_FOUND )
                MESSAGE ( FATAL_ERROR "Did not find MPI" )
            ENDIF ()
            INCLUDE_DIRECTORIES ( ${MPI_INCLUDE_PATH} )
            SET ( MPI_INCLUDE ${MPI_INCLUDE_PATH} )
        ENDIF()
        # Check if we need to use MPI for serial tests
        CHECK_ENABLE_FLAG( USE_EXT_MPI_FOR_SERIAL_TESTS 0 )
        # Set the definitions
        ADD_DEFINITIONS ( "-D USE_EXT_MPI" )  
        MESSAGE ( "Using MPI" )
        MESSAGE ( "  MPIEXEC = ${MPIEXEC}" )
        MESSAGE ( "  MPIEXEC_NUMPROC_FLAG = ${MPIEXEC_NUMPROC_FLAG}" )
        MESSAGE ( "  MPI_LINK_FLAGS = ${MPI_LINK_FLAGS}" )
        MESSAGE ( "  MPI_LIBRARIES = ${MPI_LIBRARIES}" )
    ELSE()
        SET( USE_EXT_MPI_FOR_SERIAL_TESTS 0 )
        MESSAGE ( "Not using MPI, all parallel tests will be disabled" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the libmesh libraries
MACRO ( CONFIGURE_LIBMESH )
    # Determine if we want to use libmesh
    CHECK_ENABLE_FLAG(USE_EXT_LIBMESH 1 )
    IF ( USE_EXT_LIBMESH )
        # Check if we specified the libmesh directory
        IF ( LIBMESH_DIRECTORY )
            VERIFY_PATH ( ${LIBMESH_DIRECTORY} )
            # Include the libmesh directories
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/base )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/enums )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/error_estimation )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/fe )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/geom )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/mesh )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/numerics )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/parallel )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/partitioning )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/quadrature )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/solvers )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/systems )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/include/utils )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/contrib/exodusii/Lib/include/ )
            SET ( LIBMESH_INCLUDE ${LIBMESH_INCLUDE} ${LIBMESH_DIRECTORY}/contrib/netcdf/Lib/ )
            INCLUDE_DIRECTORIES ( ${LIBMESH_INCLUDE} )
            # Find the libmesh libaries
            SET ( LIBMESH_PATH_LIB ${LIBMESH_DIRECTORY}/lib/${LIBMESH_HOSTTYPE}_${LIBMESH_COMPILE_TYPE} )
            SET ( LIBMESH_CONTRIB_PATH_LIB ${LIBMESH_DIRECTORY}/contrib/lib/${LIBMESH_HOSTTYPE}_${LIBMESH_COMPILE_TYPE} )
            VERIFY_PATH ( ${LIBMESH_PATH_LIB} )
            VERIFY_PATH ( ${LIBMESH_CONTRIB_PATH_LIB} )
            FIND_LIBRARY ( LIBMESH_MESH_LIB     NAMES mesh      PATHS ${LIBMESH_PATH_LIB}          NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_EXODUSII_LIB NAMES exodusii  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_LASPACK_LIB  NAMES laspack   PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_METIS_LIB    NAMES metis     PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_NEMESIS_LIB  NAMES nemesis   PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_NETCDF_LIB   NAMES netcdf    PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            IF ( USE_EXT_MPI ) 
                FIND_LIBRARY ( LIBMESH_PARMETIS_LIB NAMES parmetis  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            ENDIF()
            FIND_LIBRARY ( LIBMESH_SFCURVES_LIB NAMES sfcurves  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_GMV_LIB      NAMES gmv       PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_GZSTREAM_LIB NAMES gzstream  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_TETGEN_LIB   NAMES tetgen    PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( LIBMESH_TRIANGLE_LIB NAMES triangle  PATHS ${LIBMESH_CONTRIB_PATH_LIB}  NO_DEFAULT_PATH )
            IF ( ${CMAKE_SYSTEM} MATCHES ^Linux.* )
                IF ( ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL x86_64 )
                    SET ( LIBMESH_TECIO_LIB  ${LIBMESH_DIRECTORY}/contrib/tecplot/lib/x86_64-unknown-linux-gnu/tecio.a )
                ENDIF ()
            ENDIF ()
            IF ( NOT LIBMESH_MESH_LIB )
                MESSAGE ( FATAL_ERROR "Libmesh library (mesh) not found in ${LIBMESH_PATH_LIB}" )
            ENDIF ()
            IF ( (NOT LIBMESH_EXODUSII_LIB) OR (NOT LIBMESH_GMV_LIB) OR (NOT LIBMESH_GZSTREAM_LIB) OR
                 (NOT LIBMESH_LASPACK_LIB) OR 
                 (NOT LIBMESH_NEMESIS_LIB) OR (NOT LIBMESH_NETCDF_LIB) OR (NOT LIBMESH_METIS_LIB) OR 
                 (NOT LIBMESH_SFCURVES_LIB) OR (NOT LIBMESH_TETGEN_LIB) OR (NOT LIBMESH_TRIANGLE_LIB) )
                MESSAGE ( ${LIBMESH_EXODUSII_LIB} )
                MESSAGE ( ${LIBMESH_LASPACK_LIB} )
                MESSAGE ( ${LIBMESH_NEMESIS_LIB} )
                MESSAGE ( ${LIBMESH_NETCDF_LIB} )
                MESSAGE ( ${LIBMESH_METIS_LIB} )
                MESSAGE ( ${LIBMESH_SFCURVES_LIB} )
                MESSAGE ( ${LIBMESH_GMV_LIB} )
                MESSAGE ( ${LIBMESH_GZSTREAM_LIB} )
                MESSAGE ( ${LIBMESH_TETGEN_LIB} )
                MESSAGE ( ${LIBMESH_TRIANGLE_LIB} )
                MESSAGE ( FATAL_ERROR "Libmesh contribution libraries not found in ${LIBMESH_PATH_LIB}" )
            ENDIF ()
            IF ( USE_EXT_MPI AND (NOT LIBMESH_PARMETIS_LIB) )
                MESSAGE ( ${LIBMESH_PARMETIS_LIB} )
                MESSAGE ( FATAL_ERROR "Libmesh contribution libraries not found in ${LIBMESH_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for libmesh is not supported.  Use -D LIBMESH_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        SET ( LIBMESH_LIBS
            ${LIBMESH_MESH_LIB}
            ${LIBMESH_EXODUSII_LIB}
            ${LIBMESH_LASPACK_LIB}
            ${LIBMESH_NEMESIS_LIB}
            ${LIBMESH_NETCDF_LIB}
         )
        IF ( USE_EXT_MPI ) 
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
        ADD_DEFINITIONS ( -DLIBMESH_ENABLE_PARMESH )
        ADD_DEFINITIONS ( "-D USE_EXT_LIBMESH" )  
        MESSAGE ( "Using libmesh" )
        MESSAGE ( "   " ${LIBMESH_LIBS} )
    ENDIF()
ENDMACRO ()


# Macro to find and configure NEK
MACRO ( CONFIGURE_NEK )
    # Determine if we want to use NEK
    CHECK_ENABLE_FLAG( USE_EXT_NEK "false" )
    IF ( USE_EXT_NEK )
        # Check if we specified the NEK directory
        IF ( NEK_DIRECTORY )
            VERIFY_PATH ( ${NEK_DIRECTORY} )
            # Include the NEK directories
            IF ( NOT NEK_INCLUDE )
                SET ( NEK_INCLUDE ${NEK_DIRECTORY} )
            ENDIF()
            # Find the NEK libaries
            IF ( NOT NEK_PATH_LIB )
                SET ( NEK_PATH_LIB ${NEK_DIRECTORY} )
            ENDIF()
            VERIFY_PATH ( ${NEK_PATH_LIB} )
            FIND_LIBRARY ( NEK_LIB     NAMES NEK5000      PATHS ${NEK_PATH_LIB}          NO_DEFAULT_PATH )
            IF ( NOT NEK_LIB )
                MESSAGE ( FATAL_ERROR "Nek5000 library (NEK5000) not found in ${NEK_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for NEK is not supported.  Use -D NEK_DIRECTORY=" )
        ENDIF()
        CHECK_ENABLE_FLAG( NOTIMER  0 )
        CHECK_ENABLE_FLAG( MPITIMER 0 )
        CHECK_ENABLE_FLAG( MPIIO    0 )
        CHECK_ENABLE_FLAG( BG       0 )
        CHECK_ENABLE_FLAG( K10_MXM  0 )
        CHECK_ENABLE_FLAG( CVODE    0 )
        CHECK_ENABLE_FLAG( NEKNEK   0 )
        CHECK_ENABLE_FLAG( MOAB     1 )
        IF ( NOT USE_EXT_MOAB ) 
            MESSAGE ( FATAL_ERROR "Within AMP, MOAB is required to use Nek5000." )
        ENDIF()
        # Add the libraries in the appropriate order
        INCLUDE_DIRECTORIES ( ${NEK_INCLUDE} )
        SET ( NEK_LIBS
            ${NEK_LIB}
        )
        ADD_DEFINITIONS ( "-D USE_EXT_NEK" )  
        MESSAGE ( "Using NEK" )
        MESSAGE ( "   " ${NEK_LIBS} )
        SET ( CURPACKAGE "nek" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure DENDRO
MACRO ( CONFIGURE_DENDRO )
    # Determine if we want to use DENDRO
    CHECK_ENABLE_FLAG( USE_EXT_DENDRO "false" )
    IF ( USE_EXT_DENDRO )
        IF ( DENDRO_DIRECTORY )
            VERIFY_PATH ( ${DENDRO_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${DENDRO_DIRECTORY}/include )
            SET ( DENDRO_INCLUDE ${DENDRO_DIRECTORY}/include )
            FIND_LIBRARY ( DENDRO_BIN_LIB   NAMES BinOps PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_OCT_LIB   NAMES Oct    PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_PAR_LIB   NAMES Par    PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_POINT_LIB NAMES Point  PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( DENDRO_TEST_LIB  NAMES Test   PATHS ${DENDRO_DIRECTORY}/lib  NO_DEFAULT_PATH )
            IF ( (NOT DENDRO_BIN_LIB) OR (NOT DENDRO_OCT_LIB) OR (NOT DENDRO_PAR_LIB) OR
                (NOT DENDRO_POINT_LIB) OR (NOT DENDRO_TEST_LIB) )
                MESSAGE ( ${DENDRO_BIN_LIB} )
                MESSAGE ( ${DENDRO_OCT_LIB} )
                MESSAGE ( ${DENDRO_PAR_LIB} )
                MESSAGE ( ${DENDRO_POINT_LIB} )
                MESSAGE ( ${DENDRO_TEST_LIB} )
                MESSAGE ( FATAL_ERROR "DENDRO libraries not found in ${DENDRO_DIRECTORY}/lib" )
            ENDIF ()
            # Add the libraries in the appropriate order
            SET ( DENDRO_LIBS
                ${DENDRO_OCT_LIB}
                ${DENDRO_PAR_LIB}
                ${DENDRO_POINT_LIB}
                ${DENDRO_TEST_LIB}
                ${DENDRO_BIN_LIB}
             )
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for DENDRO is not supported.  Use -D DENDRO_DIRECTORY=" )
        ENDIF()
        ADD_DEFINITIONS ( "-D USE_EXT_DENDRO" )  
        MESSAGE ( "Using DENDRO" )
        MESSAGE ( "   " ${DENDRO_LIBS} )
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
            SET ( MOAB_INCLUDE ${MOAB_DIRECTORY}/include )
            SET ( IMESH_INCLUDE ${MOAB_DIRECTORY}/lib )
            # Find the MOAB libaries
            SET ( MOAB_PATH_LIB ${MOAB_DIRECTORY}/lib )
            VERIFY_PATH ( ${MOAB_PATH_LIB} )
            FIND_LIBRARY ( MOAB_MESH_LIB     NAMES MOAB      PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_iMESH_LIB    NAMES iMesh     PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_COUPLER_LIB  NAMES mbcoupler PATHS ${MOAB_PATH_LIB}          NO_DEFAULT_PATH )
            IF ( NOT MOAB_MESH_LIB )
                MESSAGE ( FATAL_ERROR "MOAB library (MOAB) not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_iMESH_LIB )
                MESSAGE ( FATAL_ERROR "iMesh library ${MOAB_iMESH_LIB}  not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_COUPLER_LIB )
                MESSAGE ( FATAL_ERROR "MBCoupler library ${MOAB_COUPLER_LIB}  not found in ${MOAB_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for MOAB is not supported.  Use -D MOAB_DIRECTORY=" )
        ENDIF()
        # Check if we specified the cgm directory
        IF ( CGM_DIRECTORY )
            VERIFY_PATH ( ${CGM_DIRECTORY} )
            # Include the CGM directories
            SET ( MOAB_INCLUDE ${MOAB_INCLUDE} ${CGM_DIRECTORY}/include )
            # Find the CGM libaries
            SET ( CGM_PATH_LIB ${CGM_DIRECTORY}/lib )
            VERIFY_PATH ( ${CGM_PATH_LIB} )
            FIND_LIBRARY ( MOAB_CGM_LIB     NAMES cgm      PATHS ${CGM_PATH_LIB}        NO_DEFAULT_PATH )
            FIND_LIBRARY ( MOAB_iGEOM_LIB   NAMES iGeom    PATHS ${CGM_PATH_LIB}        NO_DEFAULT_PATH )
            IF ( NOT MOAB_CGM_LIB )
                MESSAGE ( FATAL_ERROR "CGM library ${MOAB_CGM_LIB}  not found in ${CGM_PATH_LIB}" )
            ENDIF ()
            IF ( NOT MOAB_iGEOM_LIB )
                MESSAGE ( FATAL_ERROR "iGEOM library ${MOAB_iGEOM_LIB}  not found in ${CGM_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for cgm is not supported.  Use -D CGM_DIRECTORY=" )
        ENDIF()
        # Check if we specified the Cubit directory
        IF ( CUBIT_DIRECTORY )
            VERIFY_PATH ( ${CUBIT_DIRECTORY} )
            # Include the CUBIT directories
            # SET ( MOAB_INCLUDE ${MOAB_INCLUDE} ${CUBIT_DIRECTORY}/include )
            # Find the CGM libaries
            SET ( CUBIT_PATH_LIB ${CUBIT_DIRECTORY} )
            VERIFY_PATH ( ${CGM_PATH_LIB} )
            FIND_LIBRARY ( MOAB_CUBIT_LIB     NAMES cubiti19      PATHS ${CUBIT_PATH_LIB}        NO_DEFAULT_PATH )
            IF ( NOT MOAB_CUBIT_LIB )
                MESSAGE ( FATAL_ERROR "CUBIT librarys not found in ${CUBIT_PATH_LIB}" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for cubit is not supported.  Use -D CUBIT_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        INCLUDE_DIRECTORIES ( ${MOAB_INCLUDE} )
        SET ( MOAB_LIBS
            ${MOAB_COUPLER_LIB}
            ${MOAB_iMESH_LIB}
            ${MOAB_MESH_LIB}
            ${MOAB_CGM_LIB}
            ${MOAB_iGEOM_LIB}
            ${MOAB_CUBIT_LIB}
        )
        ADD_DEFINITIONS ( "-D USE_EXT_MOAB" )  
        MESSAGE ( "Using MOAB" )
        MESSAGE ( "   " ${MOAB_LIBS} )
    ENDIF()
ENDMACRO ()


# Macro to configure the BLAS
MACRO ( CONFIGURE_BLAS )
    # Determine if we want to use BLAS
    CHECK_ENABLE_FLAG(USE_EXT_BLAS 1 )
    IF ( USE_EXT_BLAS )
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
        MESSAGE ( "   " ${BLAS_LIBS} )
    ENDIF()
ENDMACRO ()


# Macro to configure the LAPACK
MACRO ( CONFIGURE_LAPACK )
    # Determine if we want to use LAPACK
    CHECK_ENABLE_FLAG(USE_EXT_LAPACK 1 )
    IF ( USE_EXT_LAPACK )
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
        MESSAGE ( "   " ${LAPACK_LIBS} )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the sundials libraries
MACRO ( CONFIGURE_SUNDIALS_LIBRARIES )
    # Determine if we want to use sundials
    CHECK_ENABLE_FLAG(USE_EXT_SUNDIALS 1 )
    IF ( USE_EXT_SUNDIALS )
        # Check if we specified the sundials directory
        IF ( SUNDIALS_DIRECTORY )
            VERIFY_PATH ( ${SUNDIALS_DIRECTORY} )
            INCLUDE_DIRECTORIES ( ${SUNDIALS_DIRECTORY}/include )
            SET ( SUNDIALS_INCLUDE ${SUNDIALS_DIRECTORY}/include )
            FIND_LIBRARY ( SUNDIALS_CVODE_LIB        NAMES  sundials_cvode        PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( SUNDIALS_IDA_LIB          NAMES  sundials_ida          PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( SUNDIALS_IDAS_LIB         NAMES  sundials_idas         PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( SUNDIALS_KINSOL_LIB       NAMES  sundials_kinsol       PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            FIND_LIBRARY ( SUNDIALS_NVECSERIAL_LIB   NAMES  sundials_nvecserial   PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            IF ( USE_EXT_MPI )
                FIND_LIBRARY ( SUNDIALS_NVECPARALLEL_LIB NAMES  sundials_nvecparallel PATHS ${SUNDIALS_DIRECTORY}/lib  NO_DEFAULT_PATH )
            ENDIF()
            IF ( (NOT SUNDIALS_CVODE_LIB) OR (NOT SUNDIALS_IDA_LIB) OR (NOT SUNDIALS_IDAS_LIB) OR 
                 (NOT SUNDIALS_KINSOL_LIB) OR (NOT SUNDIALS_NVECSERIAL_LIB) )
                MESSAGE ( ${SUNDIALS_CVODE_LIB} )
                MESSAGE ( ${SUNDIALS_IDA_LIB} )
                MESSAGE ( ${SUNDIALS_IDAS_LIB} )
                MESSAGE ( ${SUNDIALS_KINSOL_LIB} )
                MESSAGE ( ${SUNDIALS_NVECSERIAL_LIB} )
                MESSAGE ( FATAL_ERROR "Sundials libraries not found in ${SUNDIALS_DIRECTORY}/lib" )
            ENDIF ()
            IF ( USE_EXT_MPI AND (NOT SUNDIALS_NVECPARALLEL_LIB) )
                MESSAGE ( ${SUNDIALS_NVECPARALLEL_LIB} )
                MESSAGE ( FATAL_ERROR "Sundials libraries not found in ${SUNDIALS_DIRECTORY}/lib" )
            ENDIF ()
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for sundials is not yet supported.  Use -D SUNDIALS_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        SET ( SUNDIALS_LIBS
            ${SUNDIALS_CVODE_LIB}
            ${SUNDIALS_IDA_LIB}
            ${SUNDIALS_IDAS_LIB}
            ${SUNDIALS_KINSOL_LIB}
            ${SUNDIALS_NVECPARALLEL_LIB}
        )
        IF ( USE_EXT_MPI )
            SET ( SUNDIALS_LIBS  ${SUNDIALS_LIBS}  ${SUNDIALS_NVEC_PARALLEL_LIB} )
        ENDIF()
        ADD_DEFINITIONS ( "-D USE_EXT_SUNDIALS" )  
        MESSAGE ( "Using sundials" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the hypre libraries
MACRO ( CONFIGURE_HYPRE_LIBRARIES )
    # Determine if we want to use silo
    CHECK_ENABLE_FLAG( USE_EXT_HYPRE 1 )
    IF ( USE_EXT_HYPRE )
        # Check if we specified the hypre directory
        IF ( HYPRE_DIRECTORY )
            VERIFY_PATH ( ${HYPRE_DIRECTORY} )
            SET ( HYPRE_LIB_DIRECTORY ${HYPRE_DIRECTORY}/lib )
            FIND_LIBRARY ( HYPRE_LIB         NAMES HYPRE                PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_DM_LIB      NAMES HYPRE_DistributedMatrix  PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_DMPS_LIB    NAMES HYPRE_DistributedMatrixPilutSolver  PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            # FIND_LIBRARY ( HYPRE_EUCLID_LIB  NAMES HYPRE_Euclid  PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_IJMV_LIB    NAMES HYPRE_IJ_mv          PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_KRYLOV_LIB  NAMES HYPRE_krylov         PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            # FIND_LIBRARY ( HYPRE_LSI_LIB     NAMES HYPRE_LSI  PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_MATMAT_LIB  NAMES HYPRE_MatrixMatrix   PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_MULTIV_LIB  NAMES HYPRE_multivector    PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_PARAS_LIB   NAMES HYPRE_ParaSails      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_PBMV_LIB    NAMES HYPRE_parcsr_block_mv  PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_PLS_LIB     NAMES HYPRE_parcsr_ls      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_PMV_LIB     NAMES HYPRE_parcsr_mv      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_SEQMV_LIB   NAMES HYPRE_seq_mv         PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_SSLS_LIB    NAMES HYPRE_sstruct_ls     PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_SSMV_LIB    NAMES HYPRE_sstruct_mv     PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_SLS_LIB     NAMES HYPRE_struct_ls      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_SMV_LIB     NAMES HYPRE_struct_mv      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
            FIND_LIBRARY ( HYPRE_UTIL_LIB    NAMES HYPRE_utilities      PATHS ${HYPRE_LIB_DIRECTORY}  NO_DEFAULT_PATH )
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for hypre is not yet supported.  Use -D HYPRE_DIRECTORY=" )
        ENDIF()
        # Add the libraries in the appropriate order
        SET ( HYPRE_LIBS
            ${HYPRE_DM_LIB}
            ${HYPRE_DMPS_LIB}
            # ${HYPRE_EUCLID_LIB}
            ${HYPRE_IJMV_LIB}
            ${HYPRE_KRYLOV_LIB}
            # ${HYPRE_LSI_LIB}
            ${HYPRE_MATMAT_LIB}
            ${HYPRE_MULTIV_LIB}
            ${HYPRE_PARAS_LIB}
            ${HYPRE_PBMV_LIB}
            ${HYPRE_PLS_LIB}
            ${HYPRE_PMV_LIB}
            ${HYPRE_SEQMV_LIB}
            ${HYPRE_SSLS_LIB}
            ${HYPRE_SSMV_LIB}
            ${HYPRE_SLS_LIB}
            ${HYPRE_SMV_LIB}
            ${HYPRE_UTIL_LIB}
            ${HYPRE_LIB}
        )
        ADD_DEFINITIONS ( "-D USE_EXT_HYPRE" )  
        MESSAGE ( "Using hypre" )
    ENDIF()
ENDMACRO ()


# Macro to find and configure the petsc libraries
MACRO ( CONFIGURE_PETSC_LIBRARIES )
    # Determine if we want to use petsc
    CHECK_ENABLE_FLAG(USE_EXT_PETSC 1 )
    IF ( USE_EXT_PETSC )
        # Check if we specified the petsc directory
        IF ( PETSC_DIRECTORY )
            VERIFY_PATH ( ${PETSC_DIRECTORY} )
            VERIFY_VARIABLE ( "PETSC_ARCH" )
        ELSE()
            MESSAGE ( FATAL_ERROR "Default search for petsc is not yet supported.  Use -D PETSC_DIRECTORY=" )
        ENDIF()
        # Get the petsc version
        PETSC_GET_VERSION()
        MESSAGE("Found PETSc version ${PETSC_VERSION}")
        # Add the petsc include folders and definitiosn
        SET ( PETSC_INCLUDE ${PETSC_INCLUDE} ${PETSC_DIRECTORY}/include )
        SET ( PETSC_INCLUDE ${PETSC_INCLUDE} ${PETSC_DIRECTORY}/${PETSC_ARCH}/include )
        INCLUDE_DIRECTORIES ( ${PETSC_INCLUDE} )
        SET ( PETSC_LIB_DIRECTORY ${PETSC_DIRECTORY}/${PETSC_ARCH}/lib )
        ADD_DEFINITIONS ( "-D USE_EXT_PETSC" )  
        # Find the petsc libraries
        PETSC_SET_LIBRARIES()
        MESSAGE ( "Using petsc" )
        MESSAGE ( "   "  ${PETSC_LIBS} )
    ENDIF()
ENDMACRO ()


# Macro to configure system-specific libraries and flags
MACRO ( CONFIGURE_SYSTEM )
    # Remove extra library links
    SET_STATIC_FLAGS()
    # Add system dependent flags
    MESSAGE("System is: ${CMAKE_SYSTEM_NAME}")
    IF ( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
        # Windows specific system libraries
        #FIND_LIBRARY ( SYSTEM_LIBS           NAMES "psapi"        PATHS C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64/  )
        #C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64/psapi
        SET( SYSTEM_LIBS "C:/Program Files (x86)/Microsoft SDKs/Windows/v7.0A/Lib/x64/Psapi.lib" )
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" )
        # Linux specific system libraries
        SET( SYSTEM_LIBS "-lz -ldl" )
        if ( NOT USE_STATIC )
            SET( SYSTEM_LIBS "${SYSTEM_LIBS} -rdynamic" )   # Needed for backtrace to print function names
        ENDIF()
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin" )
        # Max specific system libraries
        SET( SYSTEM_LIBS "-lz -ldl" )
    ELSEIF( ${CMAKE_SYSTEM_NAME} STREQUAL "Generic" )
        # Generic system libraries
    ELSE()
        MESSAGE( FATAL_ERROR "OS not detected" )
    ENDIF()
ENDMACRO ()


# Macro to configure AMP-specific options
MACRO ( CONFIGURE_AMP )
    # Add the AMP install directory
    INCLUDE_DIRECTORIES ( ${AMP_INSTALL_DIR}/include )
    # Set the data directory for AMP (needed to find the meshes)
    IF ( AMP_DATA )
        VERIFY_PATH ( ${AMP_DATA} )
    ELSE()
        MESSAGE ( FATAL_ERROR "AMP_DATA must be set" )
    ENDIF()
    # Set the maximum number of processors for the tests
    IF ( NOT TEST_MAX_PROCS )
        SET( TEST_MAX_PROCS 32 )
    ENDIF()
    # Remove extra library links
    set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
    set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
    set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
    set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
    set(CMAKE_SHARED_LINKER_FLAGS)
    set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
    set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
    # Check if we are using utils
    CHECK_ENABLE_FLAG( USE_AMP_UTILS 1 )
    IF ( NOT USE_AMP_UTILS )
        MESSAGE ( FATAL_ERROR "AMP Utils must be used" )
    ENDIF()
    # Check if we are using ampmesh
    CHECK_ENABLE_FLAG( USE_AMP_MESH 1 )
    IF ( NOT USE_AMP_MESH )
        MESSAGE ( "Disabling AMP Mesh" )
    ENDIF()
    # Check if we are using discretization
    CHECK_ENABLE_FLAG( USE_AMP_DISCRETIZATION 1 )
    IF ( NOT USE_AMP_MESH )
        SET ( USE_AMP_DISCRETIZATION 0 )
    ENDIF()
    IF ( NOT USE_AMP_DISCRETIZATION )
        MESSAGE ( "Disabling AMP Descritization" )
    ENDIF()
    # Check if we are using vectors
    CHECK_ENABLE_FLAG( USE_AMP_VECTORS 1 )
    IF ( NOT USE_AMP_VECTORS )
        MESSAGE ( "Disabling AMP Vectors" )
    ENDIF()
    # Check if we are using matricies
    CHECK_ENABLE_FLAG( USE_AMP_MATRICIES 1 )
    IF ( NOT USE_AMP_MATRICIES )
        MESSAGE ( "Disabling AMP Matricies" )
    ENDIF()
    # Check if we are using materials
    CHECK_ENABLE_FLAG( USE_AMP_MATERIALS 1 )
    IF ( NOT USE_AMP_VECTORS )
        SET ( USE_AMP_MATERIALS 0 )
    ENDIF()
    IF ( NOT USE_AMP_MATRICIES )
        MESSAGE ( "Disabling AMP Materials" )
    ENDIF()
    # Check if we are using operators
    CHECK_ENABLE_FLAG( USE_AMP_OPERATORS 1 )
    IF ( (NOT USE_AMP_MESH) OR (NOT USE_AMP_VECTORS) OR (NOT USE_AMP_MATRICIES) OR (NOT USE_EXT_LIBMESH) )
    #IF ( (NOT USE_AMP_MESH) OR (NOT USE_AMP_VECTORS) OR (NOT USE_AMP_MATRICIES) )
        SET ( USE_AMP_OPERATORS 0 )
    ENDIF()
    IF ( NOT USE_AMP_OPERATORS )
        MESSAGE ( "Disabling AMP Operators" )
    ENDIF()
    # Check if we are using solvers
    CHECK_ENABLE_FLAG( USE_AMP_SOLVERS 1 )
    IF ( (NOT USE_AMP_OPERATORS) )
        SET ( USE_AMP_SOLVERS 0 )
    ENDIF()
    IF ( NOT USE_AMP_SOLVERS )
        MESSAGE ( "Disabling AMP Solvers" )
    ENDIF()
    # Check if we are using time_integrators
    CHECK_ENABLE_FLAG( USE_AMP_TIME_INTEGRATORS 1 )
    IF ( (NOT USE_AMP_SOLVERS) )
        SET ( USE_AMP_TIME_INTEGRATORS 0 )
    ENDIF()
    IF ( NOT USE_AMP_TIME_INTEGRATORS )
        MESSAGE ( "Disabling AMP Time Integrators" )
    ENDIF()
    # Set the AMP libraries (the order matters for some platforms)
    SET ( AMP_LIBS )
    SET ( AMP_DOC_DIRS " ")
    IF ( USE_EXT_NEK )
        SET ( AMP_LIBS ${AMP_LIBS} "nek" )
        ADD_DEFINITIONS ( -D USE_EXT_NEK )  
    ENDIF()
    IF ( USE_AMP_TIME_INTEGRATORS )
        SET ( AMP_LIBS ${AMP_LIBS} "time_integrators" )
        ADD_DEFINITIONS ( -D USE_AMP_TIME_INTEGRATORS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/time_integrators\"" )
    ENDIF()
    IF ( USE_AMP_SOLVERS )
        SET ( AMP_LIBS ${AMP_LIBS} "solvers" )
        ADD_DEFINITIONS ( -D USE_AMP_SOLVERS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/solvers\"" )
    ENDIF()
    IF ( USE_AMP_OPERATORS )
        SET ( AMP_LIBS ${AMP_LIBS} "operators" )
        ADD_DEFINITIONS ( -D USE_AMP_OPERATORS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/operators\"" )
    ENDIF()
    IF ( USE_AMP_MATERIALS )
        SET ( AMP_LIBS ${AMP_LIBS} "materials" )
        ADD_DEFINITIONS ( -D USE_AMP_MATERIALS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/materials\"" )
    ENDIF()
    IF ( USE_AMP_MATRICIES )
        SET ( AMP_LIBS ${AMP_LIBS} "matrices" )
        ADD_DEFINITIONS ( -D USE_AMP_MATRICIES )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/matrices\"" )
    ENDIF()
    IF ( USE_AMP_VECTORS )
        SET ( AMP_LIBS ${AMP_LIBS} "vectors" )
        ADD_DEFINITIONS ( -D USE_AMP_VECTORS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/vectors\"" )
    ENDIF()
    IF ( USE_AMP_DISCRETIZATION )
        SET ( AMP_LIBS ${AMP_LIBS} "discretization" )
        ADD_DEFINITIONS ( -D USE_AMP_DISCRETIZATION )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/discretization\"" )
    ENDIF()
    IF ( USE_AMP_MESH )
        SET ( AMP_LIBS ${AMP_LIBS} "ampmesh" )
        ADD_DEFINITIONS ( -D USE_AMP_MESH )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/ampmesh\"" )
    ENDIF()
    IF ( USE_AMP_UTILS )
        SET ( AMP_LIBS ${AMP_LIBS} "utils" )
        ADD_DEFINITIONS ( -D USE_AMP_UTILS )  
        SET ( AMP_DOC_DIRS "${AMP_DOC_DIRS}  \"${AMP_SOURCE_DIR}/src/utils\"" )
    ENDIF()
    # Repeat all amp libraries to take care of circular dependencies 
    # Example: Mesh depends on Vectors which depends on Discretization which depends on Mesh
    SET ( AMP_LIBS ${AMP_LIBS} ${AMP_LIBS} )
ENDMACRO ()




