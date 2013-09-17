# Macro to set AMP package flags and definitions
# Note: order matters, ordered from least to most dependencies
MACRO ( SET_AMP_PACKAGE_FLAGS )
    
    # Set the AMP libraries and definitions
    SET( AMP_LIBS )
    IF ( ${PROJECT_NAME}_ENABLE_AMP_UTILITIES )
        SET( USE_AMP_UTILS ON )
        ADD_DEFINITIONS ( -D USE_AMP_UTILS )  
        SET( AMP_LIBS utils ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_MESH )
        SET( USE_AMP_MESH ON )
        ADD_DEFINITIONS ( -D USE_AMP_MESH )  
        SET( AMP_LIBS ampmesh ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_DISCRETIZATION )
        SET( USE_AMP_DISCRETIZATION ON )
        ADD_DEFINITIONS ( -D USE_AMP_DISCRETIZATION )  
        SET( AMP_LIBS discretization ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_VECTORS )
        SET( USE_AMP_VECTORS ON )
        ADD_DEFINITIONS ( -D USE_AMP_VECTORS )  
        SET( AMP_LIBS vectors ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_MATRICES )
        SET( USE_AMP_MATRICES ON )
        ADD_DEFINITIONS ( -D USE_AMP_MATRICES )  
        SET( AMP_LIBS matrices ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_MATERIALS )
        SET( USE_AMP_MATERIALS ON )
        ADD_DEFINITIONS ( -D USE_AMP_MATERIALS )  
        SET( AMP_LIBS materials ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_OPERATORS )
        SET( USE_AMP_OPERATORS ON )
        ADD_DEFINITIONS ( -D USE_AMP_OPERATORS )  
        SET( AMP_LIBS operators ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_TIME_INTEGRATORS )
        SET( USE_AMP_TIME_INTEGRATORS ON )
        ADD_DEFINITIONS ( -D USE_AMP_TIME_INTEGRATORS )  
        SET( AMP_LIBS time_integrators ${AMP_LIBS} )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_AMP_SOLVERS )
        SET( USE_AMP_SOLVERS ON )
        ADD_DEFINITIONS ( -D USE_AMP_SOLVERS )  
        SET( AMP_LIBS solvers ${AMP_LIBS} )
    ENDIF()
    INCLUDE_DIRECTORIES( ${AMP_INSTALL_DIR}/include )
    #SET( ${PROJECT_NAME}_INCLUDE_DIRS  ${AMP_INSTALL_DIR}/include ${${PROJECT_NAME}_INCLUDE_DIRS} )
    #SET( ${PROJECT_NAME}_LIBRARIES ${AMP_LIBS} ${AMP_LIBS} ${${PROJECT_NAME}_LIBRARIES} )
    INCLUDE_DIRECTORIES( ${${PROJECT_NAME}_INCLUDE_DIRS} )

    # Add the Trilinos info
    IF ( ${PROJECT_NAME}_ENABLE_Epetra )
        SET( USE_TRILINOS_VECTORS 1 )
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_Teuchos )
        SET( USE_TRILINOS_TEUCHOS 1 )
        SET( USE_TRILINOS_UTILS 1 )
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_Thyra )
        SET( USE_TRILINOS_THYRA 1 )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_THYRA" )
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_Nox OR ${PROJECT_NAME}_ENABLE_NOX )
        SET( USE_TRILINOS_NOX 1 )
        SET( USE_TRILINOS_SOLVERS 1 )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_NOX" )
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_Stratimikos )
        SET( USE_TRILINOS_STRATIMIKOS 1 )
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( ${PROJECT_NAME}_ENABLE_STK )
        SET( USE_TRILINOS_STKMESH 1 )
        ADD_DEFINITIONS ( "-D USE_TRILINOS_STKMESH" )  
        SET( USE_EXT_TRILINOS 1 )
    ENDIF()
    IF ( USE_EXT_TRILINOS )
        ADD_DEFINITIONS ( "-D USE_EXT_TRILINOS" )  
    ENDIF()

    # Add libmesh info
    IF ( USE_EXT_LIBMESH )
        ADD_DEFINITIONS ( -DLIBMESH_ENABLE_PARMESH )
        ADD_DEFINITIONS( "-D USE_EXT_LIBMESH" )
        INCLUDE_DIRECTORIES( ${TPL_LIBMESH_INCLUDE_DIRS} )
    ENDIF()

    # Add dendro info
    IF ( USE_EXT_DENDRO )
        ADD_DEFINITIONS( "-D USE_EXT_DENDRO" )
        INCLUDE_DIRECTORIES( ${TPL_DENDRO_INCLUDE_DIRS} )
    ENDIF()

    # Add Petsc info
    IF ( USE_EXT_PETSC )
        ADD_DEFINITIONS( "-D USE_EXT_PETSC" )
        INCLUDE_DIRECTORIES( ${TPL_PETSC_AMP_INCLUDE_DIRS} )
    ENDIF()

    # Add sundials info
    IF ( USE_EXT_SUNDIALS )
        ADD_DEFINITIONS( "-D USE_EXT_SUNDIALS" )
        INCLUDE_DIRECTORIES( ${TPL_SUNDIALS_INCLUDE_DIRS} )
    ENDIF()

    # Add silo info
    IF ( USE_EXT_SILO )
        ADD_DEFINITIONS( "-D USE_EXT_SILO" )
        INCLUDE_DIRECTORIES( ${TPL_SILO_INCLUDE_DIRS} )
    ENDIF()

    # Add hdf5 info
    IF ( USE_EXT_HDF5 )
        ADD_DEFINITIONS( "-D USE_EXT_HDF5" )
        INCLUDE_DIRECTORIES( ${TPL_HDF5_INCLUDE_DIRS} )
    ENDIF()

    # Add X11 info
    IF ( USE_EXT_X11 )
        ADD_DEFINITIONS( "-D USE_EXT_X11" )
        INCLUDE_DIRECTORIES( ${TPL_X11_INCLUDE_DIRS} )
    ENDIF()

    # Add BLAS/LAPACK info
    IF ( USE_EXT_LAPACK )
        ADD_DEFINITIONS( "-D USE_EXT_LAPACK" )
        INCLUDE_DIRECTORIES( ${TPL_LAPACK_INCLUDE_DIRS} )
    ENDIF()
    IF ( USE_EXT_BLAS )
        ADD_DEFINITIONS( "-D USE_EXT_BLAS" )
        INCLUDE_DIRECTORIES( ${TPL_BLAS_INCLUDE_DIRS} )
    ENDIF()
    
    # Add boost info
    IF ( USE_EXT_BOOST )
        ADD_DEFINITIONS( "-D USE_EXT_BOOST" )
        INCLUDE_DIRECTORIES( ${TPL_BOOST_INCLUDE_DIRS} )
    ENDIF()

    # Add MPI info
    IF ( USE_EXT_MPI )
        ADD_DEFINITIONS( "-D USE_EXT_MPI" )
        INCLUDE_DIRECTORIES( ${TPL_MPI_INCLUDE_DIRS} )
    ENDIF()

ENDMACRO()


