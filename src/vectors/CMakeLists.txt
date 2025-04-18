BEGIN_PACKAGE_CONFIG( vectors )
ADD_PACKAGE_SUBDIRECTORY( testHelpers )
ADD_PACKAGE_SUBDIRECTORY( operations )
ADD_PACKAGE_SUBDIRECTORY( data )
ADD_PACKAGE_SUBDIRECTORY( operations/default )
IF ( USE_OPENMP )
    ADD_PACKAGE_SUBDIRECTORY( operations/OpenMP )
ENDIF()
IF ( USE_DEVICE )
    ADD_PACKAGE_SUBDIRECTORY( operations/device )
    SET_DEVICE_COMPILER( operations/device )
ENDIF()

IF ( USE_TRILINOS )
    ADD_PACKAGE_SUBDIRECTORY( trilinos )
    ADD_PACKAGE_SUBDIRECTORY( testHelpers/trilinos )
    IF ( USE_TRILINOS_THYRA )
        ADD_PACKAGE_SUBDIRECTORY( trilinos/thyra )
        ADD_PACKAGE_SUBDIRECTORY( testHelpers/trilinos/thyra )
    ENDIF()
    IF ( USE_TRILINOS_EPETRA )
        ADD_PACKAGE_SUBDIRECTORY( trilinos/epetra )
        ADD_PACKAGE_SUBDIRECTORY( testHelpers/trilinos/epetra )
    ENDIF()
    IF ( USE_TRILINOS_TPETRA )
        ADD_PACKAGE_SUBDIRECTORY( trilinos/tpetra )
    ENDIF()
ENDIF()
IF ( USE_PETSC )
    ADD_PACKAGE_SUBDIRECTORY( petsc )
    ADD_PACKAGE_SUBDIRECTORY( testHelpers/petsc )
ENDIF()
IF ( USE_SUNDIALS )
    ADD_PACKAGE_SUBDIRECTORY( sundials )
    ADD_PACKAGE_SUBDIRECTORY( testHelpers/sundials )
ENDIF()
ADD_PACKAGE_SUBDIRECTORY( newFrozenVectorDesign )
INSTALL_AMP_TARGET( vectors )

IF ( USE_TRILINOS )
    SET_DEVICE_COMPILER( VectorBuilderExternal.cpp trilinos testHelpers/trilinos )
    IF ( USE_TRILINOS_THYRA )
        SET_DEVICE_COMPILER( trilinos/thyra testHelpers/trilinos/thyra )
    ENDIF()
    IF ( USE_TRILINOS_EPETRA )
        SET_DEVICE_COMPILER( trilinos/epetra testHelpers/trilinos/epetra )
    ENDIF()
    IF ( USE_TRILINOS_TPETRA )
        SET_DEVICE_COMPILER( trilinos/tpetra/TpetraVectorData.cpp )
        SET_DEVICE_COMPILER( trilinos/tpetra/TpetraVectorOperations.cpp )
        SET_DEVICE_COMPILER( trilinos/tpetra/TpetraInstantiations.cpp )
    ENDIF()
ENDIF()

ADD_SUBDIRECTORY ( test )

