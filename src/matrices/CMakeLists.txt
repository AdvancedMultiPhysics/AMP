BEGIN_PACKAGE_CONFIG( matrices )

# Add the subdirectories
ADD_PACKAGE_SUBDIRECTORY( testHelpers )
ADD_PACKAGE_SUBDIRECTORY( data )
IF( USE_DEVICE )
    SET_DEVICE_COMPILER( data/CSRMatrixDataHelper.cpp )
ENDIF()
IF ( USE_HYPRE )
    ADD_PACKAGE_SUBDIRECTORY( data/hypre )
ENDIF()
IF ( USE_TRILINOS )
    ADD_PACKAGE_SUBDIRECTORY( trilinos )
ENDIF()
IF ( USE_PETSC )
    ADD_PACKAGE_SUBDIRECTORY( petsc )
ENDIF()

SET_DEVICE_COMPILER ( data/CSRMatrixDataHelpers.cpp )

ADD_PACKAGE_SUBDIRECTORY( operations )
ADD_PACKAGE_SUBDIRECTORY( operations/default )
ADD_PACKAGE_SUBDIRECTORY( operations/default/spgemm )
IF( USE_KOKKOS )
    ADD_PACKAGE_SUBDIRECTORY( operations/kokkos )
    IF( USE_DEVICE )
       SET_DEVICE_COMPILER( operations/kokkos/CSRMatrixOperationsKokkos.cpp )
    ENDIF()
ENDIF()
IF( USE_DEVICE )
    ADD_PACKAGE_SUBDIRECTORY( operations/device )
    ADD_PACKAGE_SUBDIRECTORY( operations/device/spgemm )
    SET_DEVICE_COMPILER( operations/device )
    SET_DEVICE_COMPILER( CSRMatrix.cpp )
    IF ( USE_CUDA )
        ADD_PACKAGE_SUBDIRECTORY( operations/device/spgemm/cuda )
        SET_DEVICE_COMPILER( operations/device/spgemm/cuda )
    ENDIF()
    IF ( USE_HIP )
        ADD_PACKAGE_SUBDIRECTORY( operations/device/spgemm/hip )
        SET_DEVICE_COMPILER( operations/device/spgemm/hip/SpGEMM_Hip.cpp )
    ENDIF()
ENDIF()

# Set the default matrix type
ADD_COMPILE_DEFINITIONS( DEFAULT_MATRIX="${DEFAULT_MATRIX}" )

# Install the matrices target
INSTALL_AMP_TARGET( matrices )

# Add the tests
ADD_SUBDIRECTORY( test )

