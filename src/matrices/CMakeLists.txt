BEGIN_PACKAGE_CONFIG( matrices )

# Add the subdirectories
ADD_PACKAGE_SUBDIRECTORY( testHelpers )
ADD_PACKAGE_SUBDIRECTORY( data )
IF ( USE_HYPRE )
    ADD_PACKAGE_SUBDIRECTORY( data/hypre )
ENDIF()
IF ( USE_TRILINOS )
    ADD_PACKAGE_SUBDIRECTORY( trilinos )
ENDIF()
IF ( USE_PETSC )
    ADD_PACKAGE_SUBDIRECTORY( petsc )
ENDIF()

IF( USE_CUDA )
   ADD_PACKAGE_SUBDIRECTORY( data/cuda )
   SET_DEVICE_COMPILER ( data/cuda )
ENDIF()
IF( USE_HIP )
    ADD_PACKAGE_SUBDIRECTORY( data/hip )
   SET_DEVICE_COMPILER ( data/hip )
ENDIF()

ADD_PACKAGE_SUBDIRECTORY( operations )
ADD_PACKAGE_SUBDIRECTORY( operations/default )
ADD_PACKAGE_SUBDIRECTORY( operations/default/spgemm )
IF( USE_KOKKOS OR USE_TRILINOS_KOKKOS )
    ADD_PACKAGE_SUBDIRECTORY( operations/kokkos )
    SET_DEVICE_COMPILER( operations/kokkos/CSRMatrixOperationsKokkos.cpp )
ENDIF()
IF( USE_DEVICE )
    ADD_PACKAGE_SUBDIRECTORY( operations/device )
    SET_DEVICE_COMPILER( operations/device )
ENDIF()

SET_DEVICE_COMPILER( CSRMatrix.cpp )

# Set the default matrix type
ADD_COMPILE_DEFINITIONS( DEFAULT_MATRIX="${DEFAULT_MATRIX}" )

# Install the matrices target
INSTALL_AMP_TARGET( matrices )

# Add the tests
ADD_SUBDIRECTORY( test )

