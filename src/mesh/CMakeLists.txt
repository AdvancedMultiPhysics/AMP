BEGIN_PACKAGE_CONFIG( mesh )
ADD_PACKAGE_SUBDIRECTORY( loadBalance )
ADD_PACKAGE_SUBDIRECTORY( structured )
ADD_PACKAGE_SUBDIRECTORY( triangle )
ADD_PACKAGE_SUBDIRECTORY( testHelpers )
IF ( USE_LIBMESH )
    ADD_PACKAGE_SUBDIRECTORY( libmesh )
ENDIF()
IF ( USE_TRILINOS_STKCLASSIC )
    # stk interface is broken due to extensive changes in Trilinos ADD_PACKAGE_SUBDIRECTORY( STKmesh )
ENDIF()
INSTALL_AMP_TARGET( mesh )

# Add the unit tests
ADD_SUBDIRECTORY( test )

# Add the examples
ADD_SUBDIRECTORY( examples )
