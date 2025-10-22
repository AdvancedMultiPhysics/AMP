BEGIN_PACKAGE_CONFIG( applications )
ADD_PACKAGE_SUBDIRECTORY( radiationDiffusionFD )
ADD_PACKAGE_SUBDIRECTORY( thermal )
INSTALL_AMP_TARGET( applications )

ADD_SUBDIRECTORY( radiationDiffusionFD )
ADD_SUBDIRECTORY( thermal/test )
