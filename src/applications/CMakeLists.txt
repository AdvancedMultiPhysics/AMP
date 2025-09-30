BEGIN_PACKAGE_CONFIG( applications )
ADD_PACKAGE_SUBDIRECTORY( thermal )
INSTALL_AMP_TARGET( applications )

ADD_SUBDIRECTORY( thermal/test )
ADD_SUBDIRECTORY( radiationDiffusionFD )
