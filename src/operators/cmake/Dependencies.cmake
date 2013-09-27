SET(LIB_REQUIRED_DEP_PACKAGES AMP_UTILITIES AMP_MESH AMP_DISCRETIZATION AMP_VECTORS AMP_MATRICES AMP_MATERIALS)
SET(LIB_OPTIONAL_DEP_PACKAGES)
SET(TEST_REQUIRED_DEP_PACKAGES)
SET(TEST_OPTIONAL_DEP_PACKAGES)
SET(LIB_REQUIRED_DEP_TPLS BOOST LIBMESH)
SET(LIB_OPTIONAL_DEP_TPLS MPI PETSC_AMP X11 DENDRO)
SET(TEST_REQUIRED_DEP_TPLS PETSC_AMP) 
SET(TEST_OPTIONAL_DEP_TPLS)

# Add trilinos package dependencies
IF ( Trilinos_PACKAGES_AND_DIRS_AND_CLASSIFICATIONS )
    SET(LIB_REQUIRED_DEP_PACKAGES ${LIB_REQUIRED_DEP_PACKAGES} Epetra Thyra)
ENDIF()


