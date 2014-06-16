INCLUDE(TribitsTplDeclareLibraries)

VERIFY_PATH( ${HDF5_INCLUDE_DIRS} )
VERIFY_PATH( ${HDF5_LIBRARY_DIRS} )
TRIBITS_TPL_DECLARE_LIBRARIES( HDF5
    REQUIRED_HEADERS hdf5.h
    REQUIRED_LIBS_NAMES hdf5_fortran hdf5hl_fortran hdf5 hdf5_hl z
)

IF(TPL_ENABLE_MPI)
    ADD_DEFINITIONS( -DH5_HAVE_PARALLEL )
ENDIF()

# Add the definitions
SET( HDF5_LIBS ${TPL_HDF5_LIBRARIES} )
SET( TPL_ENABLE_HDF5 ON )
SET( USE_EXT_HDF5 1 )

MESSAGE ( "Using HDF5" )
MESSAGE ( "   "  ${HDF5_LIBS} )
