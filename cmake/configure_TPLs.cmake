# Macro to configure the TPL builder
FUNCTION ( CONFIGURE_TPL_BUILDER )
    # Set the build type
    SET( TPL_CMAKE "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}" )
    # Set the compiler / compile flags
    IF ( C_COMPILER )
        LIST(APPEND TPL_CMAKE "-DC_COMPILER=${C_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DCFLAGS=${CFLAGS}")
    ELSE()
        LIST(APPEND TPL_CMAKE "-DC_COMPILER=${CMAKE_C_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DCFLAGS=${CMAKE_C_FLAGS}")
    ENDIF()
    IF ( CXX_COMPILER )
        LIST(APPEND TPL_CMAKE "-DCXX_COMPILER=${CXX_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DCXXFLAGS=${CXXFLAGS}")
    ELSE()
        LIST(APPEND TPL_CMAKE "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DCXXFLAGS=${CMAKE_CXX_FLAGS}")
    ENDIF()
    IF ( Fortran_COMPILER )
        LIST(APPEND TPL_CMAKE "-DFortran_COMPILER=${Fortran_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DFFLAGS=${FFLAGS}")
    ELSE()
        LIST(APPEND TPL_CMAKE "-DFortran_COMPILER=${CMAKE_Fortran_COMPILER}")
        LIST(APPEND TPL_CMAKE "-DFFLAGS=${CMAKE_Fortran_FLAGS}")
    ENDIF()
    IF( ENABLE_SHARED )
        LIST(APPEND TPL_CMAKE "-DENABLE_STATIC:BOOL=OFF")
        LIST(APPEND TPL_CMAKE "-DENABLE_SHARED:BOOL=ON")
    ELSE()
        LIST(APPEND TPL_CMAKE "-DENABLE_STATIC:BOOL=ON")
        LIST(APPEND TPL_CMAKE "-DENABLE_SHARED:BOOL=OFF")
    ENDIF()
    LIST(APPEND TPL_CMAKE "-DCXX_STD=${CXX_STD}")
    LIST(APPEND TPL_CMAKE "-DLDFLAGS=${LDFLAGS}")
    # Set the install path
    LIST(APPEND TPL_CMAKE "-DINSTALL_DIR:PATH=${${PROJ}_INSTALL_DIR}/TPLs")
    LIST(APPEND TPL_CMAKE "-DDISABLE_ALL_TESTS:BOOL=ON")
    # Set all flags that start with TPL_
    GET_CMAKE_PROPERTY( variableNames VARIABLES )
    FOREACH ( var ${variableNames} )
        STRING( REGEX REPLACE "^TPL_" "" var2 "${var}" )
        IF ( ${var} STREQUAL TPL_CMAKE OR ${var} STREQUAL TPL_URL OR ${var} STREQUAL TPL_SOURCE_DIR)
            # Special variable
        ELSEIF ( ${var} STREQUAL TPL_LIST )
            STRING( REPLACE ";" "," TPL_LIST2 "${TPL_LIST}" )
            LIST(APPEND TPL_CMAKE "-DTPL_LIST:STRING=${TPL_LIST2}")
        ELSEIF ( ${var} STREQUAL TPL_${var2} )
            string(REPLACE ";" "\\;" var_value "${${var}}")
            LIST(APPEND TPL_CMAKE "-D${var2}=${var_value}")
        ENDIF()
    ENDFOREACH()
    IF(NOT DEFINED TPL_SOURCE_DIR)
        IF(EXISTS "${CMAKE_SOURCE_DIR}/tpl-builder")
            set(TPL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/tpl-builder")
        ELSE()
            FIND_PACKAGE(Git)
            # Download the TPL builder
            set(TPL_SOURCE_DIR ${CMAKE_BINARY_DIR}/tpl-builder)
            MESSAGE( STATUS "Downloading TPL builder" )
            IF(NOT EXISTS "${TPL_SOURCE_DIR}")
                EXECUTE_PROCESS(
                    COMMAND ${GIT_EXECUTABLE} clone "${TPL_URL}"
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                )
            ELSEIF(EXISTS "${TPL_SOURCE_DIR}/.git")
                EXECUTE_PROCESS(
                    COMMAND ${GIT_EXECUTABLE} pull
                    WORKING_DIRECTORY "${TPL_SOURCE_DIR}"
                )
            ENDIF()
      ENDIF()
    ENDIF()
    # Configure the TPL builder
    FILE( MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/tpl-build" )
    MESSAGE( STATUS "Configuring TPL builder" )
    EXECUTE_PROCESS(
        COMMAND ${CMAKE_COMMAND} ${TPL_CMAKE} ${TPL_SOURCE_DIR}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tpl-build"
        COMMAND_ECHO STDOUT
    )
    # Build the TPLs
    MESSAGE( STATUS "Building TPLs" )
    EXECUTE_PROCESS(
        COMMAND ${CMAKE_MAKE_PROGRAM}
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/tpl-build"
    )
    SET( TPL_DIRECTORY "${${PROJ}_INSTALL_DIR}/TPLs" PARENT_SCOPE )
ENDFUNCTION()
