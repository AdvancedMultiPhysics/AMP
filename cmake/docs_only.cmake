# Check if a flag is enabled
MACRO( CHECK_ENABLE_FLAG FLAG DEFAULT )
    IF ( NOT DEFINED ${FLAG} )
        SET( ${FLAG} ${DEFAULT} )
    ELSEIF ( ${FLAG} STREQUAL "" )
        SET( ${FLAG} ${DEFAULT} )
    ELSEIF ( (${${FLAG}} STREQUAL "FALSE" ) OR ( ${${FLAG}} STREQUAL "false" ) OR ( ${${FLAG}} STREQUAL "0" ) OR ( ${${FLAG}} STREQUAL "OFF" ))
        SET( ${FLAG} 0 )
    ELSEIF ( (${${FLAG}} STREQUAL "TRUE" ) OR ( ${${FLAG}} STREQUAL "true" ) OR ( ${${FLAG}} STREQUAL "1" ) OR ( ${${FLAG}} STREQUAL "ON" ))
        SET( ${FLAG} 1 )
    ELSE ()
        MESSAGE( "Bad value for ${FLAG} ( ${${FLAG}} ); use true or false" )
    ENDIF()
ENDMACRO()


# Dummy functions
FUNCTION( WRITE_REPO_VERSION )
ENDFUNCTION()
FUNCTION( CREATE_RELEASE )
ENDFUNCTION()

