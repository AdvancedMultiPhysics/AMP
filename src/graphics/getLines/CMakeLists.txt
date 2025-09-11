# Create the application
SET( QtApp_SRCS get_lines.cpp mainwindow.cpp ThreadedSlotsClass.cpp )
SET( QtApp_MOCS mainwindow.h ThreadedSlotsClass.h )
SET( QtApp_RCCS application.qrc )
IF ( ${QT_VERSION} EQUAL "4" )
    QT4_ADD_RESOURCES( QtApp_RCC_SRCS ${QtApp_RCCS} )
    QT4_WRAP_CPP( QtApp_MOC_SRCS ${QtApp_MOCS} )
ELSEIF ( ${QT_VERSION} EQUAL "5" )
    QT5_ADD_RESOURCES( QtApp_RCC_SRCS ${QtApp_RCCS} )
    QT5_WRAP_CPP( QtApp_MOC_SRCS ${QtApp_MOCS} )
ENDIF()

# build sources, moc'd sources, and rcc'd sources
ADD_EXECUTABLE( get_lines ${QtApp_SRCS} ${QtApp_MOC_SRCS} ${QtApp_RCC_SRCS} )
INSTALL( TARGETS get_lines DESTINATION ${${PROJ}_INSTALL_DIR}/bin )
SET_TARGET_PROPERTIES( get_lines PROPERTIES COMPILE_FLAGS "${QT_FLAGS} ${COVERAGE_FLAGS}" )
TARGET_LINK_LIBRARIES( get_lines ${${QT}_QTMAIN_LIBRARY} ${${QT}_LIBRARIES} )
FOREACH( tmp ${QT_COMPONENTS} )
    TARGET_LINK_LIBRARIES( get_lines ${Qt}::${tmp} )
ENDFOREACH()
TARGET_LINK_LIBRARIES( get_lines ${Qt}::Charts )
ADD_PROJ_EXE_DEP( get_lines )
GLOBAL_SET( get_lines-BINDIR "${CMAKE_CURRENT_BINARY_DIR}" )

# Add the tests SET( COPY_LIST  set1.1.timer  set1.2.timer  set1.3.timer  set1.4.timer set2.1.timer  set2.2.timer  set2.3.timer  set2.4.timer set2.1.trace  set2.2.trace
# set2.3.trace  set2.4.trace set2.1.memory set2.2.memory set2.3.memory set2.4.memory ) FOREACH( tmp ${COPY_LIST} ) CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/../test/data/${tmp}
# ${CMAKE_CURRENT_BINARY_DIR}/${tmp} COPYONLY ) ENDFOREACH() ADD_TIMER_GUI_TEST( get_lines set1.1.timer set2.1.timer )
