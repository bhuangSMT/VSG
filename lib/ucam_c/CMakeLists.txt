add_library(ucam SHARED ucam.cpp)
target_compile_definitions(ucam PRIVATE UCAM_EXPORTS)
set_target_properties(ucam PROPERTIES
    EXPORT_NAME ucam
    OUTPUT_NAME ucam
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
add_library(ucam::ucam ALIAS ucam)

configure_file(ucam.h ${CMAKE_CURRENT_BINARY_DIR}/include/ucam/ucam.h COPYONLY)

target_include_directories(ucam
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/lib/geom
        ${PROJECT_SOURCE_DIR}/lib/boolean
        ${PROJECT_SOURCE_DIR}/lib/graphics
)

target_link_libraries(ucam PRIVATE ucam_graphics ${QT_PACKAGE_NAME}::Widgets)

if(WIN32)
    target_compile_definitions(ucam PRIVATE NOMINMAX)
endif()

install(TARGETS ucam
    EXPORT ucamSdkTargets
    COMPONENT ucam_sdk
    EXCLUDE_FROM_ALL
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ucam.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ucam
    COMPONENT ucam_sdk
    EXCLUDE_FROM_ALL
)
