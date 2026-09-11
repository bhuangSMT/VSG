# Shared setup for ucam_geom / ucam_boolean / ucam_graphics / ucam_aptparser.
function(ucam_add_library target export_name)
    if(UCAM_BUILD_SHARED)
        add_library(${target} SHARED ${ARGN})
    else()
        add_library(${target} STATIC ${ARGN})
    endif()

    set_target_properties(${target} PROPERTIES
        EXPORT_NAME ${export_name}
        POSITION_INDEPENDENT_CODE ON
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
    add_library(ucam::${export_name} ALIAS ${target})

    target_include_directories(${target}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
    )
endfunction()
