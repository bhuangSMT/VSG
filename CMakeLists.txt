cmake_minimum_required(VERSION 3.16)

project(vsg_qt_cube
    VERSION 1.0.0
    DESCRIPTION "VulkanSceneGraph cube rendered inside a Qt main window"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Basic C++ settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(UCAM_BUILD_SHARED "Build geom/boolean/graphics as shared libraries" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Release: -O3 -DNDEBUG plus LTO and host CPU codegen. Not for Debug.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_msg)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "LTO not available: ${_ipo_msg}")
    endif()
    if(NOT MSVC)
        add_compile_options(-O3 -march=native)
    endif()
endif()

# Keep the executable and libraries at the build-tree root for both Ninja and Visual Studio.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}")
endforeach()

include(GNUInstallDirs)

# Shared libs next to the exe in the build tree; installed libs next to each other.
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN;$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Qt code generation (moc/rcc/uic). vsgQt embeds a QWindow, so AUTOMOC is required.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Match the Qt major version that vsgQt was built against (Qt5 or Qt6).
set(QT_PACKAGE_NAME Qt6 CACHE STRING "Qt package name, i.e. Qt5 or Qt6.")

# ---------------------------------------------------------------------------
# Dependencies
#
# These are resolved via CMAKE_PREFIX_PATH. Run scripts/bootstrap.sh (macOS)
# or scripts/bootstrap.ps1 (Windows) first, which installs Vulkan + Qt and
# builds VulkanSceneGraph + vsgQt into ./.deps.
# ---------------------------------------------------------------------------
find_package(Vulkan REQUIRED)
find_package(vsg 1.1.13 REQUIRED)
find_package(vsgQt REQUIRED)
find_package(${QT_PACKAGE_NAME} COMPONENTS Widgets REQUIRED)

# oneTBB parallelises the ray casting in RayModel.
find_package(TBB REQUIRED)

# zlib inflates the parts of a 3MF package. A 3MF is a ZIP archive, and this is
# the only piece of that the importer does not do itself.
find_package(ZLIB REQUIRED)

# ---------------------------------------------------------------------------
# Libraries: geom → boolean → graphics
# ---------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/UcamLibrary.cmake)
add_subdirectory(lib/geom)
add_subdirectory(lib/boolean)
add_subdirectory(lib/graphics)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
set(VSG_QT_CUBE_SOURCES
    main/main.cpp
    main/SimulationPanel.cpp
    main/StlImporter.cpp
    main/ThreeMfImporter.cpp
)

if(COMMAND qt_add_executable)
    qt_add_executable(vsg_qt_cube ${VSG_QT_CUBE_SOURCES})
else()
    add_executable(vsg_qt_cube ${VSG_QT_CUBE_SOURCES})
endif()

if(WIN32)
    set_target_properties(vsg_qt_cube PROPERTIES WIN32_EXECUTABLE TRUE)
    target_compile_definitions(vsg_qt_cube PRIVATE NOMINMAX)
endif()

target_include_directories(vsg_qt_cube PRIVATE ${PROJECT_SOURCE_DIR}/main)

target_link_libraries(vsg_qt_cube
    PRIVATE
        ucam_graphics
        ${QT_PACKAGE_NAME}::Widgets
        ZLIB::ZLIB
)

install(TARGETS vsg_qt_cube RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# ---------------------------------------------------------------------------
# Headless hotspot profiler (cast / sweep / boolean / splat cache).
# ---------------------------------------------------------------------------
add_executable(profile_hotspots main/profile_hotspots.cpp)
target_link_libraries(profile_hotspots PRIVATE ucam_graphics)
if(WIN32)
    target_compile_definitions(profile_hotspots PRIVATE NOMINMAX)
endif()

# ---------------------------------------------------------------------------
# Install libraries, headers (from each subdir), and CMake package files so
# an external project can find_package(ucam) and link ucam::graphics.
# ---------------------------------------------------------------------------
install(TARGETS ucam_geom ucam_boolean ucam_graphics
    EXPORT ucamTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT ucamTargets
    FILE ucamTargets.cmake
    NAMESPACE ucam::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucam
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/ucamConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ucamConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucam
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ucamConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ucamConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ucamConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ucam
)

# ---------------------------------------------------------------------------
# Windows: copy VSG/vsgQt (and ucam) DLLs next to the exe and run windeployqt
# so the Qt platform plugin (qwindows.dll) is available when launching from
# the build tree.
# ---------------------------------------------------------------------------
if(WIN32)
    set(_ucam_dll_copy)
    if(UCAM_BUILD_SHARED)
        set(_ucam_dll_copy
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:ucam_geom>
                $<TARGET_FILE:ucam_boolean>
                $<TARGET_FILE:ucam_graphics>
                $<TARGET_FILE_DIR:vsg_qt_cube>
        )
    endif()

    add_custom_command(TARGET vsg_qt_cube POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:vsg::vsg>
            $<TARGET_FILE_DIR:vsg_qt_cube>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:vsgQt::vsgQt>
            $<TARGET_FILE_DIR:vsg_qt_cube>
        ${_ucam_dll_copy}
        COMMENT "Copying VSG DLLs next to vsg_qt_cube"
    )

    get_filename_component(_qt_prefix "${${QT_PACKAGE_NAME}_DIR}/../../../" ABSOLUTE)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${_qt_prefix}/bin")
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET vsg_qt_cube POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                --no-translations
                --no-compiler-runtime
                "$<TARGET_FILE:vsg_qt_cube>"
            COMMENT "Running windeployqt"
        )
    else()
        message(WARNING "windeployqt not found under ${_qt_prefix}/bin. "
                        "Qt plugins may be missing at runtime; set PATH to the Qt bin dir.")
    endif()
endif()
