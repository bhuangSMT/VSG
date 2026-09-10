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

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Keep the executable at the build-tree root for both Ninja and Visual Studio.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
foreach(_cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg} "${CMAKE_BINARY_DIR}")
endforeach()

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
# Application
# ---------------------------------------------------------------------------
set(VSG_QT_CUBE_SOURCES
    src/main.cpp
    src/StlImporter.cpp
    src/ThreeMfImporter.cpp
    src/BRep.cpp
    src/BVH.cpp
    src/BoundingBox.cpp
    src/GaussianSplat.cpp
    src/GaussianSplatCache.cpp
    src/ModelPick.cpp
    src/Parameter.cpp
    src/RayModel.cpp
    src/RayBoolean.cpp
    src/RenderManager.cpp
    src/SweptVolume.cpp
    src/ToolGeometry.cpp
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

target_link_libraries(vsg_qt_cube
    PRIVATE
        vsgQt::vsgQt
        vsg::vsg
        ${QT_PACKAGE_NAME}::Widgets
        TBB::tbb
        ZLIB::ZLIB
)

install(TARGETS vsg_qt_cube RUNTIME DESTINATION bin)

# ---------------------------------------------------------------------------
# Headless hotspot profiler (cast / sweep / boolean / splat cache).
# ---------------------------------------------------------------------------
add_executable(profile_hotspots
    src/profile_hotspots.cpp
    src/BRep.cpp
    src/BVH.cpp
    src/BoundingBox.cpp
    src/GaussianSplat.cpp
    src/GaussianSplatCache.cpp
    src/RayModel.cpp
    src/RayBoolean.cpp
    src/SweptVolume.cpp
    src/ToolGeometry.cpp
)
target_link_libraries(profile_hotspots
    PRIVATE
        vsg::vsg
        TBB::tbb
)
if(WIN32)
    target_compile_definitions(profile_hotspots PRIVATE NOMINMAX)
endif()

# ---------------------------------------------------------------------------
# Windows: copy VSG/vsgQt DLLs next to the exe and run windeployqt so the
# Qt platform plugin (qwindows.dll) is available when launching from the
# build tree.
# ---------------------------------------------------------------------------
if(WIN32)
    add_custom_command(TARGET vsg_qt_cube POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:vsg::vsg>
            $<TARGET_FILE_DIR:vsg_qt_cube>
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:vsgQt::vsgQt>
            $<TARGET_FILE_DIR:vsg_qt_cube>
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
