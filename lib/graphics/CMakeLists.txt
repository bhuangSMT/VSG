ucam_add_library(ucam_graphics graphics
    ControlCube.cpp
    GaussianSplat.cpp
    GaussianSplatCache.cpp
    ModelPick.cpp
    Parameter.cpp
    RenderManager.cpp
)

target_link_libraries(ucam_graphics PUBLIC ucam_boolean vsg::vsg vsgQt::vsgQt TBB::tbb)

install(FILES
    ControlCube.h
    GaussianSplat.h
    GaussianSplatCache.h
    ModelPick.h
    Parameter.h
    RenderManager.h
    SimulationMode.h
    ViewMode.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
