ucam_add_library(ucam_graphics graphics
    GaussianSplat.cpp
    GaussianSplatCache.cpp
    ModelPick.cpp
    Parameter.cpp
    RenderManager.cpp
)

target_link_libraries(ucam_graphics PUBLIC ucam_boolean vsg::vsg vsgQt::vsgQt TBB::tbb)

install(FILES
    GaussianSplat.h
    GaussianSplatCache.h
    ModelPick.h
    Parameter.h
    RenderManager.h
    ViewMode.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
