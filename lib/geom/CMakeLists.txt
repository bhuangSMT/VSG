ucam_add_library(ucam_geom geom
    BoundingBox.cpp
    BRep.cpp
    BVH.cpp
    SweptVolume.cpp
    ToolGeometry.cpp
)

target_link_libraries(ucam_geom PUBLIC vsg::vsg)

install(FILES
    BoundingBox.h
    BRep.h
    BVH.h
    Ray.h
    SweptVolume.h
    ToolGeometry.h
    ToolType.h
    TriangleMesh.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
