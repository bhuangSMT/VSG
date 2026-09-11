ucam_add_library(ucam_boolean boolean
    RayBoolean.cpp
    RayHit.cpp
    RayModel.cpp
)

target_link_libraries(ucam_boolean PUBLIC ucam_geom vsg::vsg TBB::tbb)

install(FILES
    BooleanOp.h
    RayBoolean.h
    RayGrid.h
    RayHit.h
    RayModel.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
