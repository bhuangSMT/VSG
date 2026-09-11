ucam_add_library(ucam_aptparser aptparser
    APTParser.cpp
)

install(FILES
    APTParser.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
