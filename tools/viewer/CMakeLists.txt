find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${ROOT_DIRECTORY}/modules/face-landmark/include"
    "${ROOT_DIRECTORY}/modules/face-detector/include")

file(GLOB TOOL_VIEWER_SRC "source/*.cpp")

add_executable(tool_viewer ${TOOL_VIEWER_SRC} )
target_link_libraries(tool_viewer module_landmark module_detector ${OpenCV_LIBS})
set_target_properties(tool_viewer PROPERTIES
    OUTPUT_NAME "tool_viewer"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
