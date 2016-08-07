find_package( OpenCV REQUIRED )

include_directories(
    "${ROOT_DIRECTORY}/modules/face-detector/include"
    "${ROOT_DIRECTORY}/modules/face-landmark/include"
    ${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCE "source/*.cpp")

add_executable(tool_video ${SOURCE} )
target_link_libraries(tool_video module_detector module_landmark ${OpenCV_LIBS})
set_target_properties(tool_video PROPERTIES
    OUTPUT_NAME "tool_video"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
