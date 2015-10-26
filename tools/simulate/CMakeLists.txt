find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "../../modules/face-detector/include")

file(GLOB TOOL_SIMULATE_SRC "source/*.cpp")

add_executable(tool_simulate ${TOOL_SIMULATE_SRC} )
target_link_libraries(tool_simulate module_detector module_landmark ${OpenCV_LIBS})
set_target_properties(tool_simulate PROPERTIES
    OUTPUT_NAME "tool_simulate"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
