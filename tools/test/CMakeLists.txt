find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    "${ROOT_DIRECTORY}/modules/face-landmark/include"
    "${ROOT_DIRECTORY}/modules/face-detector/include")

file(GLOB TOOL_TEST_SRC "source/*.cpp")

add_executable(tool_test ${TOOL_TEST_SRC} )
target_link_libraries(tool_test module_detector module_landmark ${OpenCV_LIBS})
set_target_properties(tool_test PROPERTIES
    OUTPUT_NAME "tool_test"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
