find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS})

file(GLOB TOOL_TRAIN_SRC "source/*.cpp")

add_executable(tool_train ${TOOL_TRAIN_SRC} )
target_link_libraries(tool_train ${OpenCV_LIBS})
set_target_properties(tool_train PROPERTIES
    OUTPUT_NAME "tool_train"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
