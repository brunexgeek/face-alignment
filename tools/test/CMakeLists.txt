find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS})

file(GLOB TOOL_TEST_SRC "source/*.cpp")

#add_executable(tool_test ${TOOL_TEST_SRC} )
#target_link_libraries(tool_test ${OpenCV_LIBS})
#set_target_properties(tool_test PROPERTIES
#    OUTPUT_NAME "tool_test"
#    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
