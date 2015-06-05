find_package( OpenCV REQUIRED )

include_directories(
    "../../3rd-party/flandmark/libflandmark"
    "../../modules/face-detector/include"
    "../../modules/face-landmark/include"
    ${OpenCV_INCLUDE_DIRS})

file(GLOB DEMO_GUI_SRC "source/*.cpp")

add_executable(demo_gui ${DEMO_GUI_SRC} )
target_link_libraries(demo_gui module_detector module_landmark ${OpenCV_LIBS})
set_target_properties(demo_gui PROPERTIES
    OUTPUT_NAME "demo_gui"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" )
