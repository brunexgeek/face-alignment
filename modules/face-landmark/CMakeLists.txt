find_package( OpenCV REQUIRED )

include_directories(
    "./include"
    "../../3rd-party/flandmark/libflandmark"
    ${OpenCV_INCLUDE_DIRS} )

file(GLOB MODULE_LANDMARK_SRC
	"source/*.cpp"
	"source/ert/*.cpp")

add_library(module_landmark STATIC ${MODULE_LANDMARK_SRC} )
target_link_libraries(module_landmark flandmark_static ${OpenCV_LIBS})
set_target_properties(module_landmark PROPERTIES
    OUTPUT_NAME "landmark"
    #SOVERSION "${TTS_MAJOR_VERSION}.${TTS_MINOR_VERSION}.${TTS_PATCH_VERSION}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "lib" )
