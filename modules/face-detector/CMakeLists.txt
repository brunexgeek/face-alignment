find_package( OpenCV REQUIRED )

include_directories(
    "${ROOT_DIRECTORY}/modules/face-detector/include"
    ${OpenCV_INCLUDE_DIRS} )

file(GLOB MODULE_DETECTOR_SRC "source/*.cpp")

add_library(module_detector STATIC ${MODULE_DETECTOR_SRC} )
target_link_libraries(module_detector ${OpenCV_LIBS})
set_target_properties(module_detector PROPERTIES
    OUTPUT_NAME "detector"
    #SOVERSION "${TTS_MAJOR_VERSION}.${TTS_MINOR_VERSION}.${TTS_PATCH_VERSION}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    PREFIX "lib" )
