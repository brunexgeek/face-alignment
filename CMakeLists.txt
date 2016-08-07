project(face-alignment)

cmake_minimum_required(VERSION 2.8)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



set(ENABLE_VALGRIND OFF CACHE BOOL "")

set(CMAKE_BUILD_PLATFORM "Auto"  CACHE STRING  "")
set_property(CACHE  CMAKE_BUILD_PLATFORM PROPERTY STRINGS "Auto" "64 bits" "32 bits" "armeabi" "armeabi-v7a" )

set(CMAKE_BUILD_TYPE "Auto"  CACHE STRING  "")
set_property(CACHE  CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")



##
## Operating system identification
##
set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")


##
## Root directory
##
set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})


##
## If no build type is specified, we choose "Release"
##
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "No build type selected, default to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif()


##
## If the build platform is "Auto", automatically selects the right value.
##
if(${CMAKE_BUILD_PLATFORM} STREQUAL "Auto")
    if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
        set(CMAKE_BUILD_PLATFORM "32 bits")
    elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set(CMAKE_BUILD_PLATFORM "64 bits")
    endif()
endif()


##
## Some generic libraries used by all modules/tools.
##
find_package (Threads)
set(GLOBAL_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})


if (UNIX)
    set(__ADNDROID__ 0)
    if(CMAKE_BUILD_PLATFORM STREQUAL "32 bits")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m32" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m32" )
    elseif(CMAKE_BUILD_PLATFORM STREQUAL "64 bits")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -m64" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -m64" )
    elseif(CMAKE_BUILD_PLATFORM STREQUAL "armeabi")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=vfpv3 -mfloat-abi=softfp" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=vfpv3 -mfloat-abi=softfp" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mfpu=vfpv3 -mfloat-abi=softfp" )
        set(ENABLE_ANDROID 1)
    elseif(CMAKE_BUILD_PLATFORM STREQUAL "armeabi-v7a")
        set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a -mfpu=vfpv3 -mfloat-abi=softfp" )
        set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv7-a -mfpu=vfpv3 -mfloat-abi=softfp" )
        set( CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -march=armv7-a -mfpu=vfpv3 -mfloat-abi=softfp" )
        set(ENABLE_ANDROID 1)
    endif()
endif()


if (ENABLE_VALGRIND)
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -DVALGRIND=1")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DVALGRIND=1")
endif()


if (UNIX)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC -std=gnu99 -Wall -Werror=implicit-function-declaration")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror=implicit-function-declaration")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -s -O3")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s -O3")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
endif()


################################################################################
## Compilation targets                                                        ##
################################################################################

add_subdirectory(modules/face-detector)
add_subdirectory(modules/face-landmark)
add_subdirectory(tools/video)
add_subdirectory(tools/viewer)
add_subdirectory(tools/test)
add_subdirectory(tools/train)
add_subdirectory(tools/simulate)



message("")
message("  ======== Enviroment ========")
message("                Host:  ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
message("               CMake:  ${CMAKE_VERSION}")
message("           Generator:  ${CMAKE_GENERATOR}")
message("          Build tool:  ${CMAKE_BUILD_TOOL}")
if(MSVC)
   message("        MSVC version:  ${MSVC_VERSION}")
endif()


message("")
message("  ========== Target ==========")
message("    Operating system:  ${CMAKE_SYSTEM_NAME}")
message("            Platform:  ${CMAKE_BUILD_PLATFORM}")
message("          Build type:  ${CMAKE_BUILD_TYPE}")


message("")
message("  ======== Compilation =======")
message("      Root directory: ${ROOT_DIRECTORY}")

if (CMAKE_CXX_COMPILER_VERSION)
    set(FALIGN_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_COMPILER_IS_CLANGCXX)
    set(FALIGN_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CLANG_REGEX_VERSION})")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(FALIGN_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_GCC_REGEX_VERSION})")
else()
    set(FALIGN_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${FALIGN_COMPILER_STR}" FALIGN_COMPILER_STR)
message("        C++ Compiler:  ${FALIGN_COMPILER_STR}")
message(" C++ flags (Release):  ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message("   C++ flags (Debug):   ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
message("          C Compiler:  ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}")
message("   C flags (Release):  ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}")
message("     C flags (Debug):  ${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}")
if(WIN32)
    message("  LD flags (Release): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    message("    LD flags (Debug):   ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
    message("  LD flags (Release): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    message("    LD flags (Debug):   ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()
message("")
