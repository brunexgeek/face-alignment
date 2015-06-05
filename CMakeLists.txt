project(cpqdtts)

cmake_minimum_required(VERSION 2.8)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


################################################################################
## Definição de opções globais de compilação                                  ##
################################################################################


##
## Opção para uso do Valgrind
##
set(ENABLE_VALGRIND OFF CACHE BOOL "Habilita modificações no código que facilitam análises usando Valgrind em modo Debug")

##
## Opção para escolha da plataforma alvo.
##
set(CMAKE_BUILD_PLATFORM "Auto"  CACHE STRING  "Especifica a plataforma para compilação")
set_property(CACHE  CMAKE_BUILD_PLATFORM PROPERTY STRINGS "Auto" "64 bits" "32 bits" "armeabi" "armeabi-v7a" )

##
## Opção para escolha do modo de compilação.
##
set(CMAKE_BUILD_TYPE "Auto"  CACHE STRING  "Especifica o modo de compilação")
set_property(CACHE  CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")

##
## Define o nível de log desejado
##
set(ENABLE_LOG "OFF"  CACHE STRING  "Especifica o nível de log")
set_property(CACHE ENABLE_LOG PROPERTY STRINGS "LEVEL_TRACE" "LEVEL_DEBUG" "LEVEL_INFO" "LEVEL_WARNING" "LEVEL_ERROR" "LEVEL_FATAL" "OFF")


################################################################################
## Tratamento de opções e criação de parâmetros utilizados no fonte           ##
################################################################################


##
## Define a identificação do sistema operacional
##
set(CMAKE_SYSTEM_NAME "${CMAKE_HOST_SYSTEM_NAME}")


##
## Especifica o diretório raiz do projeto
##
set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})


##
## Se nenhum modo de compilação for especificado, utilizamos "Release"
##
if (CMAKE_BUILD_TYPE STREQUAL "")
    message(STATUS "No build type selected, default to 'Release'")
    set(CMAKE_BUILD_TYPE "Release")
endif()


##
## Atualiza o identificador da plataforma alvo caso tenha sido selecionado
## o modo automático (Auto).
##
if(${CMAKE_BUILD_PLATFORM} STREQUAL "Auto")
    if(${CMAKE_SIZEOF_VOID_P} EQUAL 4)
        set(CMAKE_BUILD_PLATFORM "32 bits")
    elseif(${CMAKE_SIZEOF_VOID_P} EQUAL 8)
        set(CMAKE_BUILD_PLATFORM "64 bits")
    endif()
endif()


##
## Define as bibliotecas adicionais a serem utilizadas.
##
find_package (Threads)
set(TTS_LIBS_THREAD ${CMAKE_THREAD_LIBS_INIT})
set(TTS_LIBS_DL ${CMAKE_DL_LIBS})


##
## Define as diretivas de compilação do GCC para a plataforma alvo.
##
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


##
## Define as flags de compilação para permitir o uso do Valgrind, caso habilitado
##
if (ENABLE_VALGRIND)
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -DVALGRIND=1")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -DVALGRIND=1")
endif()


##
## Define flags de compilação exclusivas de ambientes Unix-like (usando GCC ou Clang)
##
if (UNIX)
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC -std=gnu99 -Wall -Werror=implicit-function-declaration")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC -Wall -Werror=implicit-function-declaration")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} -s -O3")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -s -O3")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -O0")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    # Introduz flags de compilação para limitar o comprimento dos nomes de arquivos fonte
    # fornecidos pelo compilador. Esses nomes são utilizados pelo logger, por exemplo,
    # para indicar o arquivo fonte que gerou determinada saída.
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D__FILENAME__='\"$(subst ${CMAKE_CURRENT_LIST_DIR}/,,$(abspath $<))\"'")
endif()

##
## Define flags de compilação exclusivas de ambientes Windows (usando Visual Studio)
##
if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


##
## Define o nível do log
##
if (${ENABLE_LOG} STREQUAL "OFF")
  set (LEVEL_LOG "LEVEL_OFF")
  set (ENABLE_LOG OFF)
else()
  set (LEVEL_LOG "${ENABLE_LOG}")
  set (ENABLE_LOG ON)
endif()


################################################################################
## Adiciona os scripts de compilação do projeto                               ##
################################################################################

add_subdirectory(3rd-party/flandmark)
add_subdirectory(demos/features)
add_subdirectory(modules/face-detector)
add_subdirectory(modules/face-landmark)


################################################################################
## Exibe informações sobre os parâmetros de compilação                        ##
################################################################################

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
if (ENABLE_ANDROID)
    message("           API Level:  ${ANDROID_API_LEVEL}")
    message("       NDK directory:  ${ANDROID_DIR}")
endif()
message("          Build type:  ${CMAKE_BUILD_TYPE}")


message("")
message("  ========== Cryptographic Key =========")
message("           Size:  ${SIZE_KEY}")
message("            Key:  ${LICENSE_CRYPTO_KEY}")

message("")
message("  ========== Options =========")
message("        Java Wrapper:  ${ENABLE_JAVA_WRAPPER}")
message("            Valgrind:  ${ENABLE_VALGRIND}")
message("                 Log:  ${ENABLE_LOG}")
if (SUPPORT_CONVERTER_WAV)
    set(AUDIO_CONVERTERS "wav")
endif()
if (SUPPORT_CONVERTER_PCM)
    set(AUDIO_CONVERTERS "${AUDIO_CONVERTERS} pcm")
endif()
if (SUPPORT_CONVERTER_ALAW)
    set(AUDIO_CONVERTERS "${AUDIO_CONVERTERS} alaw")
endif()
if (SUPPORT_CONVERTER_ALAW_HEADER)
    set(AUDIO_CONVERTERS "${AUDIO_CONVERTERS} alaw_header")
endif()
message("     Audio Converter:  ${AUDIO_CONVERTERS}")


message("")
message("  ======== Compilation =======")
message("      Root directory: ${ROOT_DIRECTORY}")

if (CMAKE_CXX_COMPILER_VERSION)
    set(CPQD_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CXX_COMPILER_VERSION})")
elseif(CMAKE_COMPILER_IS_CLANGCXX)
    set(CPQD_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_CLANG_REGEX_VERSION})")
elseif(CMAKE_COMPILER_IS_GNUCXX)
    set(CPQD_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1} (ver ${CMAKE_GCC_REGEX_VERSION})")
else()
    set(CPQD_COMPILER_STR "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
endif()
string(STRIP "${CPQD_COMPILER_STR}" CPQD_COMPILER_STR)
message("        C++ Compiler:  ${CPQD_COMPILER_STR}")
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
