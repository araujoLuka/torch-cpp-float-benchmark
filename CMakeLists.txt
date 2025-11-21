cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(torch_cpp_precision_experiment VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Install CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

# LibTorch
set(CMAKE_PREFIX_PATH "./include/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# # LIKWID
# find_library(LIKWID_LIB likwid REQUIRED)
# include_directories(/usr/local/include)

# JSON
CPMAddPackage(
    NAME nlohmann_json
    VERSION 3.12.0
    GITHUB_REPOSITORY nlohmann/json
    OPTIONS "NLOHMANN_JSON_BUILD_TESTS OFF"
)

# Adiciona a definição para habilitar os marcadores do LIKWID
# add_definitions(-DLIKWID_PERFMON)

# Fontes
set(SOURCES
    source/main.cpp
    include/Net.hpp
)

set(EXEC_NAME "cnn_experiment")

add_executable(${EXEC_NAME} ${SOURCES})

target_include_directories(${EXEC_NAME} PRIVATE include)

# Linka as bibliotecas necessárias
if (!UNIX)
    target_link_libraries(${EXEC_NAME} PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
else()
    target_link_libraries(${EXEC_NAME} PRIVATE "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json pthread)
endif()

# Garante que o padrão seja C++17
set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "libtorch")
