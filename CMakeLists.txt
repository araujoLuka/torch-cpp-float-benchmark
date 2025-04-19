cmake_minimum_required(VERSION 3.16)
project(torch_cpp_float_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LibTorch
find_package(Torch REQUIRED)
set(CMAKE_PREFIX_PATH "./include/libtorch")

# LIKWID
find_library(LIKWID_LIB likwid REQUIRED)
include_directories(/usr/local/include)

# Adiciona a definição para habilitar os marcadores do LIKWID
add_definitions(-DLIKWID_PERFMON)

# Fontes
set(SOURCES
    main.cpp
    net/net.cpp
    net/net.h
    net/linear_custom.h
    include/utils.h
    data/pklot_loader.h
    benchmark/benchmark.h
)

add_executable(${PROJECT_NAME} ${SOURCES})

# Linka Torch e LIKWID
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${LIKWID_LIB})

# Garante que o padrão seja C++17
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
