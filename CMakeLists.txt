cmake_minimum_required(VERSION 3.16)
project(torch_cpp_float_benchmark)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/path/to/libtorch")  # <--- Altere aqui

# LibTorch
find_package(Torch REQUIRED)

# LIKWID
find_library(LIKWID_LIB likwid REQUIRED)
include_directories(/usr/local/include)

# Fontes
add_executable(${PROJECT_NAME}
    main.cpp
    net/net.cpp
    net/linear_custom.h
    net/net.h
    include/utils.h
    data/pklot_loader.h
    benchmark/benchmark.h
)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" ${LIKWID_LIB})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
