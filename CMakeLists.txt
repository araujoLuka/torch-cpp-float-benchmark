cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(torch_cpp_precision_experiment VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LibTorch
set(CMAKE_PREFIX_PATH "./include/libtorch")
find_package(Torch REQUIRED)

# # LIKWID
# find_library(LIKWID_LIB likwid REQUIRED)
# include_directories(/usr/local/include)

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

# Linka Torch
target_compile_definitions(${EXEC_NAME} PRIVATE -D_GLIBCXX_USE_CXX11_ABI=0)
target_link_libraries(${EXEC_NAME} "${TORCH_LIBRARIES}")

# Garante que o padrão seja C++17
set_property(TARGET ${EXEC_NAME} PROPERTY CXX_STANDARD 17)

# Para evitar warnings de linking em algumas plataformas
if (UNIX)
  target_link_libraries(${EXEC_NAME} pthread)
endif()

# Argumentos para debug
set_target_properties(${EXEC_NAME} PROPERTIES
    VS_DEBUGGER_COMMAND_ARGUMENTS "--dataset Fruits360 --data_root ${CMAKE_SOURCE_DIR}/datasets"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)