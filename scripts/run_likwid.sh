#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/../build"

sudo likwid-perfctr -C S0:0 -g L3 -m $BUILD_DIR/torch_cpp_float_benchmark
