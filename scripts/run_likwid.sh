#!/bin/bash
likwid-perfctr -C 0-3 -g FLOPS_DP -m ./torch-cpp-float-benchmark
