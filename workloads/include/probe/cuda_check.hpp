// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

// Almost every CUDA runtime call returns a cudaError_t. Ignoring it means a
// failed allocation or launch surfaces later as wrong output or a confusing
// crash. This macro checks the result immediately and exits with the file and
// line of the call that failed.
#define PROBE_CUDA_CHECK(call)                                                                 \
    do {                                                                                       \
        const cudaError_t probe_err_ = (call);                                                 \
        if (probe_err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(probe_err_), \
                         __FILE__, __LINE__, cudaGetErrorString(probe_err_));                  \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)
