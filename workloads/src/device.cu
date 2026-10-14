// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include <cuda_runtime.h>

#include "probe/cuda_check.hpp"
#include "probe/device.hpp"

namespace probe {

bool select_device(int index, DeviceInfo& out, std::string& error) {
    PROBE_CUDA_CHECK(cudaRuntimeGetVersion(&out.runtime_version));
    PROBE_CUDA_CHECK(cudaDriverGetVersion(&out.driver_version));

    const cudaError_t err = cudaGetDeviceCount(&out.count);
    if (err != cudaSuccess || out.count == 0) {
        error = err != cudaSuccess ? cudaGetErrorString(err) : "device count is 0";
        return false;
    }
    if (index >= out.count) {
        error = "device index " + std::to_string(index) + " out of range";
        return false;
    }

    PROBE_CUDA_CHECK(cudaSetDevice(index));
    cudaDeviceProp prop{};
    PROBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, index));
    out.index = index;
    out.name = prop.name;
    out.cc_major = prop.major;
    out.cc_minor = prop.minor;
    out.sm_count = prop.multiProcessorCount;
    out.global_mem_bytes = prop.totalGlobalMem;
    return true;
}

}  // namespace probe
