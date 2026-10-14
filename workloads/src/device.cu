// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include <cuda_runtime.h>

#include "probe/cuda_check.hpp"
#include "probe/device.hpp"

namespace probe {

namespace {

// CUDA encodes versions as 1000 * major + 10 * minor (12020 -> "12.2").
std::string format_cuda_version(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

}  // namespace

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

void write_device_json(JsonWriter& w, const DeviceInfo& info) {
    w.begin_object()
        .key("index")
        .integer(info.index)
        .key("count")
        .integer(info.count)
        .key("name")
        .string(info.name)
        .key("compute_capability")
        .string(std::to_string(info.cc_major) + "." + std::to_string(info.cc_minor))
        .key("sm_count")
        .integer(info.sm_count)
        .key("global_mem_bytes")
        .uinteger(info.global_mem_bytes)
        .key("cuda_runtime_version")
        .string(format_cuda_version(info.runtime_version))
        .key("cuda_driver_version")
        .string(format_cuda_version(info.driver_version))
        .end_object();
}

}  // namespace probe
