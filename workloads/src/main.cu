// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// probe-workloads: the single executable that hosts Probe's CUDA workloads.
// The Python CLI invokes it and reads JSON from stdout.
//
// Exit codes are part of the contract with the Python side:
//   0  success
//   1  CUDA runtime error (see PROBE_CUDA_CHECK)
//   2  usage error
//   3  no usable CUDA device

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>

#include "probe/cuda_check.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;
constexpr int kExitNoDevice = 3;

void print_usage(const char* prog) {
    std::printf(
        "usage: %s --device-info\n"
        "       %s --version\n"
        "\n"
        "  --device-info  print the selected GPU's properties as JSON\n"
        "  --version      print the probe-workloads version and build type\n",
        prog, prog);
}

// CUDA encodes versions as 1000 * major + 10 * minor (12020 -> "12.2").
std::string format_cuda_version(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

// Device names are plain ASCII in practice, but escape the two characters
// that would break a JSON string so the output is always valid.
std::string json_escape(const char* s) {
    std::string out;
    for (; *s != '\0'; ++s) {
        if (*s == '"' || *s == '\\') out.push_back('\\');
        out.push_back(*s);
    }
    return out;
}

int print_device_info() {
    int runtime_version = 0;
    PROBE_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));

    // Returns 0 rather than an error when no driver is installed.
    int driver_version = 0;
    PROBE_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    // No GPU is an expected condition, not a crash: report it and exit with a
    // distinct code so callers can tell it apart from a real CUDA failure.
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "probe-workloads: no usable CUDA device (%s)\n",
                     err != cudaSuccess ? cudaGetErrorString(err) : "device count is 0");
        return kExitNoDevice;
    }

    const int device = 0;
    PROBE_CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    PROBE_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf(
        "{\"schema\": \"probe.device_info/1\", "
        "\"device_index\": %d, \"device_count\": %d, "
        "\"name\": \"%s\", \"compute_capability\": \"%d.%d\", "
        "\"sm_count\": %d, \"global_mem_bytes\": %zu, "
        "\"cuda_runtime_version\": \"%s\", \"cuda_driver_version\": \"%s\"}\n",
        device, device_count, json_escape(prop.name).c_str(), prop.major, prop.minor,
        prop.multiProcessorCount, prop.totalGlobalMem, format_cuda_version(runtime_version).c_str(),
        format_cuda_version(driver_version).c_str());
    return kExitOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        print_usage(argv[0]);
        return kExitUsage;
    }
    if (std::strcmp(argv[1], "--device-info") == 0) return print_device_info();
    if (std::strcmp(argv[1], "--version") == 0) {
        std::printf("probe-workloads %s (%s)\n", PROBE_VERSION, PROBE_BUILD_TYPE);
        return kExitOk;
    }
    if (std::strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return kExitOk;
    }
    print_usage(argv[0]);
    return kExitUsage;
}
