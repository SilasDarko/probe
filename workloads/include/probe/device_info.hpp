// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <string>

#include "probe/json_writer.hpp"

namespace probe {

// What the binary knows about the GPU it ran on. Plain data, no CUDA types,
// so it can be serialized and tested without a GPU.
struct DeviceInfo {
    int index = 0;
    int count = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t global_mem_bytes = 0;
    int runtime_version = 0;  // CUDA encoding: 1000 * major + 10 * minor
    int driver_version = 0;   // highest CUDA version the driver supports; 0 if none
};

// CUDA encodes versions as 1000 * major + 10 * minor (12020 -> "12.2").
std::string format_cuda_version(int v);

// Writes the device as a JSON object (no surrounding key).
void write_device_json(JsonWriter& w, const DeviceInfo& info);

}  // namespace probe
