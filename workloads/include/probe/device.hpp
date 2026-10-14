// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <string>

#include "probe/json_writer.hpp"

namespace probe {

struct DeviceInfo {
    int index = 0;
    int count = 0;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int sm_count = 0;
    std::size_t global_mem_bytes = 0;
    int runtime_version = 0;  // CUDA encoding: 1000 * major + 10 * minor
    int driver_version = 0;   // 0 when no driver is installed
};

// Selects device `index` and fills `out`. Returns false with a message in
// `error` when no usable device exists, which is an expected condition on
// machines without a GPU. Unexpected CUDA errors exit via PROBE_CUDA_CHECK.
bool select_device(int index, DeviceInfo& out, std::string& error);

// Writes the device as a JSON object (no surrounding key).
void write_device_json(JsonWriter& w, const DeviceInfo& info);

}  // namespace probe
