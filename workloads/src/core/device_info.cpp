// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/device_info.hpp"

namespace probe {

std::string format_cuda_version(int v) {
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

void write_device_json(JsonWriter& w, const DeviceInfo& info) {
    w.begin_object();
    w.key("index").integer(info.index);
    w.key("count").integer(info.count);
    w.key("name").string(info.name);
    w.key("compute_capability")
        .string(std::to_string(info.cc_major) + "." + std::to_string(info.cc_minor));
    w.key("sm_count").integer(info.sm_count);
    w.key("global_mem_bytes").uinteger(info.global_mem_bytes);
    w.key("cuda_runtime_version").string(format_cuda_version(info.runtime_version));
    w.key("cuda_driver_version").string(format_cuda_version(info.driver_version));
    w.end_object();
}

}  // namespace probe
