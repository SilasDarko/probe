// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <string>

#include "probe/device_info.hpp"

namespace probe {

// Selects device `index` and fills `out`. Returns false with a message in
// `error` when no usable device exists, which is an expected condition on
// machines without a GPU. Unexpected CUDA errors exit via PROBE_CUDA_CHECK.
bool select_device(int index, DeviceInfo& out, std::string& error);

}  // namespace probe
