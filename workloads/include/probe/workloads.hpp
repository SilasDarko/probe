// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <string>
#include <vector>

namespace probe {

// Each workload takes the arguments that follow its name and returns an exit
// code from probe/exit_codes.hpp.
int run_vector_add(const std::vector<std::string>& args);

}  // namespace probe
