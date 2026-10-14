// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace probe {

// Parses a non-negative decimal integer. Rejects empty input, signs,
// whitespace, trailing characters and values that overflow 64 bits.
bool parse_u64(std::string_view text, std::uint64_t& out);

// Options every workload accepts; owned by the benchmark harness.
struct CommonOptions {
    std::uint64_t warmup = 3;
    std::uint64_t iterations = 20;
    std::uint64_t seed = 42;
    bool json = false;
    bool raw_samples = false;  // include every per-iteration sample in the JSON
};

// A workload-specific integer flag, e.g. {"--n", &options.n}.
struct UintFlag {
    std::string_view flag;
    std::uint64_t* target;
};

// Parses the common flags (--warmup, --iters, --seed, --json, --raw-samples)
// plus the workload's own integer flags, and checks the common constraints.
// The workload checks its own values afterwards. On failure returns false with
// a one-line message in `error`.
bool parse_benchmark_args(const std::vector<std::string>& args, CommonOptions& common,
                          const std::vector<UintFlag>& workload_flags, std::string& error);

struct VectorAddOptions {
    CommonOptions common;
    std::uint64_t n = std::uint64_t{1} << 24;  // 16,777,216 floats = 64 MiB per vector
    std::uint64_t block_size = 256;
};

bool parse_vector_add_options(const std::vector<std::string>& args, VectorAddOptions& out,
                              std::string& error);

struct ReductionOptions {
    CommonOptions common;
    std::uint64_t n = std::uint64_t{1} << 24;
    std::uint64_t block_size = 256;
};

bool parse_reduction_options(const std::vector<std::string>& args, ReductionOptions& out,
                             std::string& error);

}  // namespace probe
