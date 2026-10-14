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

struct VectorAddOptions {
    std::uint64_t n = std::uint64_t{1} << 24;  // 16,777,216 floats = 64 MiB per vector
    std::uint64_t block_size = 256;
    std::uint64_t warmup = 3;
    std::uint64_t iterations = 20;
    std::uint64_t seed = 42;
    bool json = false;
};

// Parses the arguments after "vector_add". On failure returns false and sets
// `error` to a one-line message; `out` is then unspecified.
bool parse_vector_add_options(const std::vector<std::string>& args, VectorAddOptions& out,
                              std::string& error);

}  // namespace probe
