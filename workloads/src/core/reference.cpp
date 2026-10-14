// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/reference.hpp"

#include <random>

namespace probe {

void fill_uniform(float* data, std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i) data[i] = dist(rng);
}

void vector_add_reference(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

}  // namespace probe
