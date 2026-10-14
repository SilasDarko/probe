// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/reference.hpp"

#include <cmath>
#include <random>

namespace probe {

void fill_unit_magnitude(float* data, std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    constexpr std::uint32_t kImplicitOne = 1u << 23;  // k in [2^23, 2^24)
    constexpr std::uint32_t kMantissaMask = kImplicitOne - 1;
    const float scale = std::ldexp(1.0f, -24);  // exact power of two
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(rng());
        const std::uint32_t k = kImplicitOne | (bits & kMantissaMask);
        // k < 2^24 fits exactly in a float, and scaling by a power of two is
        // exact, so the value is exactly k * 2^-24, in [0.5, 1).
        const float magnitude = static_cast<float>(k) * scale;
        data[i] = (bits >> 31) != 0 ? -magnitude : magnitude;
    }
}

void vector_add_reference(const float* a, const float* b, float* out, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

ReductionReference reduction_reference(const float* x, std::size_t n) {
    ReductionReference r;
    for (std::size_t i = 0; i < n; ++i) {
        r.sum += static_cast<double>(x[i]);
        r.abs_sum += std::fabs(static_cast<double>(x[i]));
    }
    return r;
}

std::vector<std::uint64_t> reduction_pass_sizes(std::uint64_t n, std::uint64_t block_size) {
    std::vector<std::uint64_t> grids;
    std::uint64_t len = n;
    while (len > 1) {
        len = (len + block_size - 1) / block_size;
        grids.push_back(len);
    }
    return grids;
}

Tolerance reduction_tolerance(std::uint64_t n, std::uint64_t block_size, double abs_sum) {
    const double u_float = std::ldexp(1.0, -24);
    const double u_double = std::ldexp(1.0, -53);
    const double levels_per_pass = std::log2(static_cast<double>(block_size));
    const double depth =
        static_cast<double>(reduction_pass_sizes(n, block_size).size()) * levels_per_pass;
    const double gamma = depth * u_float / (1.0 - depth * u_float);
    const double reference_error = static_cast<double>(n > 0 ? n - 1 : 0) * u_double;
    return Tolerance{0.0, (gamma + reference_error) * abs_sum};
}

}  // namespace probe
