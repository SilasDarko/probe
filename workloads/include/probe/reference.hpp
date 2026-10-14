// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "probe/validate.hpp"

namespace probe {

// --- Input generation ----------------------------------------------------------

// Fills `data` with deterministic values whose magnitude lies in [0.5, 1) and
// whose sign is random. Every value is a finite, *normal* float; zero,
// subnormals, NaN and infinity cannot occur.
//
// Each value is built directly from one 32-bit std::mt19937 output (the
// generator's output sequence is fixed by the C++ standard): bit 31 picks the
// sign and the low 23 bits pick one of 2^23 evenly spaced magnitudes,
// k * 2^-24 for k in [2^23, 2^24). Unlike std::uniform_real_distribution,
// whose algorithm is implementation-defined, this produces the same inputs on
// every compiler and standard library for a given seed.
void fill_unit_magnitude(float* data, std::size_t n, std::uint32_t seed);

// --- vector_add ----------------------------------------------------------------

// CPU reference: out[i] = a[i] + b[i]. Deliberately the most obvious loop; it
// is the ground truth the GPU result is checked against.
void vector_add_reference(const float* a, const float* b, float* out, std::size_t n);

// --- reduction -------------------------------------------------------------------

struct ReductionReference {
    double sum = 0.0;      // sum of x[i], accumulated in double
    double abs_sum = 0.0;  // sum of |x[i]|, used to size the tolerance
};

// CPU reference for a float32 sum, accumulated sequentially in double so it is
// far more accurate than the float32 result it checks.
ReductionReference reduction_reference(const float* x, std::size_t n);

// Grid sizes of each pass of the multi-pass GPU reduction. Each pass turns
// `len` values into ceil(len / block_size) partial sums; passes repeat until
// one value remains. Examples with block_size 256:
//   n = 1          -> {}            (already one value, no kernel launch)
//   n = 1,000,003  -> {3907, 16, 1}
std::vector<std::uint64_t> reduction_pass_sizes(std::uint64_t n, std::uint64_t block_size);

// Absolute error bound for the float32 GPU result against the double
// reference (returned as a Tolerance with rtol = 0).
//
// Every element reaches the final sum through at most d additions, where
// d = passes * log2(block_size). A standard floating-point result bounds the
// rounding error of such a summation tree by
//   gamma_d * sum|x|,   gamma_d = d*u / (1 - d*u),   u = 2^-24 (float32).
// The double reference contributes at most (n - 1) * 2^-53 * sum|x| more.
// This is a worst-case bound, not an estimate: typical errors are much
// smaller, and any error above it is a bug, not rounding.
Tolerance reduction_tolerance(std::uint64_t n, std::uint64_t block_size, double abs_sum);

}  // namespace probe
