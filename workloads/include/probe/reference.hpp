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
    double abs_sum = 0.0;  // sum of |x[i]|, accumulated in double; sizes the tolerance
};

// CPU reference for the reduction. Every float32 input converts to double
// exactly; the sum is then accumulated sequentially in double (53-bit
// significand vs float32's 24), so its own rounding error is about 2^29 times
// smaller per operation than the GPU's. The GPU result is compared against
// this higher-precision value, not against another float32 sum.
// See docs/correctness.md for the full argument.
ReductionReference reduction_reference(const float* x, std::size_t n);

// Grid sizes of each pass of the multi-pass GPU reduction. Each pass turns
// `len` values into ceil(len / block_size) partial sums; passes repeat until
// one value remains. Examples with block_size 256:
//   n = 1          -> {}            (already one value, no kernel launch)
//   n = 1,000,003  -> {3907, 16, 1}
std::vector<std::uint64_t> reduction_pass_sizes(std::uint64_t n, std::uint64_t block_size);

// Absolute error bound for |gpu_result - reference.sum| (returned as a
// Tolerance with rtol = 0). With s the exact sum, gamma(k, u) = k*u / (1 - k*u):
//
//   |gpu - s| <= gamma(d, 2^-24)   * sum|x|   GPU float32 tree, depth d
//   |ref - s| <= gamma(n-1, 2^-53) * sum|x|   sequential double reference
//
// so by the triangle inequality
//   |gpu - ref| <= (gamma(d, 2^-24) + gamma(n-1, 2^-53)) * sum|x|.
//
// d = passes * log2(block_size) bounds how many additions any one element
// passes through. sum|x| is itself computed in double and may be low by a
// relative gamma(n-1, 2^-53), so the result is scaled up by that factor.
// This is a worst-case bound, not an estimate: any error above it is a bug.
// Assumptions and limits are documented in docs/correctness.md.
Tolerance reduction_tolerance(std::uint64_t n, std::uint64_t block_size, double abs_sum);

}  // namespace probe
