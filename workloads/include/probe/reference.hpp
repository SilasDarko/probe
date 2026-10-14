// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <cstdint>

namespace probe {

// Fills `data` with values uniformly distributed in [-1, 1) from a fixed seed,
// so a run's inputs are reproducible on the same machine and standard library.
void fill_uniform(float* data, std::size_t n, std::uint32_t seed);

// CPU reference: out[i] = a[i] + b[i]. Deliberately the most obvious loop; it
// is the ground truth the GPU result is checked against.
void vector_add_reference(const float* a, const float* b, float* out, std::size_t n);

}  // namespace probe
