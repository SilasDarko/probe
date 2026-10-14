// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <cstdint>

namespace probe {

// Element i is correct when |actual - expected| <= atol + rtol * |expected|.
// rtol scales with the magnitude of the expected value; atol covers values
// near zero, where a relative bound alone would be too strict.
struct Tolerance {
    double rtol = 0.0;
    double atol = 0.0;
};

// Result of checking one output array against its reference.
struct ValidationResult {
    std::size_t checked = 0;
    std::size_t mismatches = 0;
    double max_abs_error = 0.0;  // largest finite error seen; infinity if any was infinite
    std::int64_t first_mismatch_index = -1;

    bool passed() const { return checked > 0 && mismatches == 0; }
};

// NaN in either array always counts as a mismatch.
ValidationResult validate_close(const float* expected, const float* actual, std::size_t n,
                                Tolerance tol);

// Aggregate over every measured iteration. A run is valid only if at least
// one iteration was checked and none failed: correctness has to be shown,
// not assumed.
struct ValidationSummary {
    std::size_t iterations_checked = 0;
    std::size_t iterations_failed = 0;
    std::size_t total_mismatches = 0;
    double max_abs_error = 0.0;
    std::int64_t first_failed_iteration = -1;

    void add(const ValidationResult& result);
    bool passed() const { return iterations_checked > 0 && iterations_failed == 0; }
};

}  // namespace probe
