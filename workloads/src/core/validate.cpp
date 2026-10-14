// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/validate.hpp"

#include <cmath>

namespace probe {

bool within_tolerance(double expected, double actual, Tolerance tol) {
    const double err = std::fabs(actual - expected);
    const double limit = tol.atol + tol.rtol * std::fabs(expected);
    return err <= limit;  // false for NaN; callers must not rewrite this as !(err > limit)
}

namespace {

void record(ValidationResult& result, std::size_t index, double expected, double actual,
            Tolerance tol) {
    if (!within_tolerance(expected, actual, tol)) {
        if (result.first_mismatch_index < 0) {
            result.first_mismatch_index = static_cast<std::int64_t>(index);
        }
        ++result.mismatches;
    }
    const double err = std::fabs(actual - expected);
    if (err > result.max_abs_error) result.max_abs_error = err;
}

}  // namespace

ValidationResult validate_close(const float* expected, const float* actual, std::size_t n,
                                Tolerance tol) {
    ValidationResult result;
    result.checked = n;
    for (std::size_t i = 0; i < n; ++i) record(result, i, expected[i], actual[i], tol);
    return result;
}

ValidationResult validate_scalar(double expected, double actual, Tolerance tol) {
    ValidationResult result;
    result.checked = 1;
    record(result, 0, expected, actual, tol);
    return result;
}

void ValidationSummary::add(const ValidationResult& result) {
    if (!result.passed() && first_failed_iteration < 0) {
        first_failed_iteration = static_cast<std::int64_t>(iterations_checked);
    }
    ++iterations_checked;
    if (!result.passed()) ++iterations_failed;
    total_mismatches += result.mismatches;
    if (result.max_abs_error > max_abs_error) max_abs_error = result.max_abs_error;
}

}  // namespace probe
