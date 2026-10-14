// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/validate.hpp"

#include <cmath>

namespace probe {

ValidationResult validate_close(const float* expected, const float* actual, std::size_t n,
                                Tolerance tol) {
    ValidationResult result;
    result.checked = n;
    for (std::size_t i = 0; i < n; ++i) {
        const double e = expected[i];
        const double a = actual[i];
        const double err = std::fabs(a - e);
        const double limit = tol.atol + tol.rtol * std::fabs(e);
        // Written as !(err <= limit) rather than (err > limit): every
        // comparison with NaN is false, so this form flags NaN as a mismatch
        // instead of silently accepting it.
        if (!(err <= limit)) {
            if (result.first_mismatch_index < 0) {
                result.first_mismatch_index = static_cast<std::int64_t>(i);
            }
            ++result.mismatches;
        }
        if (err > result.max_abs_error) result.max_abs_error = err;
    }
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
