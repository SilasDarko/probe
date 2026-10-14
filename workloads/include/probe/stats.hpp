// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace probe {

// Summary statistics of one series of timing samples (milliseconds).
//
// Definitions, chosen to match NumPy so Python can verify them:
//   mean    arithmetic mean
//   p50/p95 linear interpolation between closest ranks (NumPy's default,
//           Hyndman-Fan type 7): for sorted x[0..n-1] the rank is
//           h = (n - 1) * p / 100 and the value is
//           x[floor(h)] + (h - floor(h)) * (x[floor(h) + 1] - x[floor(h)])
//   stddev  sample standard deviation (divides by n - 1); NaN when n < 2,
//           because one sample says nothing about spread
struct SampleStats {
    std::size_t count = 0;
    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double stddev = 0.0;
};

// Percentile of an already sorted, non-empty series, p in [0, 100].
double percentile_linear(const std::vector<double>& sorted, double p);

// Rejects empty input and any value that is NaN, infinite or negative: a
// duration with one of those values means the measurement itself is broken,
// and summarizing it would hide that. Returns false with a message in `error`.
bool compute_stats(const std::vector<double>& samples, SampleStats& out, std::string& error);

}  // namespace probe
