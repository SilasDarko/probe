// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace probe {

double percentile_linear(const std::vector<double>& sorted, double p) {
    const double h = static_cast<double>(sorted.size() - 1) * p / 100.0;
    const auto lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= sorted.size()) return sorted.back();
    return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[lo + 1] - sorted[lo]);
}

bool compute_stats(const std::vector<double>& samples, SampleStats& out, std::string& error) {
    if (samples.empty()) {
        error = "no samples";
        return false;
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (!std::isfinite(samples[i]) || samples[i] < 0.0) {
            error = "sample " + std::to_string(i) + " is not a finite non-negative duration";
            return false;
        }
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    const auto n = static_cast<double>(sorted.size());
    double sum = 0.0;
    for (const double x : sorted) sum += x;
    const double mean = sum / n;

    // Two passes (mean first, then squared deviations) instead of the
    // one-pass sum-of-squares formula, which loses precision when the spread
    // is small relative to the mean, as it usually is for timings.
    double stddev = std::numeric_limits<double>::quiet_NaN();
    if (sorted.size() > 1) {
        double sq = 0.0;
        for (const double x : sorted) sq += (x - mean) * (x - mean);
        stddev = std::sqrt(sq / (n - 1.0));
    }

    out.count = sorted.size();
    out.min = sorted.front();
    out.max = sorted.back();
    out.mean = mean;
    out.p50 = percentile_linear(sorted, 50.0);
    out.p95 = percentile_linear(sorted, 95.0);
    out.stddev = stddev;
    return true;
}

}  // namespace probe
