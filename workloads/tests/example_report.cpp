// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// Test helper: prints a probe.run JSON document built from synthetic data
// through the same code path probe-workloads uses. The data is fake and
// clearly labeled; it exists only so the schema can be tested without a GPU.
//
//   probe_example_report valid     every series valid, raw samples included
//   probe_example_report invalid   validation failed in one iteration
//   probe_example_report single    one measured iteration (stddev undefined)

#include <cstdio>
#include <cstring>

#include "probe/report.hpp"

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "valid";
    const bool single = std::strcmp(mode, "single") == 0;
    const bool invalid = std::strcmp(mode, "invalid") == 0;
    const std::uint64_t iterations = single ? 1 : 6;

    probe::RunReport r;
    r.workload = "synthetic_example";
    r.parameters = {{"n", std::uint64_t{1024}}, {"dtype", std::string("float32")}};
    r.options.warmup = 2;
    r.options.iterations = iterations;
    r.options.seed = 42;
    r.options.json = true;
    r.options.raw_samples = true;
    r.build = {"0.1.0", "Release", "75-real,90", "NVIDIA 0.0", "GNU 0.0", "-O3"};
    r.device.name = "SYNTHETIC (not a real GPU)";
    r.validation_method = "synthetic";

    for (std::uint64_t i = 0; i < iterations; ++i) {
        probe::ValidationResult v;
        v.checked = 1;
        v.mismatches = (invalid && i == 3) ? 1 : 0;
        r.validation.add(v);
    }
    // Deterministic, obviously artificial values.
    const double h2d[] = {0.5, 0.25, 1.75, 2.0, 0.125, 3.5};
    const double kernel[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const double d2h[] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    r.gpu_phases = {{"h2d", {}}, {"kernel", {}}, {"d2h", {}}};
    for (std::uint64_t i = 0; i < iterations; ++i) {
        r.gpu_phases[0].samples_ms.push_back(h2d[i]);
        r.gpu_phases[1].samples_ms.push_back(kernel[i]);
        r.gpu_phases[2].samples_ms.push_back(d2h[i]);
        r.gpu_total_ms.push_back(h2d[i] + kernel[i] + d2h[i]);
        r.host_end_to_end_ms.push_back(h2d[i] + kernel[i] + d2h[i] + 0.05);
    }

    std::printf("%s\n", probe::run_report_json(r).c_str());
    return 0;
}
