// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// The benchmark result: what was run, how, on what, whether it was correct,
// and how long it took. This is the one place the result schema is defined;
// the baseline and comparison layers depend on it.
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/device_info.hpp"
#include "probe/validate.hpp"

namespace probe {

// Bump when the JSON layout changes incompatibly. Version 1 was the
// pre-harness vector_add output and was never persisted.
inline constexpr int kRunSchemaVersion = 2;

// A workload parameter, e.g. {"n", 1048576} or {"dtype", "float32"}.
struct Param {
    std::string key;
    std::variant<std::uint64_t, std::string> value;
};

struct BuildInfo {
    std::string probe_version;
    std::string build_type;
    std::string cuda_architectures;
    std::string cuda_compiler;
    std::string cxx_compiler;
    std::string cuda_flags;
};

// Per-iteration durations of one timed GPU phase, in stream order.
struct PhaseSeries {
    std::string name;
    std::vector<double> samples_ms;
};

struct RunReport {
    std::string workload;
    std::vector<Param> parameters;
    CommonOptions options;
    BuildInfo build;
    DeviceInfo device;

    std::string validation_method;
    Tolerance tolerance;
    ValidationSummary validation;

    std::vector<PhaseSeries> gpu_phases;     // CUDA-event time of each phase
    std::vector<double> gpu_total_ms;        // CUDA-event time, first to last event
    std::vector<double> host_end_to_end_ms;  // CPU wall clock around the whole iteration
};

// Phase names must be non-empty, unique, and not "total" (reserved).
bool check_phase_names(const std::vector<std::string>& names, std::string& error);

// Empty when the run is valid. Otherwise one human-readable reason per
// problem: failed validation, or a timing series that is not exactly
// `iterations` finite, non-negative samples.
std::vector<std::string> invalid_reasons(const RunReport& report);

std::string run_report_json(const RunReport& report);

std::string run_report_summary(const RunReport& report);

// Prints the JSON or the human summary to stdout and returns the exit code.
int emit_run_report(const RunReport& report);

}  // namespace probe
