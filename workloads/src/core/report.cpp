// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/report.hpp"

#include <cstdio>
#include <set>

#include "probe/exit_codes.hpp"
#include "probe/json_writer.hpp"
#include "probe/stats.hpp"

namespace probe {

bool check_phase_names(const std::vector<std::string>& names, std::string& error) {
    std::set<std::string> seen;
    for (const std::string& name : names) {
        if (name.empty() || name == "total") {
            error = "phase name '" + name + "' is empty or reserved";
            return false;
        }
        if (!seen.insert(name).second) {
            error = "duplicate phase name '" + name + "'";
            return false;
        }
    }
    return true;
}

namespace {

void check_series(const std::string& label, const std::vector<double>& samples,
                  std::uint64_t expected_count, std::vector<std::string>& reasons) {
    if (samples.size() != expected_count) {
        reasons.push_back(label + ": expected " + std::to_string(expected_count) +
                          " samples, got " + std::to_string(samples.size()));
        return;
    }
    SampleStats stats;
    std::string error;
    if (!compute_stats(samples, stats, error)) reasons.push_back(label + ": " + error);
}

// A statistics object, or null if the series cannot be summarized (the run
// is then already marked invalid with the reason).
void write_stats(JsonWriter& w, const std::vector<double>& samples) {
    SampleStats s;
    std::string error;
    if (!compute_stats(samples, s, error)) {
        w.null();
        return;
    }
    w.begin_object();
    w.key("count").uinteger(s.count);
    w.key("min").number(s.min);
    w.key("max").number(s.max);
    w.key("mean").number(s.mean);
    w.key("p50").number(s.p50);
    w.key("p95").number(s.p95);
    w.key("stddev").number(s.stddev);  // NaN (one sample) is written as null
    w.end_object();
}

void write_array(JsonWriter& w, const std::vector<double>& values) {
    w.begin_array();
    for (const double v : values) w.number(v);
    w.end_array();
}

}  // namespace

std::vector<std::string> invalid_reasons(const RunReport& r) {
    std::vector<std::string> reasons;
    if (!r.validation.passed()) {
        reasons.push_back("validation failed in " + std::to_string(r.validation.iterations_failed) +
                          " of " + std::to_string(r.validation.iterations_checked) +
                          " checked iterations");
    }
    const std::uint64_t n = r.options.iterations;
    for (const PhaseSeries& p : r.gpu_phases)
        check_series("gpu." + p.name, p.samples_ms, n, reasons);
    check_series("gpu.total", r.gpu_total_ms, n, reasons);
    check_series("host.end_to_end", r.host_end_to_end_ms, n, reasons);
    return reasons;
}

std::string run_report_json(const RunReport& r) {
    const std::vector<std::string> reasons = invalid_reasons(r);
    JsonWriter w;
    w.begin_object();
    w.key("schema").string("probe.run");
    w.key("schema_version").integer(kRunSchemaVersion);
    w.key("workload").string(r.workload);

    w.key("valid").boolean(reasons.empty());
    w.key("invalid_reasons").begin_array();
    for (const std::string& reason : reasons) w.string(reason);
    w.end_array();

    w.key("parameters").begin_object();
    for (const Param& p : r.parameters) {
        w.key(p.key);
        if (const auto* u = std::get_if<std::uint64_t>(&p.value)) {
            w.uinteger(*u);
        } else {
            w.string(std::get<std::string>(p.value));
        }
    }
    w.end_object();

    w.key("config").begin_object();
    w.key("warmup").uinteger(r.options.warmup);
    w.key("iterations").uinteger(r.options.iterations);
    w.key("seed").uinteger(r.options.seed);
    w.end_object();

    w.key("build").begin_object();
    w.key("probe_version").string(r.build.probe_version);
    w.key("build_type").string(r.build.build_type);
    w.key("cuda_architectures").string(r.build.cuda_architectures);
    w.key("cuda_compiler").string(r.build.cuda_compiler);
    w.key("cxx_compiler").string(r.build.cxx_compiler);
    w.key("cuda_flags").string(r.build.cuda_flags);
    w.end_object();

    w.key("environment").begin_object();
    w.key("device");
    write_device_json(w, r.device);
    w.end_object();

    w.key("validation").begin_object();
    w.key("passed").boolean(r.validation.passed());
    w.key("method").string(r.validation_method);
    w.key("rtol").number(r.tolerance.rtol);
    w.key("atol").number(r.tolerance.atol);
    w.key("iterations_checked").uinteger(r.validation.iterations_checked);
    w.key("iterations_failed").uinteger(r.validation.iterations_failed);
    w.key("total_mismatches").uinteger(r.validation.total_mismatches);
    w.key("max_abs_error").number(r.validation.max_abs_error);
    w.key("first_failed_iteration").integer(r.validation.first_failed_iteration);
    w.end_object();

    w.key("timing").begin_object();
    w.key("unit").string("ms");
    w.key("gpu").begin_object();
    w.key("method").string("cuda_events");
    w.key("phase_order").begin_array();
    for (const PhaseSeries& p : r.gpu_phases) w.string(p.name);
    w.end_array();
    w.key("phases").begin_object();
    for (const PhaseSeries& p : r.gpu_phases) {
        w.key(p.name);
        write_stats(w, p.samples_ms);
    }
    w.end_object();
    w.key("total");
    write_stats(w, r.gpu_total_ms);
    w.end_object();
    w.key("host").begin_object();
    w.key("method").string("steady_clock");
    w.key("end_to_end");
    write_stats(w, r.host_end_to_end_ms);
    w.end_object();
    w.end_object();

    // Raw samples only on request: they are what later regression analysis
    // needs, but they make the default output much larger.
    if (r.options.raw_samples) {
        w.key("samples").begin_object();
        w.key("gpu").begin_object();
        for (const PhaseSeries& p : r.gpu_phases) {
            w.key(p.name);
            write_array(w, p.samples_ms);
        }
        w.key("total");
        write_array(w, r.gpu_total_ms);
        w.end_object();
        w.key("host").begin_object();
        w.key("end_to_end");
        write_array(w, r.host_end_to_end_ms);
        w.end_object();
        w.end_object();
    }

    w.end_object();
    return w.str();
}

std::string run_report_summary(const RunReport& r) {
    std::string out = r.workload;
    for (const Param& p : r.parameters) {
        out += "  " + p.key + "=";
        if (const auto* u = std::get_if<std::uint64_t>(&p.value)) {
            out += std::to_string(*u);
        } else {
            out += std::get<std::string>(p.value);
        }
    }
    char line[160];
    std::snprintf(line, sizeof line, "\ndevice %s (cc %d.%d)   warmup %llu   iterations %llu\n",
                  r.device.name.c_str(), r.device.cc_major, r.device.cc_minor,
                  static_cast<unsigned long long>(r.options.warmup),
                  static_cast<unsigned long long>(r.options.iterations));
    out += line;
    out += "                        p50 ms      p95 ms      min ms\n";

    auto row = [&](const std::string& label, const std::vector<double>& samples) {
        SampleStats s;
        std::string error;
        if (compute_stats(samples, s, error)) {
            std::snprintf(line, sizeof line, "  %-18s %10.4f  %10.4f  %10.4f\n", label.c_str(),
                          s.p50, s.p95, s.min);
        } else {
            std::snprintf(line, sizeof line, "  %-18s %s\n", label.c_str(), error.c_str());
        }
        out += line;
    };
    for (const PhaseSeries& p : r.gpu_phases) row("gpu " + p.name, p.samples_ms);
    row("gpu total", r.gpu_total_ms);
    row("host end-to-end", r.host_end_to_end_ms);

    const std::vector<std::string> reasons = invalid_reasons(r);
    if (reasons.empty()) {
        out += "result: VALID (" + std::to_string(r.validation.iterations_checked) +
               " iterations validated, " + r.validation_method + ")\n";
    } else {
        out += "result: INVALID, timings above must not be used\n";
        for (const std::string& reason : reasons) out += "  - " + reason + "\n";
    }
    return out;
}

int emit_run_report(const RunReport& r) {
    if (r.options.json) {
        std::printf("%s\n", run_report_json(r).c_str());
    } else {
        std::fputs(run_report_summary(r).c_str(), stdout);
    }
    return invalid_reasons(r).empty() ? exit_code::kOk : exit_code::kInvalidResult;
}

}  // namespace probe
