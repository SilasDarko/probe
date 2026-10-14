// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// Unit tests for probe_core. No GPU is involved; these run on any machine via
// ctest. Usage: probe_core_tests <path to tests/fixtures/stats_golden.txt>
//
// assert() is compiled out in Release builds (NDEBUG), so this file uses its
// own CHECK macro that always runs and counts failures.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/json_writer.hpp"
#include "probe/reference.hpp"
#include "probe/report.hpp"
#include "probe/stats.hpp"
#include "probe/validate.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                       \
    do {                                                                                  \
        if (!(cond)) {                                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                                 \
        }                                                                                 \
    } while (0)

using probe::Tolerance;
using probe::ValidationSummary;

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// --- argument parsing -----------------------------------------------------------

void test_parse_u64() {
    std::uint64_t v = 0;
    CHECK(probe::parse_u64("0", v) && v == 0);
    CHECK(probe::parse_u64("1048576", v) && v == 1048576);
    CHECK(probe::parse_u64("18446744073709551615", v) && v == 18446744073709551615ull);

    CHECK(!probe::parse_u64("", v));
    CHECK(!probe::parse_u64("-1", v));
    CHECK(!probe::parse_u64("+1", v));
    CHECK(!probe::parse_u64(" 1", v));
    CHECK(!probe::parse_u64("12abc", v));
    CHECK(!probe::parse_u64("1e6", v));
    CHECK(!probe::parse_u64("18446744073709551616", v));  // 2^64 overflows
}

bool parse_common(const std::vector<std::string>& args, probe::CommonOptions& c, std::uint64_t& n) {
    std::string error;
    c = probe::CommonOptions{};
    return probe::parse_benchmark_args(args, c, {{"--n", &n}}, error);
}

void test_parse_benchmark_args() {
    probe::CommonOptions c;
    std::uint64_t n = 7;
    CHECK(parse_common({}, c, n));
    CHECK(c.warmup == 3 && c.iterations == 20 && c.seed == 42 && !c.json && !c.raw_samples);
    CHECK(n == 7);  // untouched workload flag keeps its default

    CHECK(parse_common(
        {"--warmup", "0", "--iters", "5", "--seed", "9", "--json", "--raw-samples", "--n", "123"},
        c, n));
    CHECK(c.warmup == 0 && c.iterations == 5 && c.seed == 9 && c.json && c.raw_samples);
    CHECK(n == 123);  // routed to the workload's flag

    CHECK(!parse_common({"--block-size", "64"}, c, n));  // not registered by this workload
    CHECK(!parse_common({"--iters"}, c, n));             // missing value
    CHECK(!parse_common({"--iters", "0"}, c, n));        // no measurements is not a benchmark
    CHECK(!parse_common({"--iters", "100001"}, c, n));
    CHECK(!parse_common({"--warmup", "100001"}, c, n));
    CHECK(!parse_common({"--seed", "4294967296"}, c, n));  // does not fit in 32 bits
    CHECK(!parse_common({"--json", "extra"}, c, n));       // stray positional argument

    std::string error;
    probe::CommonOptions c2;
    CHECK(!probe::parse_benchmark_args({"--bogus"}, c2, {}, error));
    CHECK(error == "unknown option: --bogus");
}

bool parse_va(const std::vector<std::string>& args, probe::VectorAddOptions& out) {
    std::string error;
    return probe::parse_vector_add_options(args, out, error);
}

void test_vector_add_options() {
    probe::VectorAddOptions o;
    CHECK(parse_va({}, o));
    CHECK(o.n == (1u << 24) && o.block_size == 256);

    CHECK(parse_va({"--n", "1000003", "--block-size", "96", "--iters", "5"}, o));
    CHECK(o.n == 1000003 && o.block_size == 96 && o.common.iterations == 5);

    CHECK(!parse_va({"--n", "0"}, o));
    CHECK(!parse_va({"--block-size", "100"}, o));   // not a multiple of 32
    CHECK(!parse_va({"--block-size", "2048"}, o));  // above CUDA's 1024 limit
    // 2^31 blocks of 32 threads is one block more than the grid allows.
    CHECK(!parse_va({"--n", "68719476736", "--block-size", "32"}, o));
    CHECK(parse_va({"--n", "68719476704", "--block-size", "32"}, o));  // exactly 2^31 - 1
}

bool parse_red(const std::vector<std::string>& args, probe::ReductionOptions& out) {
    std::string error;
    return probe::parse_reduction_options(args, out, error);
}

void test_reduction_options() {
    probe::ReductionOptions o;
    CHECK(parse_red({}, o));
    CHECK(o.n == (1u << 24) && o.block_size == 256);
    CHECK(parse_red({"--n", "1000003", "--block-size", "128"}, o));
    CHECK(!parse_red({"--block-size", "96"}, o));  // multiple of 32, not a power of two
    CHECK(!parse_red({"--n", "0"}, o));
    CHECK(!parse_red({"--iters", "0"}, o));
}

// --- input generation ------------------------------------------------------------

void test_generator_domain() {
    std::vector<float> x(200000);
    probe::fill_unit_magnitude(x.data(), x.size(), 42);
    std::size_t outside = 0, not_normal = 0, negative = 0;
    for (const float v : x) {
        if (!std::isnormal(v)) ++not_normal;  // excludes zero, subnormal, inf, NaN
        if (!(std::fabs(v) >= 0.5f && std::fabs(v) < 1.0f)) ++outside;
        if (v < 0.0f) ++negative;
    }
    CHECK(not_normal == 0);
    CHECK(outside == 0);
    // Signs come from one bit of each draw; expect roughly half negative.
    CHECK(negative > 95000 && negative < 105000);
}

void test_generator_deterministic() {
    std::vector<float> x(1000), y(1000), z(1000);
    probe::fill_unit_magnitude(x.data(), x.size(), 42);
    probe::fill_unit_magnitude(y.data(), y.size(), 42);
    probe::fill_unit_magnitude(z.data(), z.size(), 43);
    CHECK(x == y);
    CHECK(x != z);

    // The C++ standard fixes std::mt19937's output sequence, so the first
    // value for a given seed is the same on every platform. First output of
    // mt19937 seeded with 42 is 1608637542; bit 31 is clear (positive) and
    // the low 23 bits select the magnitude.
    const std::uint32_t first = 1608637542u;
    const float expected = static_cast<float>((1u << 23) | (first & 0x7FFFFFu)) * 0x1p-24f;
    CHECK(x[0] == expected);
}

// --- vector_add reference ----------------------------------------------------------

void test_vector_add_reference() {
    const float a[] = {1.0f, -2.5f, 0.0f, 1e30f};
    const float b[] = {2.0f, 2.5f, -0.0f, 1e30f};
    float out[4] = {};
    probe::vector_add_reference(a, b, out, 4);
    CHECK(out[0] == 3.0f);
    CHECK(out[1] == 0.0f);
    CHECK(out[2] == 0.0f);
    CHECK(out[3] == 2e30f);
}

void test_vector_add_sums_stay_out_of_subnormal_range() {
    // The exact-equality argument for vector_add relies on this: with inputs
    // from fill_unit_magnitude, a + b is either exactly zero or at least 2^-24
    // in magnitude, never subnormal.
    const std::size_t n = 200000;
    std::vector<float> a(n), b(n), c(n);
    probe::fill_unit_magnitude(a.data(), n, 1);
    probe::fill_unit_magnitude(b.data(), n, 2);
    probe::vector_add_reference(a.data(), b.data(), c.data(), n);
    std::size_t bad = 0;
    for (const float v : c) {
        if (!(v == 0.0f || (std::isnormal(v) && std::fabs(v) >= 0x1p-24f))) ++bad;
    }
    CHECK(bad == 0);
}

// --- reduction reference, plan and tolerance -----------------------------------------

void test_reduction_reference() {
    const float x[] = {0.5f, -0.75f, 0.875f, -0.5f};
    const auto r = probe::reduction_reference(x, 4);
    CHECK(r.sum == 0.125);
    CHECK(r.abs_sum == 2.625);
}

void test_reduction_pass_sizes() {
    using V = std::vector<std::uint64_t>;
    CHECK(probe::reduction_pass_sizes(1, 256) == V{});
    CHECK(probe::reduction_pass_sizes(2, 256) == (V{1}));
    CHECK(probe::reduction_pass_sizes(256, 256) == (V{1}));
    CHECK(probe::reduction_pass_sizes(257, 256) == (V{2, 1}));
    CHECK(probe::reduction_pass_sizes(1000003, 256) == (V{3907, 16, 1}));
    CHECK(probe::reduction_pass_sizes(1u << 24, 256) == (V{65536, 256, 1}));
    CHECK(probe::reduction_pass_sizes(1000003, 1024) == (V{977, 1}));
}

void test_reduction_tolerance_formula() {
    // n = 1: no additions, so the only error budget is zero.
    CHECK(probe::reduction_tolerance(1, 256, 0.75).atol == 0.0);

    // n = 1000003, block 256: 3 passes * 8 levels = 24 additions deep.
    const double abs_sum = 750000.0;
    const double d = 24.0, u = 0x1p-24;
    const double expected = (d * u / (1.0 - d * u) + 1000002.0 * 0x1p-53) * abs_sum;
    const Tolerance t = probe::reduction_tolerance(1000003, 256, abs_sum);
    CHECK(t.rtol == 0.0);
    CHECK(std::fabs(t.atol - expected) <= 1e-12 * expected);
}

// A CPU model of the GPU kernel: the same tree, in the same order, in float32.
// It lets us check the tolerance against the actual algorithm's rounding.
float simulate_gpu_reduction(std::vector<float> data, std::uint64_t block) {
    while (data.size() > 1) {
        std::vector<float> partial((data.size() + block - 1) / block);
        for (std::size_t b = 0; b < partial.size(); ++b) {
            std::vector<float> s(block, 0.0f);
            for (std::size_t t = 0; t < block; ++t) {
                const std::size_t i = b * block + t;
                if (i < data.size()) s[t] = data[i];
            }
            for (std::size_t stride = block / 2; stride > 0; stride /= 2) {
                for (std::size_t t = 0; t < stride; ++t) s[t] += s[t + stride];
            }
            partial[b] = s[0];
        }
        data = std::move(partial);
    }
    return data[0];
}

void test_reduction_tolerance_against_algorithm() {
    // Awkward sizes: primes and non-powers-of-two, so the last block is
    // partial on every pass. `must_detect_drop` is fixed here, not derived
    // from the tolerance under test: for these sizes the bound is below 0.5,
    // the smallest possible element magnitude, so losing one element has to
    // be caught. For n = 1,000,003 the bound (~1.1) is larger than one
    // element, so a single dropped element is not guaranteed to be detected.
    struct Case {
        std::uint64_t n;
        bool must_detect_drop;
    };
    const Case cases[] = {{1, false},   {2, true},    {3, true},     {255, true},     {257, true},
                          {1000, true}, {4097, true}, {65537, true}, {1000003, false}};
    for (const Case& c : cases) {
        std::vector<float> x(c.n);
        probe::fill_unit_magnitude(x.data(), c.n, 42);
        const auto ref = probe::reduction_reference(x.data(), c.n);
        const Tolerance tol = probe::reduction_tolerance(c.n, 256, ref.abs_sum);

        // The correct algorithm must pass...
        const float gpu_like = simulate_gpu_reduction(x, 256);
        CHECK(probe::validate_scalar(ref.sum, gpu_like, tol).passed());

        // ...and a tail-handling bug (last element dropped) must fail.
        if (c.must_detect_drop) {
            std::vector<float> dropped(x.begin(), x.end() - 1);
            const float buggy = simulate_gpu_reduction(dropped, 256);
            CHECK(!probe::validate_scalar(ref.sum, buggy, tol).passed());
        }
    }
}

// --- validation ----------------------------------------------------------------------

void test_validate_exact_match() {
    const float e[] = {1.0f, 2.0f, 3.0f};
    const auto r = probe::validate_close(e, e, 3, Tolerance{});
    CHECK(r.passed());
    CHECK(r.checked == 3 && r.mismatches == 0 && r.first_mismatch_index == -1);
    CHECK(r.max_abs_error == 0.0);
}

void test_validate_reports_mismatches() {
    const float e[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float a[] = {1.0f, 2.5f, 3.0f, 3.0f};
    const auto r = probe::validate_close(e, a, 4, Tolerance{});
    CHECK(!r.passed());
    CHECK(r.mismatches == 2);
    CHECK(r.first_mismatch_index == 1);
    CHECK(r.max_abs_error == 1.0);
}

void test_validate_tolerance_boundaries() {
    const float e[] = {100.0f};
    const float within[] = {100.5f};  // error 0.5 == atol 0 + rtol 0.005 * 100
    const float outside[] = {100.75f};
    const Tolerance tol{0.005, 0.0};
    CHECK(probe::validate_close(e, within, 1, tol).passed());
    CHECK(!probe::validate_close(e, outside, 1, tol).passed());

    // Near zero a relative bound is useless; atol takes over.
    const float zero[] = {0.0f};
    const float tiny[] = {1e-7f};
    CHECK(!probe::validate_close(zero, tiny, 1, Tolerance{1e-3, 0.0}).passed());
    CHECK(probe::validate_close(zero, tiny, 1, Tolerance{1e-3, 1e-6}).passed());
}

void test_validate_non_finite() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float e[] = {1.0f, 1.0f};
    const float with_nan[] = {1.0f, nan};
    const float with_inf[] = {inf, 1.0f};
    const Tolerance loose{1.0, 1.0};  // even a very loose tolerance must reject these

    const auto r_nan = probe::validate_close(e, with_nan, 2, loose);
    CHECK(!r_nan.passed() && r_nan.first_mismatch_index == 1);

    const auto r_inf = probe::validate_close(e, with_inf, 2, loose);
    CHECK(!r_inf.passed() && std::isinf(r_inf.max_abs_error));
}

void test_validate_scalar() {
    CHECK(probe::validate_scalar(10.0, 10.25, Tolerance{0.0, 0.25}).passed());
    CHECK(!probe::validate_scalar(10.0, 10.5, Tolerance{0.0, 0.25}).passed());
    // The reduction resets its host result to NaN; a missing copy must fail.
    CHECK(!probe::validate_scalar(10.0, std::nan(""), Tolerance{0.0, 1e9}).passed());
}

void test_validate_empty_is_not_a_pass() {
    const auto r = probe::validate_close(nullptr, nullptr, 0, Tolerance{});
    CHECK(!r.passed());
}

void test_summary() {
    ValidationSummary none;
    CHECK(!none.passed());  // nothing checked is not a pass

    const float e[] = {1.0f, 2.0f};
    const float bad[] = {1.0f, 9.0f};
    ValidationSummary s;
    s.add(probe::validate_close(e, e, 2, Tolerance{}));
    s.add(probe::validate_close(e, bad, 2, Tolerance{}));
    s.add(probe::validate_close(e, e, 2, Tolerance{}));
    CHECK(!s.passed());
    CHECK(s.iterations_checked == 3);
    CHECK(s.iterations_failed == 1);
    CHECK(s.first_failed_iteration == 1);
    CHECK(s.total_mismatches == 1);
    CHECK(s.max_abs_error == 7.0);

    ValidationSummary ok;
    ok.add(probe::validate_close(e, e, 2, Tolerance{}));
    CHECK(ok.passed());
}

// --- statistics -------------------------------------------------------------------------

bool close(double expected, double actual) {
    if (std::isnan(expected)) return std::isnan(actual);
    return std::fabs(actual - expected) <= 1e-12 * std::max(1.0, std::fabs(expected));
}

void test_stats_golden(const char* path) {
    std::ifstream in(path);
    CHECK(in.good());
    std::string line;
    int cases = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto bar = line.find('|');
        std::istringstream samples_in(line.substr(0, bar));
        std::istringstream expected_in(line.substr(bar + 1));
        std::vector<double> samples;
        for (std::string tok; samples_in >> tok;)
            samples.push_back(std::strtod(tok.c_str(), nullptr));
        std::vector<double> e;
        for (std::string tok; expected_in >> tok;) e.push_back(std::strtod(tok.c_str(), nullptr));
        CHECK(e.size() == 7);
        if (e.size() != 7) continue;

        probe::SampleStats s;
        std::string error;
        CHECK(probe::compute_stats(samples, s, error));
        CHECK(static_cast<double>(s.count) == e[0]);
        CHECK(close(e[1], s.min));
        CHECK(close(e[2], s.max));
        CHECK(close(e[3], s.mean));
        CHECK(close(e[4], s.p50));
        CHECK(close(e[5], s.p95));
        if (!close(e[6], s.stddev)) {
            std::fprintf(stderr, "  golden case: %s\n", line.c_str());
        }
        CHECK(close(e[6], s.stddev));
        ++cases;
    }
    CHECK(cases >= 5);
}

void test_stats_rejects_bad_input() {
    probe::SampleStats s;
    std::string error;
    CHECK(!probe::compute_stats({}, s, error));
    CHECK(error == "no samples");
    CHECK(!probe::compute_stats({1.0, std::nan("")}, s, error));
    CHECK(!probe::compute_stats({1.0, std::numeric_limits<double>::infinity()}, s, error));
    CHECK(!probe::compute_stats({1.0, -0.5}, s, error));  // a negative duration
    CHECK(error == "sample 1 is not a finite non-negative duration");
}

void test_stats_single_sample() {
    probe::SampleStats s;
    std::string error;
    CHECK(probe::compute_stats({2.5}, s, error));
    CHECK(s.count == 1 && s.min == 2.5 && s.max == 2.5 && s.mean == 2.5);
    CHECK(s.p50 == 2.5 && s.p95 == 2.5);
    CHECK(std::isnan(s.stddev));  // spread is undefined with one sample
}

void test_percentile_definition() {
    const std::vector<double> x = {10.0, 20.0, 30.0, 40.0, 50.0};
    CHECK(probe::percentile_linear(x, 0.0) == 10.0);
    CHECK(probe::percentile_linear(x, 100.0) == 50.0);
    CHECK(probe::percentile_linear(x, 50.0) == 30.0);
    // rank h = 4 * 0.95 = 3.8 -> 40 + 0.8 * (50 - 40) = 48
    CHECK(close(48.0, probe::percentile_linear(x, 95.0)));
    // rank h = 4 * 0.10 = 0.4 -> 10 + 0.4 * 10 = 14
    CHECK(close(14.0, probe::percentile_linear(x, 10.0)));

    probe::SampleStats a, b;
    std::string error;
    probe::compute_stats({5.0, 1.0, 4.0, 2.0, 3.0}, a, error);
    probe::compute_stats({1.0, 2.0, 3.0, 4.0, 5.0}, b, error);
    CHECK(a.p50 == b.p50 && a.p95 == b.p95);  // input order does not matter
}

// --- result report -----------------------------------------------------------------------

probe::RunReport make_report(std::uint64_t iterations) {
    probe::RunReport r;
    r.workload = "test";
    r.parameters = {{"n", std::uint64_t{8}}, {"dtype", std::string("float32")}};
    r.options.iterations = iterations;
    r.validation_method = "test";
    r.gpu_phases = {{"kernel", {}}};
    for (std::uint64_t i = 0; i < iterations; ++i) {
        probe::ValidationResult v;
        v.checked = 1;
        r.validation.add(v);
        r.gpu_phases[0].samples_ms.push_back(1.0 + static_cast<double>(i));
        r.gpu_total_ms.push_back(1.5 + static_cast<double>(i));
        r.host_end_to_end_ms.push_back(2.0 + static_cast<double>(i));
    }
    return r;
}

void test_check_phase_names() {
    std::string error;
    CHECK(probe::check_phase_names({"h2d", "kernel", "d2h"}, error));
    CHECK(!probe::check_phase_names({"h2d", "h2d"}, error));
    CHECK(!probe::check_phase_names({"total"}, error));
    CHECK(!probe::check_phase_names({""}, error));
}

void test_report_valid() {
    const probe::RunReport r = make_report(3);
    CHECK(probe::invalid_reasons(r).empty());
    const std::string json = probe::run_report_json(r);
    CHECK(contains(json, R"("schema":"probe.run","schema_version":2,)"));
    CHECK(contains(json, R"("valid":true,"invalid_reasons":[])"));
    CHECK(contains(json, R"("phase_order":["kernel"])"));
    CHECK(contains(
        json, R"("kernel":{"count":3,"min":1,"max":3,"mean":2,"p50":2,"p95":2.9,"stddev":1})"));
    CHECK(!contains(json, R"("samples")"));  // raw samples only on request

    probe::RunReport with_raw = r;
    with_raw.options.raw_samples = true;
    CHECK(contains(probe::run_report_json(with_raw), R"("samples":{"gpu":{"kernel":[1,2,3],)"));
}

void test_report_invalid() {
    probe::RunReport failed = make_report(3);
    probe::ValidationResult bad;
    bad.checked = 1;
    bad.mismatches = 1;
    failed.validation.add(bad);
    CHECK(probe::invalid_reasons(failed).size() == 1);
    CHECK(contains(probe::run_report_json(failed), R"("valid":false)"));

    probe::RunReport short_series = make_report(3);
    short_series.gpu_total_ms.pop_back();
    const auto reasons = probe::invalid_reasons(short_series);
    CHECK(reasons.size() == 1 && reasons[0] == "gpu.total: expected 3 samples, got 2");
    CHECK(contains(probe::run_report_json(short_series), R"("valid":false)"));

    // A series that cannot be summarized at all is written as null.
    probe::RunReport nan_sample = make_report(3);
    nan_sample.host_end_to_end_ms[1] = std::nan("");
    CHECK(probe::invalid_reasons(nan_sample).size() == 1);
    CHECK(contains(probe::run_report_json(nan_sample), R"("end_to_end":null)"));
}

void test_report_single_iteration() {
    const std::string json = probe::run_report_json(make_report(1));
    CHECK(contains(json, R"("stddev":null)"));
    CHECK(contains(json, R"("valid":true)"));
}

// --- JSON writer --------------------------------------------------------------------------

void test_json_structure() {
    probe::JsonWriter w;
    w.begin_object()
        .key("a")
        .integer(-3)
        .key("b")
        .begin_array()
        .uinteger(1)
        .boolean(true)
        .null()
        .begin_object()
        .end_object()
        .end_array()
        .key("c")
        .string("x")
        .end_object();
    CHECK(w.str() == R"({"a":-3,"b":[1,true,null,{}],"c":"x"})");

    probe::JsonWriter empty;
    empty.begin_array().end_array();
    CHECK(empty.str() == "[]");
}

void test_json_escaping() {
    probe::JsonWriter w;
    w.string("say \"hi\"\\\n\t\x01");
    CHECK(w.str() == R"("say \"hi\"\\\n\t\u0001")");
}

void test_json_numbers() {
    probe::JsonWriter w;
    w.begin_array()
        .number(1.5)
        .number(0.1)
        .number(0.1f)  // the double nearest the float 0.1f, printed exactly
        .number(1.0 / 3.0)
        .number(1e-7)
        .number(std::numeric_limits<double>::quiet_NaN())
        .number(std::numeric_limits<double>::infinity())
        .end_array();
    CHECK(w.str() == "[1.5,0.1,0.10000000149011612,0.3333333333333333,1e-07,null,null]");

    // Round trip: what is written parses back to exactly the same double.
    const double values[] = {1.3541666666666667, 2.4350000000000005, 0.29752664630778514};
    for (const double v : values) {
        probe::JsonWriter one;
        one.number(v);
        CHECK(std::strtod(one.str().c_str(), nullptr) == v);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <stats_golden.txt>\n", argv[0]);
        return 2;
    }

    test_parse_u64();
    test_parse_benchmark_args();
    test_vector_add_options();
    test_reduction_options();
    test_generator_domain();
    test_generator_deterministic();
    test_vector_add_reference();
    test_vector_add_sums_stay_out_of_subnormal_range();
    test_reduction_reference();
    test_reduction_pass_sizes();
    test_reduction_tolerance_formula();
    test_reduction_tolerance_against_algorithm();
    test_validate_exact_match();
    test_validate_reports_mismatches();
    test_validate_tolerance_boundaries();
    test_validate_non_finite();
    test_validate_scalar();
    test_validate_empty_is_not_a_pass();
    test_summary();
    test_stats_golden(argv[1]);
    test_stats_rejects_bad_input();
    test_stats_single_sample();
    test_percentile_definition();
    test_check_phase_names();
    test_report_valid();
    test_report_invalid();
    test_report_single_iteration();
    test_json_structure();
    test_json_escaping();
    test_json_numbers();

    if (g_failures == 0) {
        std::printf("probe_core_tests: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "probe_core_tests: %d check(s) failed\n", g_failures);
    return 1;
}
