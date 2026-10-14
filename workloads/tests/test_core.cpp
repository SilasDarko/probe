// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// Unit tests for probe_core: argument parsing, CPU reference, validation and
// JSON output. No GPU is involved; these run on any machine via ctest.
//
// assert() is compiled out in Release builds (NDEBUG), so this file uses its
// own CHECK macro that always runs and counts failures.

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/json_writer.hpp"
#include "probe/reference.hpp"
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

// --- parse_u64 -------------------------------------------------------------

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

// --- parse_vector_add_options ----------------------------------------------

bool parse(const std::vector<std::string>& args, probe::VectorAddOptions& out) {
    std::string error;
    return probe::parse_vector_add_options(args, out, error);
}

void test_vector_add_options() {
    probe::VectorAddOptions o;
    CHECK(parse({}, o));
    CHECK(o.n == (1u << 24) && o.block_size == 256 && o.warmup == 3 && o.iterations == 20);
    CHECK(!o.json);

    CHECK(parse({"--n", "1000003", "--block-size", "128", "--warmup", "0", "--iters", "5", "--seed",
                 "7", "--json"},
                o));
    CHECK(o.n == 1000003 && o.block_size == 128 && o.warmup == 0 && o.iterations == 5);
    CHECK(o.seed == 7 && o.json);

    CHECK(!parse({"--n", "0"}, o));
    CHECK(!parse({"--n"}, o));                   // missing value
    CHECK(!parse({"--n", "abc"}, o));            // not a number
    CHECK(!parse({"--iters", "0"}, o));          // no measurements is not a benchmark
    CHECK(!parse({"--block-size", "100"}, o));   // not a multiple of 32
    CHECK(!parse({"--block-size", "2048"}, o));  // above CUDA's 1024 limit
    CHECK(!parse({"--seed", "4294967296"}, o));  // does not fit in 32 bits
    CHECK(!parse({"--bogus"}, o));
    // 2^31 blocks of 32 threads is one block more than the grid allows.
    CHECK(!parse({"--n", "68719476736", "--block-size", "32"}, o));
    CHECK(parse({"--n", "68719476704", "--block-size", "32"}, o));  // exactly 2^31 - 1 blocks
}

// --- reference ---------------------------------------------------------------

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

void test_fill_uniform() {
    std::vector<float> x(1000), y(1000), z(1000);
    probe::fill_uniform(x.data(), x.size(), 42);
    probe::fill_uniform(y.data(), y.size(), 42);
    probe::fill_uniform(z.data(), z.size(), 43);
    CHECK(x == y);  // same seed, same data
    CHECK(x != z);  // different seed, different data
    bool in_range = true;
    for (const float v : x) in_range = in_range && v >= -1.0f && v < 1.0f;
    CHECK(in_range);
}

// --- validate_close ----------------------------------------------------------

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

void test_validate_empty_is_not_a_pass() {
    const auto r = probe::validate_close(nullptr, nullptr, 0, Tolerance{});
    CHECK(!r.passed());
}

// --- ValidationSummary --------------------------------------------------------

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

// --- JsonWriter ----------------------------------------------------------------

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
        .number(0.1f)  // float widened to double: 9 digits reproduce it exactly
        .number(std::numeric_limits<double>::quiet_NaN())
        .number(std::numeric_limits<double>::infinity())
        .end_array();
    CHECK(w.str() == "[1.5,0.100000001,null,null]");
    CHECK(static_cast<float>(std::stod("0.100000001")) == 0.1f);
}

}  // namespace

int main() {
    test_parse_u64();
    test_vector_add_options();
    test_vector_add_reference();
    test_fill_uniform();
    test_validate_exact_match();
    test_validate_reports_mismatches();
    test_validate_tolerance_boundaries();
    test_validate_non_finite();
    test_validate_empty_is_not_a_pass();
    test_summary();
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
