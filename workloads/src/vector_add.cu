// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// vector_add: c[i] = a[i] + b[i] for n float32 elements.
//
// Each measured iteration is the full round trip a real program pays:
//   copy a and b host->device, run the kernel, copy c device->host.
// CUDA events bracket each phase so transfer time and kernel time are reported
// separately. Every measured iteration's output is checked against a CPU
// reference; a run with any incorrect output is reported as invalid.
//
// Host buffers are ordinary pageable memory (std::vector). Pinned memory would
// make transfers faster; it is deliberately not used in this baseline.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/cuda_check.hpp"
#include "probe/device.hpp"
#include "probe/exit_codes.hpp"
#include "probe/json_writer.hpp"
#include "probe/reference.hpp"
#include "probe/validate.hpp"
#include "probe/workloads.hpp"

namespace probe {

namespace {

// One thread computes one element.
//
// The grid is a 1-D array of blocks and each block a 1-D array of threads, so
// a thread's global index is (its block's index * threads per block) + its
// index within the block. The grid is rounded up to whole blocks, so the last
// block can contain threads whose index is past the end of the arrays; the
// bounds check stops those threads from reading and writing out of bounds.
//
// blockIdx.x * blockDim.x is computed in 64 bits: in 32-bit unsigned
// arithmetic it would wrap around for n above about 4.29 billion.
__global__ void vector_add_kernel(const float* a, const float* b, float* c, std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Timing for one iteration, in milliseconds.
struct Sample {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
    float total_ms;
};

// All measured iterations, one vector per phase (the layout of the JSON).
struct PhaseSamples {
    std::vector<float> h2d, kernel, d2h, total;

    void add(const Sample& s) {
        h2d.push_back(s.h2d_ms);
        kernel.push_back(s.kernel_ms);
        d2h.push_back(s.d2h_ms);
        total.push_back(s.total_ms);
    }
};

// Four events, created once and reused. An event is a marker placed in a
// stream; the GPU stamps it with a timestamp when the stream reaches it.
struct PhaseEvents {
    cudaEvent_t start, after_h2d, after_kernel, after_d2h;

    PhaseEvents() {
        PROBE_CUDA_CHECK(cudaEventCreate(&start));
        PROBE_CUDA_CHECK(cudaEventCreate(&after_h2d));
        PROBE_CUDA_CHECK(cudaEventCreate(&after_kernel));
        PROBE_CUDA_CHECK(cudaEventCreate(&after_d2h));
    }
    ~PhaseEvents() {
        cudaEventDestroy(start);
        cudaEventDestroy(after_h2d);
        cudaEventDestroy(after_kernel);
        cudaEventDestroy(after_d2h);
    }
    PhaseEvents(const PhaseEvents&) = delete;
    PhaseEvents& operator=(const PhaseEvents&) = delete;
};

struct DeviceBuffers {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
};

// Runs one H2D -> kernel -> D2H round trip on the default stream and returns
// its phase timings. Everything is issued to one stream, so the phases run in
// order and each event's timestamp marks the boundary between two phases.
Sample run_iteration(const std::vector<float>& h_a, const std::vector<float>& h_b,
                     std::vector<float>& h_c, const DeviceBuffers& d, std::size_t n, unsigned grid,
                     unsigned block, const PhaseEvents& ev) {
    const std::size_t bytes = n * sizeof(float);

    // Clear the output before timing starts, so the result validated after
    // this iteration can only have come from this iteration's kernel.
    PROBE_CUDA_CHECK(cudaMemset(d.c, 0, bytes));

    PROBE_CUDA_CHECK(cudaEventRecord(ev.start));
    PROBE_CUDA_CHECK(cudaMemcpy(d.a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    PROBE_CUDA_CHECK(cudaMemcpy(d.b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    PROBE_CUDA_CHECK(cudaEventRecord(ev.after_h2d));

    vector_add_kernel<<<grid, block>>>(d.a, d.b, d.c, n);
    // A bad launch configuration fails here; faults inside the kernel surface
    // at the next synchronizing call (the D2H copy below).
    PROBE_CUDA_CHECK(cudaGetLastError());
    PROBE_CUDA_CHECK(cudaEventRecord(ev.after_kernel));

    PROBE_CUDA_CHECK(cudaMemcpy(h_c.data(), d.c, bytes, cudaMemcpyDeviceToHost));
    PROBE_CUDA_CHECK(cudaEventRecord(ev.after_d2h));

    // Block the CPU until the GPU has reached the last event; only then are
    // all four timestamps available.
    PROBE_CUDA_CHECK(cudaEventSynchronize(ev.after_d2h));

    Sample s{};
    PROBE_CUDA_CHECK(cudaEventElapsedTime(&s.h2d_ms, ev.start, ev.after_h2d));
    PROBE_CUDA_CHECK(cudaEventElapsedTime(&s.kernel_ms, ev.after_h2d, ev.after_kernel));
    PROBE_CUDA_CHECK(cudaEventElapsedTime(&s.d2h_ms, ev.after_kernel, ev.after_d2h));
    PROBE_CUDA_CHECK(cudaEventElapsedTime(&s.total_ms, ev.start, ev.after_d2h));
    return s;
}

void write_samples(JsonWriter& w, const char* name, const std::vector<float>& values) {
    w.key(name).begin_array();
    for (const float v : values) w.number(v);
    w.end_array();
}

void print_json(const VectorAddOptions& opt, unsigned grid, const DeviceInfo& device,
                const PhaseSamples& samples, const Tolerance& tol,
                const ValidationSummary& validation) {
    JsonWriter w;
    w.begin_object();
    w.key("schema").string("probe.run/1");
    w.key("workload").string("vector_add");
    w.key("valid").boolean(validation.passed());

    w.key("parameters").begin_object();
    w.key("n").uinteger(opt.n);
    w.key("dtype").string("float32");
    w.key("block_size").uinteger(opt.block_size);
    w.key("grid_size").uinteger(grid);
    w.key("warmup").uinteger(opt.warmup);
    w.key("iterations").uinteger(opt.iterations);
    w.key("seed").uinteger(opt.seed);
    w.key("host_memory").string("pageable");
    w.end_object();

    w.key("build").begin_object();
    w.key("probe_version").string(PROBE_VERSION);
    w.key("build_type").string(PROBE_BUILD_TYPE);
    w.key("cuda_architectures").string(PROBE_CUDA_ARCHITECTURES);
    w.end_object();

    w.key("device");
    write_device_json(w, device);

    w.key("timing").begin_object();
    w.key("method").string("cuda_events");
    w.key("unit").string("ms");
    w.key("samples").begin_object();
    write_samples(w, "h2d", samples.h2d);
    write_samples(w, "kernel", samples.kernel);
    write_samples(w, "d2h", samples.d2h);
    write_samples(w, "total", samples.total);
    w.end_object();
    w.end_object();

    w.key("validation").begin_object();
    w.key("passed").boolean(validation.passed());
    w.key("reference").string("cpu");
    w.key("rtol").number(tol.rtol);
    w.key("atol").number(tol.atol);
    w.key("iterations_checked").uinteger(validation.iterations_checked);
    w.key("iterations_failed").uinteger(validation.iterations_failed);
    w.key("total_mismatches").uinteger(validation.total_mismatches);
    w.key("max_abs_error").number(validation.max_abs_error);
    w.key("first_failed_iteration").integer(validation.first_failed_iteration);
    w.end_object();

    w.end_object();
    std::printf("%s\n", w.str().c_str());
}

void print_human(const VectorAddOptions& opt, const DeviceInfo& device, const PhaseSamples& samples,
                 const ValidationSummary& validation) {
    std::printf("vector_add  n=%llu  block=%llu  warmup=%llu  iters=%llu  device=%s\n",
                static_cast<unsigned long long>(opt.n),
                static_cast<unsigned long long>(opt.block_size),
                static_cast<unsigned long long>(opt.warmup),
                static_cast<unsigned long long>(opt.iterations), device.name.c_str());
    // A quick human summary only; the statistics Probe reports (median, p95,
    // spread) are computed by the Python CLI from the JSON samples.
    const std::pair<const char*, const std::vector<float>*> phases[] = {{"h2d", &samples.h2d},
                                                                        {"kernel", &samples.kernel},
                                                                        {"d2h", &samples.d2h},
                                                                        {"total", &samples.total}};
    for (const auto& [name, values] : phases) {
        const float mn = *std::min_element(values->begin(), values->end());
        const double mean = std::accumulate(values->begin(), values->end(), 0.0) /
                            static_cast<double>(values->size());
        std::printf("  %-7s min %10.4f ms   mean %10.4f ms\n", name, mn, mean);
    }
    if (validation.passed()) {
        std::printf("validation: PASSED (%zu iterations checked against CPU reference)\n",
                    validation.iterations_checked);
    } else {
        std::printf(
            "validation: FAILED in %zu of %zu iterations (%zu mismatches); "
            "timings above are INVALID\n",
            validation.iterations_failed, validation.iterations_checked,
            validation.total_mismatches);
    }
}

}  // namespace

int run_vector_add(const std::vector<std::string>& args) {
    // 1. Arguments are validated before touching the GPU, so usage errors are
    //    reported identically on machines with and without one.
    VectorAddOptions opt;
    std::string error;
    if (!parse_vector_add_options(args, opt, error)) {
        std::fprintf(stderr, "vector_add: %s\n", error.c_str());
        return exit_code::kUsage;
    }

    DeviceInfo device;
    if (!select_device(0, device, error)) {
        std::fprintf(stderr, "probe-workloads: no usable CUDA device (%s)\n", error.c_str());
        return exit_code::kNoDevice;
    }

    const std::size_t n = static_cast<std::size_t>(opt.n);
    const unsigned block = static_cast<unsigned>(opt.block_size);
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);  // round up
    const std::size_t bytes = n * sizeof(float);

    // 2. Inputs and the CPU reference, computed once.
    std::vector<float> h_a(n), h_b(n), h_c(n), expected(n);
    fill_uniform(h_a.data(), n, static_cast<std::uint32_t>(opt.seed));
    fill_uniform(h_b.data(), n, static_cast<std::uint32_t>(opt.seed) + 1);
    vector_add_reference(h_a.data(), h_b.data(), expected.data(), n);

    // A single float addition is correctly rounded under IEEE 754 on both the
    // CPU and the GPU (no fast-math, no fused operations possible), so the
    // results must match exactly.
    const Tolerance tol{0.0, 0.0};

    DeviceBuffers d;
    PROBE_CUDA_CHECK(cudaMalloc(&d.a, bytes));
    PROBE_CUDA_CHECK(cudaMalloc(&d.b, bytes));
    PROBE_CUDA_CHECK(cudaMalloc(&d.c, bytes));
    PhaseSamples samples;
    ValidationSummary validation;
    {
        const PhaseEvents ev;

        // 3. Warmup: absorbs one-time costs (context and module loading, first
        //    touch of device memory, clock ramp-up). Not recorded.
        for (std::uint64_t i = 0; i < opt.warmup; ++i) {
            run_iteration(h_a, h_b, h_c, d, n, grid, block, ev);
        }

        // 4. Measured iterations. Each output is validated after its timings
        //    are taken, so validation cost never enters the measurement.
        for (std::uint64_t i = 0; i < opt.iterations; ++i) {
            samples.add(run_iteration(h_a, h_b, h_c, d, n, grid, block, ev));
            validation.add(validate_close(expected.data(), h_c.data(), n, tol));
        }
    }
    PROBE_CUDA_CHECK(cudaFree(d.a));
    PROBE_CUDA_CHECK(cudaFree(d.b));
    PROBE_CUDA_CHECK(cudaFree(d.c));

    // 5. Report. An invalid run still prints its data (useful for debugging)
    //    but is marked invalid and exits non-zero.
    if (opt.json) {
        print_json(opt, grid, device, samples, tol, validation);
    } else {
        print_human(opt, device, samples, validation);
    }
    return validation.passed() ? exit_code::kOk : exit_code::kValidationFailed;
}

}  // namespace probe
