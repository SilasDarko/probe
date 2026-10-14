// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// vector_add: c[i] = a[i] + b[i] for n float32 elements.
//
// Timed phases (one stream, in order):
//   h2d     copy a and b host -> device
//   kernel  one thread per element
//   d2h     copy c device -> host
// Untimed reset: clear the device output so each validated result can only
// have come from that iteration's kernel.
//
// Host buffers are ordinary pageable memory (std::vector). Pinned memory would
// make transfers faster; it is deliberately not used in this baseline.

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/cuda_check.hpp"
#include "probe/device.hpp"
#include "probe/exit_codes.hpp"
#include "probe/harness.hpp"
#include "probe/reference.hpp"
#include "probe/report.hpp"
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

}  // namespace

int run_vector_add(const std::vector<std::string>& args) {
    // Arguments are validated before touching the GPU, so usage errors are
    // reported identically on machines with and without one.
    VectorAddOptions opt;
    std::string error;
    if (!parse_vector_add_options(args, opt, error)) {
        std::fprintf(stderr, "vector_add: %s\n", error.c_str());
        return exit_code::kUsage;
    }

    RunReport report;
    if (!select_device(0, report.device, error)) {
        std::fprintf(stderr, "probe-workloads: no usable CUDA device (%s)\n", error.c_str());
        return exit_code::kNoDevice;
    }

    const std::size_t n = static_cast<std::size_t>(opt.n);
    const unsigned block = static_cast<unsigned>(opt.block_size);
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);  // round up
    const std::size_t bytes = n * sizeof(float);

    // Inputs and CPU reference, computed once, before any timing.
    std::vector<float> h_a(n), h_b(n), h_c(n), expected(n);
    const auto seed = static_cast<std::uint32_t>(opt.common.seed);
    fill_unit_magnitude(h_a.data(), n, seed);
    fill_unit_magnitude(h_b.data(), n, seed + 1);
    vector_add_reference(h_a.data(), h_b.data(), expected.data(), n);

    // Exact equality is justified for this controlled input domain only, not
    // as a general CPU/GPU floating-point guarantee:
    //  - IEEE 754 requires a single addition to be correctly rounded, and both
    //    the CPU (SSE) and the GPU (no --use_fast_math) implement that;
    //  - a lone a + b cannot be fused or reassociated by either compiler;
    //  - every input is a normal float of magnitude [0.5, 1), a multiple of
    //    2^-24, so every sum is zero or at least 2^-24 in magnitude: no
    //    subnormals arise, so flush-to-zero settings cannot cause a difference.
    const Tolerance tol{0.0, 0.0};

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    PROBE_CUDA_CHECK(cudaMalloc(&d_a, bytes));
    PROBE_CUDA_CHECK(cudaMalloc(&d_b, bytes));
    PROBE_CUDA_CHECK(cudaMalloc(&d_c, bytes));

    WorkloadOps ops;
    ops.reset = [&] { PROBE_CUDA_CHECK(cudaMemset(d_c, 0, bytes)); };
    ops.phases = {
        {"h2d",
         [&] {
             PROBE_CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
             PROBE_CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
         }},
        {"kernel", [&] { vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n); }},
        {"d2h",
         [&] { PROBE_CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost)); }},
    };
    ops.validate = [&] { return validate_close(expected.data(), h_c.data(), n, tol); };

    report.workload = "vector_add";
    report.parameters = {{"n", opt.n},
                         {"block_size", opt.block_size},
                         {"grid_size", std::uint64_t{grid}},
                         {"dtype", std::string("float32")},
                         {"host_memory", std::string("pageable")},
                         {"input_domain", std::string("normal floats, |x| in [0.5, 1)")}};
    report.build = this_build();
    report.validation_method = "elementwise exact (rtol 0, atol 0)";
    report.tolerance = tol;

    run_benchmark(ops, opt.common, report);

    PROBE_CUDA_CHECK(cudaFree(d_a));
    PROBE_CUDA_CHECK(cudaFree(d_b));
    PROBE_CUDA_CHECK(cudaFree(d_c));
    return emit_run_report(report);
}

}  // namespace probe
