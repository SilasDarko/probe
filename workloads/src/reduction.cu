// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// reduction: sum of n float32 values.
//
// Baseline implementation, deliberately unoptimized: each block loads one
// element per thread into shared memory and sums it with a tree; one thread
// per block writes the block's partial sum. The kernel is relaunched on the
// partial sums until a single value remains.
//
// Timed phases (one stream, in order):
//   h2d     copy the input host -> device
//   kernel  all reduction passes (one launch per pass)
//   d2h     copy the single result device -> host
// Untimed reset: clear the scratch buffers and poison the host result with
// NaN, so a missing or stale result fails validation.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <limits>
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

// Sums the up-to-blockDim.x elements assigned to this block into out[blockIdx.x].
//
// Unlike vector_add, the threads of a block must cooperate: every element
// contributes to one output. They do it through shared memory, a small
// on-chip buffer visible to all threads of one block and much faster than
// global memory.
//
//   1. Each thread copies one element into sdata[tid]. Threads past the end of
//      the input store 0, the identity for addition, so non-power-of-two
//      sizes need no special case later.
//   2. Tree reduction: with `stride` halving each step, thread tid adds
//      sdata[tid + stride] into sdata[tid]. After log2(blockDim.x) steps the
//      block's sum is in sdata[0]. This needs blockDim.x to be a power of two.
//   3. Thread 0 writes the result.
//
// __syncthreads() is a barrier for all threads of the block. It guarantees
// every write to shared memory before it is visible to every thread after it.
// Without it, a thread could read sdata[tid + stride] before the thread
// responsible for that slot has written it. It sits outside the `if` because
// every thread of the block must reach the same barrier.
__global__ void reduce_sum_kernel(const float* in, float* out, std::size_t n) {
    extern __shared__ float sdata[];  // size set at launch: blockDim.x floats

    const unsigned tid = threadIdx.x;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
    sdata[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

}  // namespace

int run_reduction(const std::vector<std::string>& args) {
    ReductionOptions opt;
    std::string error;
    if (!parse_reduction_options(args, opt, error)) {
        std::fprintf(stderr, "reduction: %s\n", error.c_str());
        return exit_code::kUsage;
    }

    RunReport report;
    if (!select_device(0, report.device, error)) {
        std::fprintf(stderr, "probe-workloads: no usable CUDA device (%s)\n", error.c_str());
        return exit_code::kNoDevice;
    }

    const std::size_t n = static_cast<std::size_t>(opt.n);
    const unsigned block = static_cast<unsigned>(opt.block_size);
    const std::size_t shared_bytes = block * sizeof(float);
    const std::vector<std::uint64_t> passes = reduction_pass_sizes(opt.n, opt.block_size);

    // Inputs, CPU reference and tolerance, computed once, before any timing.
    std::vector<float> h_in(n);
    fill_unit_magnitude(h_in.data(), n, static_cast<std::uint32_t>(opt.common.seed));
    const ReductionReference ref = reduction_reference(h_in.data(), n);
    const Tolerance tol = reduction_tolerance(opt.n, opt.block_size, ref.abs_sum);

    // Two scratch buffers for partial sums; passes alternate between them
    // ("ping-pong") so a pass never reads the buffer it is writing.
    const std::size_t scratch_len = passes.empty() ? 1 : static_cast<std::size_t>(passes[0]);
    float *d_in = nullptr, *d_scratch[2] = {nullptr, nullptr};
    PROBE_CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    PROBE_CUDA_CHECK(cudaMalloc(&d_scratch[0], scratch_len * sizeof(float)));
    PROBE_CUDA_CHECK(cudaMalloc(&d_scratch[1], scratch_len * sizeof(float)));

    // Where the final value ends up is fixed by the pass count, so it is
    // computed once instead of inside the timed phase.
    const float* d_result = passes.empty() ? d_in : d_scratch[(passes.size() - 1) % 2];
    float h_result = 0.0f;

    WorkloadOps ops;
    ops.reset = [&] {
        PROBE_CUDA_CHECK(cudaMemset(d_scratch[0], 0, scratch_len * sizeof(float)));
        PROBE_CUDA_CHECK(cudaMemset(d_scratch[1], 0, scratch_len * sizeof(float)));
        h_result = std::numeric_limits<float>::quiet_NaN();
    };
    ops.phases = {
        {"h2d",
         [&] {
             PROBE_CUDA_CHECK(
                 cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
         }},
        {"kernel",
         [&] {
             const float* src = d_in;
             std::size_t len = n;
             for (std::size_t p = 0; p < passes.size(); ++p) {
                 float* dst = d_scratch[p % 2];
                 const auto grid = static_cast<unsigned>(passes[p]);
                 reduce_sum_kernel<<<grid, block, shared_bytes>>>(src, dst, len);
                 src = dst;
                 len = grid;
             }
         }},
        {"d2h",
         [&] {
             PROBE_CUDA_CHECK(
                 cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
         }},
    };
    ops.validate = [&] { return validate_scalar(ref.sum, static_cast<double>(h_result), tol); };

    report.workload = "reduction";
    report.parameters = {{"n", opt.n},
                         {"block_size", opt.block_size},
                         {"passes", std::uint64_t{passes.size()}},
                         {"dtype", std::string("float32")},
                         {"accumulate", std::string("float32")},
                         {"host_memory", std::string("pageable")},
                         {"input_domain", std::string("normal floats, |x| in [0.5, 1)")}};
    report.build = this_build();
    report.validation_method = "abs error <= float32 tree-summation bound vs double reference";
    report.tolerance = tol;

    run_benchmark(ops, opt.common, report);

    PROBE_CUDA_CHECK(cudaFree(d_in));
    PROBE_CUDA_CHECK(cudaFree(d_scratch[0]));
    PROBE_CUDA_CHECK(cudaFree(d_scratch[1]));
    return emit_run_report(report);
}

}  // namespace probe
