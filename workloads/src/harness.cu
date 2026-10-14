// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "probe/cuda_check.hpp"
#include "probe/exit_codes.hpp"
#include "probe/harness.hpp"

namespace probe {

namespace {

// Owns the k + 1 events that bracket k phases. An event is a marker placed in
// a stream; the GPU writes a timestamp into it when the stream reaches it.
class EventSet {
   public:
    explicit EventSet(std::size_t count) : events_(count) {
        for (cudaEvent_t& e : events_) PROBE_CUDA_CHECK(cudaEventCreate(&e));
    }
    ~EventSet() {
        for (cudaEvent_t e : events_) cudaEventDestroy(e);
    }
    EventSet(const EventSet&) = delete;
    EventSet& operator=(const EventSet&) = delete;

    cudaEvent_t operator[](std::size_t i) const { return events_[i]; }
    std::size_t size() const { return events_.size(); }

   private:
    std::vector<cudaEvent_t> events_;
};

double elapsed_ms(cudaEvent_t from, cudaEvent_t to) {
    float ms = 0.0f;
    PROBE_CUDA_CHECK(cudaEventElapsedTime(&ms, from, to));
    return static_cast<double>(ms);
}

struct IterationTiming {
    std::vector<double> phase_ms;
    double gpu_total_ms;
    double host_end_to_end_ms;
};

IterationTiming run_iteration(const WorkloadOps& ops, const EventSet& events) {
    // --- Untimed preparation -------------------------------------------------
    if (ops.reset) ops.reset();
    // Explicit boundary: nothing from reset() or the previous iteration may
    // still be executing when the clocks start.
    PROBE_CUDA_CHECK(cudaDeviceSynchronize());

    // --- Timed region --------------------------------------------------------
    const auto host_start = std::chrono::steady_clock::now();
    PROBE_CUDA_CHECK(cudaEventRecord(events[0]));
    for (std::size_t p = 0; p < ops.phases.size(); ++p) {
        ops.phases[p].enqueue();
        // Reports a bad launch configuration immediately. Faults inside a
        // kernel surface at the next synchronizing call instead.
        PROBE_CUDA_CHECK(cudaGetLastError());
        PROBE_CUDA_CHECK(cudaEventRecord(events[p + 1]));
    }
    // Wait until the GPU has passed the last event: all work is finished and
    // every timestamp is available.
    PROBE_CUDA_CHECK(cudaEventSynchronize(events[events.size() - 1]));
    const auto host_stop = std::chrono::steady_clock::now();

    IterationTiming t;
    for (std::size_t p = 0; p < ops.phases.size(); ++p) {
        t.phase_ms.push_back(elapsed_ms(events[p], events[p + 1]));
    }
    t.gpu_total_ms = elapsed_ms(events[0], events[events.size() - 1]);
    t.host_end_to_end_ms =
        std::chrono::duration<double, std::milli>(host_stop - host_start).count();
    return t;
}

}  // namespace

void run_benchmark(const WorkloadOps& ops, const CommonOptions& options, RunReport& report) {
    std::vector<std::string> names;
    for (const Phase& p : ops.phases) names.push_back(p.name);
    std::string error;
    if (ops.phases.empty() || !ops.validate || !check_phase_names(names, error)) {
        // A programming error in a workload, not a runtime condition.
        std::fprintf(
            stderr, "probe-workloads: invalid workload definition: %s\n",
            ops.phases.empty() ? "no phases" : (!ops.validate ? "no validate" : error.c_str()));
        std::exit(exit_code::kCudaError);
    }

    report.options = options;
    report.gpu_phases.clear();
    for (const std::string& name : names) report.gpu_phases.push_back({name, {}});
    report.gpu_total_ms.clear();
    report.host_end_to_end_ms.clear();
    report.validation = ValidationSummary{};

    const EventSet events(ops.phases.size() + 1);

    // Warmup absorbs one-time costs (module loading, first touch of device
    // memory, GPU clocks ramping up) so they do not land in the measurement.
    // Warmup output is not validated or recorded.
    for (std::uint64_t i = 0; i < options.warmup; ++i) run_iteration(ops, events);

    for (std::uint64_t i = 0; i < options.iterations; ++i) {
        const IterationTiming t = run_iteration(ops, events);
        for (std::size_t p = 0; p < t.phase_ms.size(); ++p) {
            report.gpu_phases[p].samples_ms.push_back(t.phase_ms[p]);
        }
        report.gpu_total_ms.push_back(t.gpu_total_ms);
        report.host_end_to_end_ms.push_back(t.host_end_to_end_ms);
        // After the clocks have stopped: validation never enters a timing.
        report.validation.add(ops.validate());
    }
}

BuildInfo this_build() {
    return BuildInfo{PROBE_VERSION,       PROBE_BUILD_TYPE,   PROBE_CUDA_ARCHITECTURES,
                     PROBE_CUDA_COMPILER, PROBE_CXX_COMPILER, PROBE_CUDA_FLAGS};
}

}  // namespace probe
