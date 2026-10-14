// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// The benchmark harness. A workload describes what to run; the harness decides
// how it is measured, so every workload is timed and validated the same way.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "probe/cli_args.hpp"
#include "probe/report.hpp"
#include "probe/validate.hpp"

namespace probe {

// One timed phase. `enqueue` issues the phase's GPU work on the default stream
// (copies, kernel launches). It may block the host, as cudaMemcpy from pageable
// memory does, but must not run CPU-side computation that is not part of the
// operation being measured.
struct Phase {
    std::string name;
    std::function<void()> enqueue;
};

// What a workload provides. No base class: three callables and a list.
struct WorkloadOps {
    // Optional. Runs before every iteration, OUTSIDE the timed region, e.g. to
    // clear output buffers so the result validated afterwards can only have
    // come from that iteration. The harness waits for it to finish
    // (cudaDeviceSynchronize) before timing starts.
    std::function<void()> reset;

    // Timed, in order. The harness records a CUDA event before the first
    // phase and after each phase.
    std::vector<Phase> phases;

    // Runs after timing has ended, on the host. Checks this iteration's
    // output against the CPU reference.
    std::function<ValidationResult()> validate;
};

// Runs `options.warmup` unrecorded iterations, then `options.iterations`
// measured ones, each of them:
//
//   reset()  -> cudaDeviceSynchronize()             not timed
//   host clock start
//     event 0, phase 1, event 1, ..., phase k, event k
//     cudaEventSynchronize(event k)
//   host clock stop                                  host.end_to_end
//   validate()                                       not timed
//
// Fills the timing series and validation summary of `report`.
void run_benchmark(const WorkloadOps& ops, const CommonOptions& options, RunReport& report);

// Build metadata compiled into this binary.
BuildInfo this_build();

}  // namespace probe
