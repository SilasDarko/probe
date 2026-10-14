# Probe

A GPU performance regression and diagnostics toolkit for CUDA workloads.

Author: Silas Offei Darko

> Status: early development. The benchmark harness, two workloads (`vector_add`,
> `reduction`), and the baseline and regression-comparison CLI are in place.
> The CUDA code compiles, but no GPU code path has yet been executed on NVIDIA
> hardware; see [BENCHMARKS.md](BENCHMARKS.md).

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

This always builds `probe_core` (plain C++: argument parsing, input generation,
CPU reference implementations, validation, statistics, the result schema) and
its unit tests. The CUDA
executable `probe-workloads` is built only when a CUDA compiler is found. On a
GPU host, make a missing toolkit an error:

```sh
cmake -S . -B build -DPROBE_REQUIRE_CUDA=ON
```

## What the binary contains

GPU code can be embedded in two forms:

- **SASS** is native machine code for one GPU architecture.
- **PTX** is an intermediate form. The driver compiles it at load time (JIT) for the GPU it finds. PTX for `compute_XX` can only be JIT-compiled for architecture XX or newer.

The default `CMAKE_CUDA_ARCHITECTURES` is `75-real;80-real;86-real;89-real;90`. In CMake's syntax, `NN-real` means SASS only. A bare `NN` means SASS plus PTX. With CUDA 12.0, nvcc receives:

```
--generate-code=arch=compute_75,code=[sm_75]
--generate-code=arch=compute_80,code=[sm_80]
--generate-code=arch=compute_86,code=[sm_86]
--generate-code=arch=compute_89,code=[sm_89]
--generate-code=arch=compute_90,code=[compute_90,sm_90]
```

| GPU | Compute capability | Code that runs |
|---|---|---|
| T4 (Turing) | 7.5 | native `sm_75` SASS |
| A100 (Ampere) | 8.0 | native `sm_80` SASS |
| RTX 30-series, A10 (Ampere) | 8.6 | native `sm_86` SASS |
| L4, RTX 40-series (Ada) | 8.9 | native `sm_89` SASS |
| H100 (Hopper) | 9.0 | native `sm_90` SASS |
| Newer than 9.0 | > 9.0 | `compute_90` PTX, JIT-compiled by the driver at load time |
| Older than Turing | < 7.5 | not supported |

Some limits apply to the table above:

- **PTX is forward-only.** The single PTX image covers GPUs newer than Hopper. It is not a fallback for anything older than 9.0.
- **JIT is untested.** JIT-compiled code has a first-load compile cost, and its performance is not guaranteed to match native code. Probe has not been tested on any GPU that needs the JIT path.
- **Building fewer architectures.** To compile only for the GPU in the machine, use `-DCMAKE_CUDA_ARCHITECTURES=native`. This compiles faster, and the binary then runs only on that architecture.

To check what a build actually contains:

```sh
cuobjdump --list-elf --list-ptx build/workloads/probe-workloads
```

`probe-workloads --version` and every JSON result record the architecture list the binary was built with.

## Workloads

```sh
probe-workloads vector_add [--n N] [--block-size B] [common options]
probe-workloads reduction  [--n N] [--block-size B] [common options]
probe-workloads --device-info
```

Common options: `--warmup W` (default 3), `--iters I` (default 20), `--seed S`
(default 42), `--json`, `--raw-samples`. Both workloads default to
N = 16,777,216 and B = 256.

- **vector_add** computes `c[i] = a[i] + b[i]`, one thread per element. B must be a multiple of 32.
- **reduction** sums N floats. Each block loads one element per thread into shared memory and sums them as a tree. The kernel is relaunched on the per-block partial sums until one value remains. B must be a power of two. This is a deliberately unoptimized baseline.

### How an iteration is measured

Both workloads go through one shared harness (`harness.hpp`). A workload supplies three things: an untimed `reset`, a list of named timed phases, and a `validate` function. The harness owns everything else. Each iteration runs as follows:

```
reset()                      untimed: clear output buffers
cudaDeviceSynchronize()      untimed: nothing from reset or the previous
                             iteration is still running
host clock start
  event, h2d, event, kernel, event, d2h, event
  cudaEventSynchronize(last event)
host clock stop
validate()                   untimed: compare against the CPU reference
```

Each run reports two kinds of time, kept separate in the JSON:

- **`timing.gpu`** comes from CUDA events. It measures GPU execution between points in the stream: each phase, plus `total` from the first event to the last.
- **`timing.host.end_to_end`** comes from `std::chrono::steady_clock` around the whole timed region. It includes the time the host spends issuing work, driver overhead, any blocking copies, and the time for the host to wake up after the final synchronization.

The two clocks start at slightly different moments and measure different things. In the current synchronous workloads, host time is expected to be somewhat larger than GPU time, but that is an expectation, not a checked invariant. It has not been measured on hardware yet. Asynchronous copies, pinned memory and multiple streams will make the relationship more complicated.

Warmup iterations run the same sequence but are neither recorded nor validated.

### Correctness

Every measured iteration is validated against a CPU reference after its clocks have stopped. A run is valid only if at least one iteration was checked and none failed.

**vector_add uses exact equality.** This holds for this controlled input domain only; it is not a general CPU/GPU guarantee. Inputs come from `fill_unit_magnitude`. They are normal floats with magnitude in [0.5, 1), built directly from `std::mt19937` bits, so they are the same on every platform. A single IEEE 754 addition is correctly rounded on both CPU and GPU. With these inputs, every sum is either zero or at least 2^-24, so no subnormal values arise.

**reduction uses an error bound.** Floating-point addition is not associative. The GPU's tree order and the CPU's sequential order round differently, so the two results legitimately differ. The GPU result is compared against a double-precision reference using a worst-case bound for float32 tree summation, `gamma_d * sum|x|`. Here `d = passes * log2(B)` and `gamma_d = d*u/(1 - d*u)` with `u = 2^-24`; a small term is added for the reference itself.

The bound is safe but loose. Below roughly n = 450,000 it is smaller than any single element, so even one missing element is detected. Above that size, a single lost element can hide inside the rounding budget. The JSON reports the bound (`validation.atol`) so every result states its own sensitivity. The full derivation, its assumptions and its blind spots are in [docs/correctness.md](docs/correctness.md).

### Result JSON (`schema: probe.run`, `schema_version: 2`)

The top-level keys are `workload`, `valid`, `invalid_reasons`, `parameters`, `config`, `build`, `environment`, `validation`, `timing`, and, with `--raw-samples` only, `samples`.

Each timing series is summarized as `count`, `min`, `max`, `mean`, `p50`, `p95` and `stddev`, defined as follows:

- **Percentiles** use linear interpolation between closest ranks. This is NumPy's `method="linear"`.
- **`stddev`** is the sample standard deviation (n - 1). It is `null` for a single sample.
- **Rejected input:** a series containing NaN, infinite or negative durations is not summarized, and the run becomes invalid.

The C++ and Python implementations are both tested against `tests/fixtures/stats_golden.txt`.

### probe-workloads exit codes

| Code | Meaning |
|---|---|
| 0 | success, valid result |
| 1 | CUDA runtime error |
| 2 | usage error |
| 3 | no usable CUDA device |
| 4 | completed, but the result is invalid (JSON still printed, with reasons) |

## The `probe` CLI: baselines and regression comparison

```sh
pip install -e .

probe run vector-add --size 10000000             # writes results/vector-add-<time>.json
probe baseline save vector-add --name main --size 10000000   # runs, saves as 'main'
# ... change the code, rebuild ...
probe compare vector-add --baseline main --fail-if-regression 10

probe baseline list
probe baseline show main
probe baseline delete main
```

**Workflow model.** Every command that needs a measurement runs the workload itself, unless it is given a result file explicitly. There is no hidden "last run".

- `baseline save --from FILE` promotes an existing result file.
- `compare --current FILE` compares an existing result file.
- `compare` without `--current` re-runs the workload with the baseline's own recorded parameters (size, block size, warmup, iterations, seed), so both runs measure the same problem. Explicit options override those values, and the compatibility check then reports any mismatch.

Every run records raw per-iteration samples and adds provenance: the git commit and dirty flag of the directory it was run from, a timestamp, the command line, and host OS/CPU details.

**Storage.** Baselines are plain JSON in `<store>/baselines/<name>.json`. The store is `--store`, else `$PROBE_STORE`, else `./.probe`. A baseline wraps the complete result document, including raw samples. It is not gitignored, so baselines can be committed for CI.

Probe refuses to save an invalid run. Every load re-validates the file: the schema and version, required fields, that the run was valid, that every raw sample is a finite non-negative number, and that every stored statistic matches what is recomputed from the stored raw samples. A file that fails is reported as corrupted, malformed, or internally inconsistent.

This is a consistency check, not authentication. A deliberate edit that keeps samples and statistics consistent will not be noticed, and neither would a checksum stored in the same file. Detecting malicious modification would need a signature verified against a key kept outside the file, which Probe does not implement.

**Compatibility.** Before comparing, Probe sorts every difference into one of three groups:

| Group | Fields | Effect |
|---|---|---|
| fatal | workload, any workload parameter | refuses; cannot be overridden |
| environment | GPU name, compute capability, build type | refuses unless `--allow-environment-mismatch`, then a prominent warning |
| warning | CUDA runtime/driver, compiled architectures, compilers and flags, warmup/iterations/seed, host OS/CPU, Probe version | printed; comparison proceeds |

A different git commit is not a mismatch; comparing commits is the point.

**Decision rule.** The decision is made per metric and statistic. The gated metric defaults to `gpu_total` and the statistic to `p50`; use `--metric kernel`, `--stat p95`, and so on to change them. `delta = current - baseline`, and `percent = 100 * delta / baseline`. Then:

```
noise_floor    = 3 * sqrt(se_b^2 + se_c^2)              estimated from the samples
se             = c(stat) * 1.4826 * MAD / sqrt(n)      c(p50) = 1.2533, c(p95) = 2.113, c(mean) = 1
min_detectable = max(noise_floor, clock_resolution)    0.0005 ms for CUDA-event metrics,
                                                       1 ns (nominal) for host end-to-end

fewer than min(stat) samples on a side       -> INCONCLUSIVE   min: p50 20, mean 20, p95 100
|delta| <= min_detectable, |percent| >  T     -> INCONCLUSIVE
|delta| <= min_detectable, |percent| <= T     -> PASS
|delta| >  min_detectable, |percent| <= T     -> PASS
|delta| >  min_detectable, percent   >  T     -> REGRESSION
|delta| >  min_detectable, percent   < -T     -> IMPROVEMENT
```

`T` is `--fail-if-regression`, or 5% for an informational verdict when that option is not given. "Exceeds" is strict throughout.

The noise floor is the only estimate of execution noise, and it comes from the observed samples. Clock resolution is a property of the timer, not of the workload. It matters only when the samples show almost no spread: below it, a difference cannot be measured at all. The reasoning behind the rule is in [docs/regression-model.md](docs/regression-model.md).

**Exit codes of `probe`:**

| Code | Meaning |
|---|---|
| 0 | success; comparison did not fail the gate |
| 1 | `--fail-if-regression` given and the gated metric is a REGRESSION |
| 2 | usage error |
| 3 | no usable CUDA device |
| 4 | a run (or file) is an invalid run; never saved or compared |
| 5 | runs are not comparable (fatal or un-overridden environment mismatch) |
| 6 | baseline or result file missing, corrupted, malformed, or internally inconsistent |
| 7 | workload failed: binary missing, CUDA error, unusable output |
| 8 | `--fail-on-inconclusive` given and the gated metric is INCONCLUSIVE |

## Test

```sh
ctest --test-dir build --output-on-failure   # C++ core unit tests, no GPU needed

python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests marked `gpu` are skipped automatically when no NVIDIA GPU is detected.

## License

MIT. See [LICENSE](LICENSE).
