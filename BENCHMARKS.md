# Benchmarks

Every number in this file was measured on real NVIDIA hardware by
`scripts/validate_gpu.py` (the procedure in
[docs/gpu-validation.md](docs/gpu-validation.md)), run unmodified through
`Probe_GPU_Validation.ipynb`, and analyzed with `scripts/analyze_validation.py`.
Nothing here is estimated or carried over from another machine. Results from one
GPU say nothing about another.

## Run 20261014T170028Z: Tesla T4 on Google Colab

### Environment

| item | value |
|---|---|
| Probe commit | `dedbd54b976d613b2ad706d58e18b71e1adef5e1` |
| GPU | Tesla T4, compute capability 7.5, 40 SMs, ECC enabled |
| GPU memory | 15360 MiB per `nvidia-smi`; 14913 MiB usable per the CUDA runtime (ECC reserves the difference) |
| NVIDIA driver | 580.82.07 (supports CUDA 13.0); persistence mode disabled |
| CUDA toolkit / runtime | nvcc 13.0.88; runtime 13.0 |
| compute-sanitizer / Nsight Systems / Nsight Compute | 2025.3.1 / 2025.3.1 / 2025.3.1 (present; Nsight was not run) |
| CPU | Intel Xeon @ 2.00 GHz, 2 vCPUs |
| OS | Ubuntu 24.04.4 LTS |
| Host compiler | GCC 13.3.0 |
| Build | CMake 3.31.10, Release, CUDA flags `-O3 -DNDEBUG`, architectures `75-real;80-real;86-real;89-real;90` (native `sm_75` SASS used, no PTX JIT expected) |
| Host memory | pageable (`std::vector`), not pinned |
| Timing | CUDA events per phase (h2d, kernel, d2h) and for the GPU total; host `steady_clock` end to end; validation outside the timed region |

Colab is a shared, virtualized machine with no clock locking. Treat these
numbers as one sample of that environment, not as a property of the T4.

### Validation outcome

- 142 recorded steps, all exit 0. C++ tests: 1/1 passed. Python tests: 175 passed, 4
  skipped (the 4 skips exercise the no-GPU path, which cannot run on a GPU machine).
- `vector_add`: 9 of 9 sizes valid (n = 1 to 10,000,000), exact match at every size.
- `reduction`: 27 of 27 sizes valid (n = 1 to 16,777,216). The largest observed
  error was 1.72e-4 against a bound of 18.0 at n = 16,777,216. The highest
  error-to-bound ratio was 0.016, at n = 255.
- compute-sanitizer: 52 runs (memcheck, racecheck, synccheck and initcheck over both
  workloads at boundary sizes), all with a zero-error summary.
- Baseline workflow: save, show, list and four compares all exited 0 with verdict PASS.

### Timing: 200 measured iterations after 5 warmup iterations, seed 42

**vector_add**, n = 10,000,000 float32, block 256, grid 39,063

| metric | p50 ms | p95 ms | min ms | max ms |
|---|---|---|---|---|
| h2d | 17.295 | 18.493 | 16.747 | 21.084 |
| kernel | 0.461 | 0.464 | 0.453 | 0.471 |
| d2h | 9.072 | 9.969 | 8.766 | 14.280 |
| gpu_total | 26.839 | 28.629 | 26.086 | 35.159 |
| host_end_to_end | 26.850 | 28.639 | 26.098 | 35.173 |

**reduction**, n = 16,777,216 float32, block 256, 3 passes

| metric | p50 ms | p95 ms | min ms | max ms |
|---|---|---|---|---|
| h2d | 14.123 | 15.570 | 13.673 | 16.443 |
| kernel | 0.641 | 0.955 | 0.633 | 0.962 |
| d2h | 0.014 | 0.021 | 0.012 | 0.031 |
| gpu_total | 14.896 | 16.455 | 14.325 | 17.241 |
| host_end_to_end | 14.905 | 16.464 | 14.333 | 17.249 |

The effective bandwidths below are derived from the p50 values above using the bytes each phase moves:

| quantity | value |
|---|---|
| vector_add kernel (reads 2 arrays, writes 1; 120 MB) | 260.5 GB/s, 81% of the T4's published 320 GB/s peak |
| vector_add h2d (80 MB, pageable) | 4.63 GB/s |
| vector_add d2h (40 MB, pageable) | 4.41 GB/s |
| reduction h2d (67.1 MB, pageable) | 4.75 GB/s |
| reduction kernel (first-pass input only, 67.1 MB) | 104.7 GB/s |

Both workloads are dominated by pageable PCIe transfers: the kernel is 1.7%
(vector_add) and 4.3% (reduction) of the median GPU total.

Accounting holds per iteration: h2d + kernel + d2h equals the GPU total to within
7.2e-6%. Host end-to-end minus GPU total was never negative. The overhead was
9.1–39.2 µs for vector_add (median 11.1) and 7.0–25.9 µs for reduction (median 8.2).

### Run-to-run noise (5 separate processes, 100 iterations each, 3 warmup)

| workload, metric | medians range | between-run SD | mean within-run SE | ratio | A/A verdicts at T = 5% |
|---|---|---|---|---|---|
| vector_add, gpu_total | 26.349–26.616 ms | 101.9 µs | 42.5 µs | 2.4 | 10 PASS |
| vector_add, kernel | 0.458–0.460 ms | 0.7 µs | 0.4 µs | 1.6 | 10 PASS |
| reduction, gpu_total | 14.810–15.020 ms | 88.4 µs | 22.3 µs | 4.0 | 10 PASS |
| reduction, kernel | 0.637–0.769 ms | 58.4 µs | 3.4 µs | 17 | 6 PASS, 4 IMPROVEMENT |

The default gate (`gpu_total` p50, 5% threshold) produced no false verdicts across
the 20 same-code pairs. For `gpu_total`, between-run variation was 2.4–4× what the
within-run noise model predicts. In 3 of 10 (vector_add) and 6 of 10 (reduction)
same-code pairs, the difference exceeded the noise floor; the 5% threshold absorbed it.

### Observed GPU clock ramp on short kernels

In the timed reduction run, kernel time fell in discrete steps:

| iterations | kernel time |
|---|---|
| 0–18 | about 0.955 ms |
| 19–31 | about 0.873 ms |
| 32–45 | about 0.817 ms |
| 46–58 | about 0.777 ms |
| 59–72 | about 0.721 ms |
| 73–85 | about 0.672 ms |
| 86 onward | 0.633–0.642 ms |

The pattern is consistent with the GPU raising its clocks while the workload runs
(persistence mode disabled; clocks not locked on Colab); Probe did not record clocks
to confirm it. Five warmup iterations do not cover the ramp. That is why the reduction
kernel p95 (0.955 ms) is far above its p50. It is also why noise run 1 had a kernel
median of 0.769 ms, which produced the 4 IMPROVEMENT verdicts in same-code
kernel-metric comparisons. The vector_add kernel did not show the effect. The warmup
table's iteration-0-versus-settled ratio is based on `gpu_total`, where transfers
dominate, so it stayed at about 1.0 and does not reveal the ramp.
