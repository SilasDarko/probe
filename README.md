# Probe

A GPU performance regression and diagnostics toolkit for CUDA workloads.

Author: Silas Offei Darko

> Status: early development. The build system and the first workload
> (`vector_add`) are in place. The CUDA code compiles, but no GPU code path has
> yet been executed on NVIDIA hardware; see [BENCHMARKS.md](BENCHMARKS.md).

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

This always builds `probe_core` (plain C++: argument parsing, CPU reference
implementations, result validation, JSON output) and its unit tests. The CUDA
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
probe-workloads vector_add [--n N] [--block-size B] [--warmup W] [--iters I] [--seed S] [--json]
probe-workloads --device-info
```

The defaults for `vector_add` are N = 16,777,216, B = 256, W = 3, I = 20 and S = 42.

Each measured iteration copies the inputs host-to-device, runs the kernel, and copies the result back. CUDA events time each phase separately. Every measured iteration's output is compared against a CPU reference. If any iteration is wrong, the run is marked `"valid": false` and the binary exits with code 4.

The exit codes form the interface the Python CLI relies on:

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | CUDA runtime error |
| 2 | usage error |
| 3 | no usable CUDA device |
| 4 | output failed validation |

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
