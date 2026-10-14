# GPU validation procedure

Probe was developed on a machine without an NVIDIA GPU. Everything that needs hardware is checked by one script, on one machine, in one sitting:

```sh
python3 scripts/validate_gpu.py                 # records everything
python3 scripts/analyze_validation.py validation-runs/<timestamp>   # writes report.md
```

`validate_gpu.py` uses only the Python standard library and interprets nothing. It runs each step, saves the command, exit code, duration, stdout and stderr, and moves on when a step fails, so one problem cannot hide the rest. `steps.json` is rewritten after every step, so an interrupted session keeps what it gathered. At the end, everything except the CMake build tree is packed into `validation-runs/<timestamp>.tar.gz`.

`--quick` runs a reduced version (a few sizes, few iterations) to check that the procedure works on a new machine before spending time on the full run.

## What it runs

| Step | What | Output |
|---|---|---|
| A | `nvidia-smi` (plain, `--query-gpu`, `-q`), `nvcc --version`, `uname -a`, `lscpu`, `/etc/os-release`, cmake, python, gcc/g++, git commit and status, `compute-sanitizer --version` | `env/` |
| B | clean CMake build (`-DCMAKE_BUILD_TYPE=Release -DPROBE_REQUIRE_CUDA=ON`, default architectures), `ctest`, `cuobjdump --list-elf --list-ptx`, `pip install -e .[dev]`, full `pytest` | `build/` |
| C | `probe-workloads --device-info`, and a human-readable `vector_add` run | `device/` |
| D | `vector_add` at n = 1, 31, 32, 33, 255, 256, 257, 1000003, 10000000 (20 iterations each) | `vector_add/` |
| E | `reduction` at tiny, block-boundary, prime, detection-boundary (around n = 466,000) and large sizes | `reduction/` |
| D/E | `compute-sanitizer` with memcheck, racecheck, synccheck and initcheck, on awkward sizes of both workloads | `sanitizer/` |
| F | 200-iteration runs of both workloads with raw samples | `timing/` |
| G | the same runs with 0, 1, 5, 10 and 50 warmup iterations, each a separate process | `warmup/` |
| H | five independent processes of 100 iterations per workload | `noise/` |
| I | `probe baseline save`, `show`, and two `compare` runs in separate processes, for both workloads | `workflow/` |

The analysis compares Probe's device JSON against `nvidia-smi` and `nvcc`. It then reports:

- which SASS or PTX image the device should load;
- correctness and sanitizer summaries;
- reduction error against its analytical bound, and whether a single dropped element would be detected at each size;
- the phase-sum gap and the host-minus-GPU gap;
- iteration-by-iteration warmup behavior;
- within-run against between-run noise, including pairwise A/A comparisons of unchanged code with the current decision rule;
- the workflow verdicts.

One known difference is reported as a GAP rather than hidden. Probe records the CUDA version the driver supports (the `CUDA Version` shown by `nvidia-smi`), not the NVIDIA driver build number.

## On Google Colab

Use `Probe_GPU_Validation.ipynb` at the repository root; the minimal steps are in [COLAB_VALIDATION.md](../COLAB_VALIDATION.md). The notebook runs the following stages:

1. Checks for a GPU, and stops at once if there is none.
2. Records the toolchain, including whether compute-sanitizer, Nsight Systems and Nsight Compute are available.
3. Clones `probe.bundle`.
4. Installs only what `pyproject.toml` and `CMakeLists.txt` require. It never installs a driver or CUDA.
5. Builds cleanly, then runs ctest and pytest.
6. Compares `--device-info` with `nvidia-smi`.
7. Runs `validate_gpu.py --quick`, and stops if that finds a problem.
8. Runs the full `validate_gpu.py` and `analyze_validation.py`.
9. Packs one `probe-stage5-colab-<gpu>-<timestamp>.tar.gz`.

It prints "STAGE 5 HARDWARE VALIDATION FINISHED" only if both workloads produced valid results on the GPU.

## On a cloud VM

A Linux VM with an NVIDIA driver, the CUDA toolkit (`nvcc`, `cuobjdump`, `compute-sanitizer`), CMake 3.25+, Python 3.11+ and git:

```sh
git clone probe.bundle probe && cd probe     # or clone from a remote
python3 scripts/validate_gpu.py --quick && python3 scripts/validate_gpu.py
python3 scripts/analyze_validation.py validation-runs/<timestamp>
```

## Rules for the results

- **Do not change settings to make the first run pass.** Do not alter architecture settings, sizes or tolerances before recording what happened with the defaults.
- **Do not edit recorded outputs.** If Probe's environment JSON disagrees with `nvidia-smi`, the disagreement is a finding.
- **Report only what was recorded.** BENCHMARKS.md stays "pending hardware validation" until a run from this procedure exists, and then reports only what it recorded, with the hardware, versions, commit, commands, sizes, warmups and iterations.
