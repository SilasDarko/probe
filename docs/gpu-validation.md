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

1. **Runtime → Change runtime type → GPU (T4 is enough).**
2. Upload `probe.bundle` (a git bundle of the repository) and run:

```python
!nvidia-smi
from google.colab import files
files.upload()                       # choose probe.bundle
!git clone probe.bundle probe
%cd probe
!cmake --version                     # needs 3.25+; if older: !pip install -U cmake
!python3 scripts/validate_gpu.py --quick
```

3. If the quick run completes, start the full run:

```python
!python3 scripts/validate_gpu.py
!python3 scripts/analyze_validation.py $(ls -d validation-runs/*/ | tail -1)
files.download(sorted(__import__("glob").glob("validation-runs/*.tar.gz"))[-1])
```

The `.tar.gz` contains every raw output. That file, not a summary of it, is what the results in BENCHMARKS.md must be derived from.

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
