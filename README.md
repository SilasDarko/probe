# Probe

A GPU performance regression and diagnostics toolkit for CUDA workloads.

Author: Silas Offei Darko

> Status: early development. The repository skeleton and build system are in
> place; workloads, the CLI, and profiling integration are being added in
> stages. GPU execution has not yet been validated on NVIDIA hardware.

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

Without a CUDA compiler, configuration succeeds and skips the GPU workloads.
On a GPU host, require CUDA and target only the installed GPU:

```sh
cmake -S . -B build -DPROBE_REQUIRE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
```

## Test

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests marked `gpu` are skipped automatically when no NVIDIA GPU is detected.

## License

MIT. See [LICENSE](LICENSE).
