// Copyright (c) 2026 Silas Offei Darko. MIT License.
#pragma once

// Exit codes of probe-workloads. They are part of the contract with the Python
// CLI, which uses them to tell "no GPU" apart from "CUDA failed" apart from
// "ran, but the result cannot be trusted" without parsing error text.
namespace probe::exit_code {

inline constexpr int kOk = 0;
inline constexpr int kCudaError = 1;  // also used by PROBE_CUDA_CHECK (EXIT_FAILURE)
inline constexpr int kUsage = 2;
inline constexpr int kNoDevice = 3;
// The run completed but is invalid (validation failed or timing data is
// malformed). The JSON is still printed, with "valid": false and reasons.
inline constexpr int kInvalidResult = 4;

}  // namespace probe::exit_code
