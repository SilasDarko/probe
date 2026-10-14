# Probe: GPU performance regression and diagnostics toolkit.
#
# Top-level build. The CPU-side core library and its unit tests always build.
# The CUDA executable is only built when a CUDA compiler is found, so the
# project still configures on machines without CUDA (for example, ordinary CI
# runners). Pass -DPROBE_REQUIRE_CUDA=ON on a GPU host to turn a missing
# toolkit into a hard error instead of a silent skip.

cmake_minimum_required(VERSION 3.25)  # 3.25+ provides the CUDA::nvtx3 target

project(
  probe
  VERSION 0.1.0
  DESCRIPTION "GPU performance regression and diagnostics toolkit"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Benchmarks are meaningless in an unoptimized build, so default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

option(PROBE_REQUIRE_CUDA "Fail configuration if no CUDA compiler is found" OFF)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  # Architecture policy (see README, "What the binary contains"). Must be set
  # before enable_language(CUDA), otherwise CMake picks the compiler default.
  #
  #   NN-real  -> native SASS for sm_NN only, no PTX
  #   NN       -> SASS for sm_NN plus PTX for compute_NN
  #
  # Native SASS: sm_75 (T4), sm_80 (A100), sm_86 (RTX 30xx/A10), sm_89 (Ada),
  # sm_90 (H100). One PTX image, compute_90, lets the driver JIT-compile for
  # GPUs newer than 9.0. PTX only runs on its own or a newer architecture, so
  # it is no fallback for anything older than 9.0.
  #
  # Override on a GPU host with -DCMAKE_CUDA_ARCHITECTURES=native.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real 90)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(PROBE_HAVE_CUDA ON)
elseif(PROBE_REQUIRE_CUDA)
  message(FATAL_ERROR "PROBE_REQUIRE_CUDA=ON but no CUDA compiler was found.")
else()
  message(STATUS "No CUDA compiler found: building CPU-side core and tests only.")
  set(PROBE_HAVE_CUDA OFF)
endif()

enable_testing()
add_subdirectory(workloads)
