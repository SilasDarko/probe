# Probe: GPU performance regression and diagnostics toolkit.
#
# Top-level build. The C++/CUDA workloads are only built when a CUDA compiler
# is found, so the project still configures on machines without CUDA (for
# example, ordinary CI runners). Pass -DPROBE_REQUIRE_CUDA=ON on a GPU host to
# turn a missing toolkit into a hard error instead of a silent skip.

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
  # Must be set before enable_language(CUDA), otherwise CMake picks the
  # compiler's default architecture. Each entry produces native code (SASS)
  # for that architecture plus PTX, so newer GPUs can JIT-compile the PTX.
  #   75 = Turing (T4)   80 = Ampere (A100)   86 = Ampere (A10, RTX 30xx)
  #   89 = Ada (L4, RTX 40xx)
  # On a GPU host, -DCMAKE_CUDA_ARCHITECTURES=native builds only for the
  # installed GPU and compiles faster.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_subdirectory(workloads)
elseif(PROBE_REQUIRE_CUDA)
  message(FATAL_ERROR "PROBE_REQUIRE_CUDA=ON but no CUDA compiler was found.")
else()
  message(STATUS "No CUDA compiler found: skipping GPU workloads.")
endif()
