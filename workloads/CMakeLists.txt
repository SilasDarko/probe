# probe_core: plain C++ (argument parsing, input generation, CPU reference
# implementations, validation, statistics, the result schema and JSON output).
# It is compiled by the host compiler and unit-tested on any machine.
add_library(
  probe_core STATIC
  src/core/cli_args.cpp
  src/core/device_info.cpp
  src/core/json_writer.cpp
  src/core/reference.cpp
  src/core/report.cpp
  src/core/stats.cpp
  src/core/validate.cpp)
target_include_directories(probe_core PUBLIC include)
target_compile_options(probe_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(probe_core_tests tests/test_core.cpp)
target_link_libraries(probe_core_tests PRIVATE probe_core)
target_compile_options(probe_core_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME probe_core_tests
         COMMAND probe_core_tests ${PROJECT_SOURCE_DIR}/tests/fixtures/stats_golden.txt)

# Prints a result JSON built from synthetic samples, so the Python side can
# test the schema against real C++ output without a GPU.
add_executable(probe_example_report tests/example_report.cpp)
target_link_libraries(probe_example_report PRIVATE probe_core)

if(NOT PROBE_HAVE_CUDA)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(probe-workloads src/main.cu src/device.cu src/harness.cu src/vector_add.cu
                               src/reduction.cu)
target_link_libraries(probe-workloads PRIVATE probe_core CUDA::cudart)

# Baked into the binary so every result can report how it was built.
string(REPLACE ";" "," PROBE_ARCH_STRING "${CMAKE_CUDA_ARCHITECTURES}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" PROBE_BUILD_TYPE_UPPER)
string(STRIP "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${PROBE_BUILD_TYPE_UPPER}}"
             PROBE_CUDA_FLAGS_STRING)
target_compile_definitions(
  probe-workloads
  PRIVATE PROBE_VERSION="${PROJECT_VERSION}"
          PROBE_BUILD_TYPE="$<CONFIG>"
          PROBE_CUDA_ARCHITECTURES="${PROBE_ARCH_STRING}"
          PROBE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}"
          PROBE_CXX_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
          PROBE_CUDA_FLAGS="${PROBE_CUDA_FLAGS_STRING}")

target_compile_options(probe-workloads PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
