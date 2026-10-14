# probe_core: plain C++ (argument parsing, CPU reference implementations,
# result validation, JSON output). It is compiled by the host compiler and
# unit-tested on any machine, GPU or not.
add_library(probe_core STATIC src/core/cli_args.cpp src/core/json_writer.cpp
                              src/core/reference.cpp src/core/validate.cpp)
target_include_directories(probe_core PUBLIC include)
target_compile_options(probe_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(probe_core_tests tests/test_core.cpp)
target_link_libraries(probe_core_tests PRIVATE probe_core)
target_compile_options(probe_core_tests PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME probe_core_tests COMMAND probe_core_tests)

if(NOT PROBE_HAVE_CUDA)
  return()
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(probe-workloads src/main.cu src/device.cu src/vector_add.cu)
target_link_libraries(probe-workloads PRIVATE probe_core CUDA::cudart)

# Baked into the binary so every result can report how it was built.
string(REPLACE ";" "," PROBE_ARCH_STRING "${CMAKE_CUDA_ARCHITECTURES}")
target_compile_definitions(
  probe-workloads
  PRIVATE PROBE_VERSION="${PROJECT_VERSION}" PROBE_BUILD_TYPE="$<CONFIG>"
          PROBE_CUDA_ARCHITECTURES="${PROBE_ARCH_STRING}")

target_compile_options(probe-workloads PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
