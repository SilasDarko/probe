find_package(CUDAToolkit REQUIRED)

add_executable(probe-workloads src/main.cu)

target_include_directories(probe-workloads PRIVATE include)
target_link_libraries(probe-workloads PRIVATE CUDA::cudart)

# Baked into the binary so every result can report how it was built.
target_compile_definitions(
  probe-workloads PRIVATE PROBE_VERSION="${PROJECT_VERSION}"
                          PROBE_BUILD_TYPE="$<CONFIG>")

target_compile_options(
  probe-workloads
  PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
          $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
