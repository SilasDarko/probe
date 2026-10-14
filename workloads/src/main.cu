// Copyright (c) 2026 Silas Offei Darko. MIT License.
//
// probe-workloads: the single executable that hosts Probe's CUDA workloads.
// The Python CLI invokes it and reads JSON from stdout. Exit codes are listed
// in probe/exit_codes.hpp.

#include <cstdio>
#include <string>
#include <vector>

#include "probe/device.hpp"
#include "probe/exit_codes.hpp"
#include "probe/json_writer.hpp"
#include "probe/workloads.hpp"

namespace {

void print_usage(const char* prog) {
    std::printf(
        "usage: %s <workload> [options]\n"
        "       %s --device-info | --version | --help\n"
        "\n"
        "workloads:\n"
        "  vector_add  [--n N] [--block-size B] [--warmup W] [--iters I] [--seed S] [--json]\n"
        "\n"
        "  --device-info  print the selected GPU's properties as JSON\n"
        "  --version      print the version, build type and CUDA architectures\n",
        prog, prog);
}

int print_device_info() {
    probe::DeviceInfo info;
    std::string error;
    if (!probe::select_device(0, info, error)) {
        std::fprintf(stderr, "probe-workloads: no usable CUDA device (%s)\n", error.c_str());
        return probe::exit_code::kNoDevice;
    }
    probe::JsonWriter w;
    w.begin_object().key("schema").string("probe.device_info/1").key("device");
    probe::write_device_json(w, info);
    w.end_object();
    std::printf("%s\n", w.str().c_str());
    return probe::exit_code::kOk;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return probe::exit_code::kUsage;
    }
    const std::string command = argv[1];
    const std::vector<std::string> rest(argv + 2, argv + argc);

    if (command == "--help") {
        print_usage(argv[0]);
        return probe::exit_code::kOk;
    }
    if (command == "--version") {
        std::printf("probe-workloads %s (%s; cuda architectures %s)\n", PROBE_VERSION,
                    PROBE_BUILD_TYPE, PROBE_CUDA_ARCHITECTURES);
        return probe::exit_code::kOk;
    }
    if (command == "--device-info") return print_device_info();
    if (command == "vector_add") return probe::run_vector_add(rest);

    std::fprintf(stderr, "probe-workloads: unknown command '%s'\n", command.c_str());
    print_usage(argv[0]);
    return probe::exit_code::kUsage;
}
