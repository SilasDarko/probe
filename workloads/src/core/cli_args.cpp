// Copyright (c) 2026 Silas Offei Darko. MIT License.
#include "probe/cli_args.hpp"

#include <charconv>
#include <limits>

namespace probe {

bool parse_u64(std::string_view text, std::uint64_t& out) {
    if (text.empty()) return false;
    const char* first = text.data();
    const char* last = text.data() + text.size();
    // from_chars rejects signs and whitespace for unsigned types and reports
    // overflow as result_out_of_range; we additionally require that it
    // consumed the whole string, so "12abc" is an error rather than 12.
    const auto [ptr, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} && ptr == last;
}

namespace {

// The largest grid CUDA allows in the x dimension is 2^31 - 1 blocks.
constexpr std::uint64_t kMaxGridX = std::numeric_limits<std::int32_t>::max();
constexpr std::uint64_t kMaxIterations = 100000;

bool is_power_of_two(std::uint64_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Checks shared by both 1-D workloads: one thread per element.
bool check_size_and_block(std::uint64_t n, std::uint64_t block_size, std::string& error) {
    if (n == 0) {
        error = "--n must be at least 1";
        return false;
    }
    // Threads are scheduled in warps of 32, so a block size that is not a
    // multiple of 32 leaves lanes idle in every block. 1024 is CUDA's limit.
    if (block_size < 32 || block_size > 1024 || block_size % 32 != 0) {
        error = "--block-size must be a multiple of 32 between 32 and 1024";
        return false;
    }
    if ((n + block_size - 1) / block_size > kMaxGridX) {
        error = "--n is too large for a one-dimensional grid at this block size";
        return false;
    }
    return true;
}

}  // namespace

bool parse_benchmark_args(const std::vector<std::string>& args, CommonOptions& common,
                          const std::vector<UintFlag>& workload_flags, std::string& error) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& flag = args[i];
        if (flag == "--json") {
            common.json = true;
            continue;
        }
        if (flag == "--raw-samples") {
            common.raw_samples = true;
            continue;
        }

        std::uint64_t* target = nullptr;
        if (flag == "--warmup") {
            target = &common.warmup;
        } else if (flag == "--iters") {
            target = &common.iterations;
        } else if (flag == "--seed") {
            target = &common.seed;
        } else {
            for (const UintFlag& f : workload_flags) {
                if (flag == f.flag) target = f.target;
            }
        }
        if (target == nullptr) {
            error = "unknown option: " + flag;
            return false;
        }
        if (i + 1 >= args.size()) {
            error = flag + " requires a value";
            return false;
        }
        if (!parse_u64(args[++i], *target)) {
            error = flag + " expects a non-negative integer, got '" + args[i] + "'";
            return false;
        }
    }

    if (common.iterations == 0 || common.iterations > kMaxIterations) {
        error = "--iters must be between 1 and " + std::to_string(kMaxIterations);
        return false;
    }
    if (common.warmup > kMaxIterations) {
        error = "--warmup must be at most " + std::to_string(kMaxIterations);
        return false;
    }
    if (common.seed > std::numeric_limits<std::uint32_t>::max()) {
        error = "--seed must fit in 32 bits";
        return false;
    }
    return true;
}

bool parse_vector_add_options(const std::vector<std::string>& args, VectorAddOptions& out,
                              std::string& error) {
    out = VectorAddOptions{};
    return parse_benchmark_args(args, out.common,
                                {{"--n", &out.n}, {"--block-size", &out.block_size}}, error) &&
           check_size_and_block(out.n, out.block_size, error);
}

bool parse_reduction_options(const std::vector<std::string>& args, ReductionOptions& out,
                             std::string& error) {
    out = ReductionOptions{};
    if (!parse_benchmark_args(args, out.common,
                              {{"--n", &out.n}, {"--block-size", &out.block_size}}, error) ||
        !check_size_and_block(out.n, out.block_size, error)) {
        return false;
    }
    // The tree reduction halves the active threads each step, which only
    // covers every element when the block size is a power of two.
    if (!is_power_of_two(out.block_size)) {
        error = "--block-size must be a power of two for reduction";
        return false;
    }
    return true;
}

}  // namespace probe
