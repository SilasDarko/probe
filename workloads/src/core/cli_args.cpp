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

}  // namespace

bool parse_vector_add_options(const std::vector<std::string>& args, VectorAddOptions& out,
                              std::string& error) {
    out = VectorAddOptions{};
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& flag = args[i];
        if (flag == "--json") {
            out.json = true;
            continue;
        }

        std::uint64_t* target = nullptr;
        if (flag == "--n") {
            target = &out.n;
        } else if (flag == "--block-size") {
            target = &out.block_size;
        } else if (flag == "--warmup") {
            target = &out.warmup;
        } else if (flag == "--iters") {
            target = &out.iterations;
        } else if (flag == "--seed") {
            target = &out.seed;
        } else {
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

    if (out.n == 0) {
        error = "--n must be at least 1";
        return false;
    }
    // Threads are scheduled in warps of 32, so a block size that is not a
    // multiple of 32 leaves lanes idle in every block. 1024 is CUDA's limit.
    if (out.block_size < 32 || out.block_size > 1024 || out.block_size % 32 != 0) {
        error = "--block-size must be a multiple of 32 between 32 and 1024";
        return false;
    }
    if (out.iterations == 0 || out.iterations > kMaxIterations) {
        error = "--iters must be between 1 and " + std::to_string(kMaxIterations);
        return false;
    }
    if (out.seed > std::numeric_limits<std::uint32_t>::max()) {
        error = "--seed must fit in 32 bits";
        return false;
    }
    if ((out.n + out.block_size - 1) / out.block_size > kMaxGridX) {
        error = "--n is too large for a one-dimensional grid at this block size";
        return false;
    }
    return true;
}

}  // namespace probe
