# Correctness checks

Probe treats a timing as meaningful only if the output it timed was correct.
This document explains how each workload's output is checked and how far each
check can be trusted.

## Input domain (both workloads)

Inputs come from `fill_unit_magnitude` (`reference.hpp`).

- **How values are built.** Each value is built from one 32-bit `std::mt19937` output: bit 31 is the sign, and the low 23 bits select a magnitude `k * 2^-24` with `k` in `[2^23, 2^24)`.
- **Range.** Every input is a normal float32 with `|x|` in `[0.5, 1)`. Zero, subnormals, NaN and infinity cannot occur.
- **Reproducibility.** The C++ standard fixes `mt19937`'s output sequence, so a given seed gives the same inputs on every platform.
- **Granularity.** Every input is an exact multiple of `2^-24`.

## vector_add: exact equality

Each output `c[i] = a[i] + b[i]` is a single IEEE 754 addition. The standard requires it to be correctly rounded (round to nearest even), so the CPU (SSE) and the GPU compute the same value from the same operands. Three more facts close the remaining gaps:

- **No contraction.** Neither compiler can fuse or reassociate a lone addition. FMA contraction needs a multiply, and the build does not use `--use_fast_math`.
- **No subnormals.** Both operands are multiples of `2^-24` with magnitude below 1. Their sum is therefore zero or at least `2^-24` in magnitude, which is far above the subnormal range (`< 2^-126`). Flush-to-zero settings cannot cause a difference.
- **No overflow.** `|a + b| < 2`.

So exact equality (`rtol = 0`, `atol = 0`) is the right check for this input domain. It is **not** a general claim that CPU and GPU float results always match. They do not, once operations are fused, reordered, or involve transcendental functions.

## reduction: an analytical error bound

### Accumulation types

| Side | Accumulates in | How |
|---|---|---|
| GPU (`reduce_sum_kernel`) | float32 | tree in shared memory, relaunched on partial sums |
| CPU reference (`reduction_reference`) | double | sequential loop; each float32 input converts to double exactly |

The GPU result is compared against the higher-precision reference. Two float32 sums in different orders are never compared against each other, and neither is treated as ground truth.

### The bound

Let `s` be the exact sum, `u_f = 2^-24` (float32 unit roundoff), `u_d = 2^-53` (double), and `gamma(k, u) = k*u / (1 - k*u)`.

**GPU error.** A summation in which every input passes through at most `d` rounded additions satisfies `|gpu - s| <= gamma(d, u_f) * sum|x|`. This is the standard result for summation in floating point; see Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 4.

- Within one pass, an element goes through `log2(B)` tree levels.
- The kernel runs `P` passes, so `d = P * log2(B)`.
- Padding threads contribute exact zeros. Adding zero is exact, so counting those levels only makes the bound more conservative.

**Reference error.** Sequential double summation of `n` values performs `n - 1` additions: `|ref - s| <= gamma(n - 1, u_d) * sum|x|`.

**Combined.** By the triangle inequality,

```
|gpu - ref| <= (gamma(d, u_f) + gamma(n - 1, u_d)) * sum|x|
```

`sum|x|` is itself accumulated in double and may be low by a relative `gamma(n - 1, u_d)`, so Probe multiplies it by `(1 + gamma(n - 1, u_d))`. The result is `validation.atol` in every reduction JSON.

**Worked example** (block size 256, n = 1,000,003):

- P = 3 passes, so d = 24.
- `gamma(24, 2^-24)` is about 1.4e-6, and `gamma(1000002, 2^-53)` is about 1.1e-10. The GPU term dominates by four orders of magnitude.
- With `sum|x|` about 750,000, the bound is about 1.07.

### Assumptions

1. **IEEE arithmetic on the GPU.** Every float32 addition is IEEE 754 round-to-nearest. The kernel is built without `--use_fast_math`, and a reduction contains no multiplies to contract.
2. **No underflow.** Partial sums of multiples of `2^-24` are multiples of `2^-24`, so they never become subnormal. With arbitrary inputs, gradual underflow would add a small absolute error term the bound does not include.
3. **No overflow.** `|partial sum| <= n`, far below the float32 maximum for any n the grid allows.
4. **`d * u_f < 1`.** This is true by a huge margin: d is at most about 31 * 10.
5. **The depth is computed for the algorithm actually run.** The bound covers any summation tree of depth at most `d`, whatever its shape. A future variant that changes depth must recompute `d`; for example, a variant where each thread first sums several elements sequentially has a larger depth.

### Where the bound is too loose

The bound is a worst case. It never rejects a correct result, but it pays for that in sensitivity:

- **It scales with `sum|x|`, not `|s|`.** Inputs have random signs, so `|s|` grows like `sqrt(n)` while `sum|x|` grows like `n`. Typical real errors are far smaller than the bound.
- **A dropped or duplicated element.** Such an element shifts the sum by at least 0.5, the smallest input magnitude. That is detected only while `atol < 0.5`, which at block size 256 means n up to about 466,000. The C++ tests enforce detection at a fixed list of sizes below that, using a CPU model of the kernel. Above that size, one lost element can hide inside the rounding budget.
- **A dropped block of partial sums.** At n = 16,777,216, `atol` is about 18. A block of 256 random-sign inputs typically sums to a magnitude of about 12, so losing an entire block can also go unnoticed at that size.
- **Small systematic errors.** A wrong rounding mode, for example, produces errors far below the bound and is not detectable by this check.

Errors that this check catches reliably at every size include NaN or infinite results, garbage memory, missing passes, a result that was never copied back (the host value is reset to NaN before each iteration), and anything that changes the sum by more than `atol`.

A stronger check is possible without loosening the analysis. Inputs could be chosen so that every float32 partial sum is exact (small integers), which would allow exact comparison at any n. That is future work. The analytical bound stays as the primary check, because it covers the realistic random-input case.
