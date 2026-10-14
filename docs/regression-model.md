# The regression decision rule

`probe compare` answers one question per metric: did this run get slower than the baseline in a way that is both large enough to matter and too large to be noise? This document explains the rule (implemented in `src/probe/compare.py`) and why it has this shape.

## The two questions

**1. Is the change big enough to matter?** This is a percentage threshold `T` that the user chooses (`--fail-if-regression T`).

This is a judgment about cost, not statistics. A 1% change can be real and still not worth failing a build. The threshold also absorbs something the samples of a single run cannot reveal: run-to-run drift from GPU clocks, temperature and other processes. Two runs of identical code on the same machine typically differ by more than their within-run noise suggests.

**2. Is the change bigger than chance?** This is a noise floor derived from the samples. If nothing changed, the median of the current run would still differ a little from the baseline's median, simply because each was estimated from a finite number of iterations. The noise floor estimates how large that chance difference can plausibly be.

A change counts as a regression only if it passes both tests.

## The noise floor and the smallest detectable change

```
sigma          = 1.4826 * MAD                    robust spread of one iteration
se             = c(stat) * sigma / sqrt(n)       how much the statistic wobbles
noise_floor    = 3 * sqrt(se_b^2 + se_c^2)       estimated from the samples
min_detectable = max(noise_floor, clock_resolution)
```

- **MAD**, the median absolute deviation `median(|x - median(x)|)`, measures spread while ignoring a few extreme iterations: a context switch, a clock dip. The standard deviation would let one bad iteration inflate the floor and hide real regressions.
- **1.4826** converts MAD into a standard deviation for normally distributed data.
- **`c(stat) / sqrt(n)`** is the standard error of the statistic. For the mean, `c = 1`. For the q-quantile, asymptotic theory gives `c = sqrt(q(1-q)) / phi(z_q)` under a normal model: 1.2533 for the median and 2.113 for p95. Tail percentiles are noisier, so comparing p95 needs a larger change before Probe trusts it.
- **`sqrt(se_b^2 + se_c^2)`** is the standard error of the difference of two independent statistics.
- **3 standard errors.** If nothing changed and the normal approximation holds, the difference exceeds 3 standard errors about 0.3% of the time. With five metrics compared per run, that keeps false alarms rare. Two standard errors (about 5% each) would flag noise in CI regularly.
- **Clock resolution is not noise.** The noise floor is the only estimate of execution noise, and it comes entirely from the observed samples. Clock resolution is a property of the timer: CUDA events resolve time to about 0.5 µs (CUDA Runtime documentation for `cudaEventElapsedTime`), and `steady_clock` has a nominal 1 ns tick on Linux. It sets a lower limit on what can be measured at all. It only decides a verdict when the samples show almost no spread, for example identical samples with MAD = 0. Each metric uses its own clock: 0.0005 ms for the GPU phases and `gpu_total`, 1 ns for `host_end_to_end`. The host clock's real precision is worse than its tick, but that imprecision shows up as spread in the samples and therefore in the noise floor.
- **A minimum number of samples per statistic, on each side.** Below the minimum the verdict is INCONCLUSIVE; see the next section.

## Minimum samples per statistic

| Statistic | Minimum | Binding reason |
|---|---|---|
| p50 | 20 | stability of the MAD-based noise floor |
| mean | 20 | same |
| p95 | 100 | the estimate itself is unreliable with fewer |

**Why a high percentile needs far more data than the median.** The median is decided by the middle of the sample, where observations are densest. Half the data lies on each side of it, so every iteration helps pin it down. The 95th percentile is decided by the sparse upper tail: with n samples, only about `0.05 * n` of them lie above it. With 20 samples that is a single iteration, so "p95" is really "the second-slowest iteration", and one unlucky context switch moves it.

This can be made precise without assuming any distribution. A 95% confidence interval for the q-quantile can be built from the sorted samples. Its upper end is an observed sample only if `1 - q^n >= 0.975`, that is, `n >= ln(0.025) / ln(q)`:

- median (q = 0.5): n >= 6
- p95 (q = 0.95): n >= 72

With fewer than 72 samples, the data cannot even bound p95 from above. 100 leaves about five observations above the estimate. The normal-model standard error points the same way: the factor `c` is 2.113 for p95 against 1.2533 for the median, so p95 needs `(2.113 / 1.2533)^2`, about 2.8 times as many samples, for the same precision.

For the median and the mean, the estimate is fine at much smaller n. The weaker link is the noise floor, which needs sigma, estimated from MAD. For normal data a MAD-based sigma has a relative standard error of about `1.17 / sqrt(n)`: about 37% at n = 10 and 26% at n = 20. 20 keeps that uncertainty moderate, and it is probe-workloads' default iteration count. Comparing p95 therefore requires `--iters 100` or more.

## Verdicts

```
fewer than min(stat) samples on either side    -> INCONCLUSIVE   (p50 20, mean 20, p95 100)
|delta| <= min_detectable and |percent| >  T    -> INCONCLUSIVE   matters, but could be chance
|delta| <= min_detectable and |percent| <= T    -> PASS           no evidence of change
|delta| >  min_detectable and |percent| <= T    -> PASS           real, but too small to matter
|delta| >  min_detectable and percent   >  T    -> REGRESSION
|delta| >  min_detectable and percent   < -T    -> IMPROVEMENT
```

"Exceeds" is strict for both tests. A change exactly equal to the threshold, or exactly equal to `min_detectable`, does not count. The verdict's reason says whether the noise floor or the clock resolution was the binding bound.

INCONCLUSIVE exits with 0 unless `--fail-on-inconclusive` is given (then 8). Whether "we could not tell" should block a merge is a policy choice for the user, not a statistical one. The usual remedy is more iterations.

## Alternatives considered

**Percentage threshold only.** This is simple, but a noisy workload, such as one dominated by pageable host transfers, would trip it on chance alone.

**"Change must exceed 3 × MAD."** This compares the shift with the spread of individual iterations, not with the uncertainty of the median. It does not improve with more iterations. With 1,000 iterations and 2% per-iteration jitter, it would ignore a 5% median shift that is statistically unmistakable. The standard-error floor shrinks as `1/sqrt(n)`, so collecting more data makes the comparison sharper.

**A formal test** (Mann-Whitney U, bootstrap confidence intervals). This is more principled in some respects, but harder to explain, and it would still not see run-to-run drift. The threshold still has to carry that.

## Default gated metric: `gpu_total` p50

`gpu_total` is the CUDA-event interval from before the first H2D copy to after the last D2H copy.

- It catches kernel regressions, transfer regressions, and idle gaps between operations (for example an added synchronization). The kernel phase alone would miss the last two, and those are classes of slowdown Probe exists to find.
- It is measured on the GPU's own clock, so it excludes host scheduling jitter. That makes it steadier than `host_end_to_end`.

The trade-off is dilution. In `vector_add`, transfers dominate, so a 20% kernel regression may move `gpu_total` by only a few percent. The comparison table always shows a verdict for every phase. When the kernel is what you care about, gate on it explicitly with `--metric kernel`.

## Assumptions and known limits

- **Iterations are treated as independent.** Thermal drift or clock ramping during a run creates correlation, which makes the noise floor optimistic. Warmup reduces this; it does not remove it.
- **The standard-error factors assume roughly normal timing distributions** near the statistic. Real timings are often right-skewed. MAD keeps the floor robust to outliers, but the 0.3% figure is approximate.
- **One baseline run and one current run carry no information about run-to-run variance.** That is left to the threshold. Comparing against several baseline runs would be the next step if this proves to be a problem on hardware.
- **None of this has been calibrated on real GPU measurements yet.** The constants are reasoned defaults, to be revisited once real noise is observed.
