"""Exit codes of the ``probe`` CLI.

These are the CLI's own codes; probe-workloads' codes are translated by the
runner (for example, its CUDA error code 1 becomes WORKLOAD_FAILED, so that 1
can mean "regression" here).
"""

OK = 0
REGRESSION = 1  # compare --fail-if-regression: the gated metric regressed
USAGE = 2  # bad command line (also argparse's own code)
NO_GPU = 3  # no usable CUDA device
INVALID_RESULT = 4  # a run completed but failed validation; never baselined or compared
INCOMPATIBLE = 5  # baseline and current run cannot be compared
BAD_FILE = 6  # baseline or result file missing, malformed, or inconsistent
WORKLOAD_FAILED = 7  # binary missing, CUDA error, or unusable output
INCONCLUSIVE = 8  # compare --fail-on-inconclusive: the gated metric was inconclusive


class ProbeError(Exception):
    """An error that ends the command with a specific exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
