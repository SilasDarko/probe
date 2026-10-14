"""Detection of NVIDIA hardware on the current machine."""

import shutil
import subprocess


def nvidia_gpu_available() -> bool:
    """Return True if ``nvidia-smi -L`` lists at least one GPU.

    ``nvidia-smi`` ships with the driver, so this answers "is there a GPU the
    driver can see?", independent of whether the CUDA toolkit is installed.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return False
    try:
        result = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout
