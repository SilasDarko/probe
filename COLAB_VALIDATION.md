# Validating Probe on a Colab GPU

You need `Probe_GPU_Validation.ipynb` and `probe.bundle`. No GPU or CUDA is needed on your own computer.

1. Open <https://colab.research.google.com>, then **File → Upload notebook** and choose `Probe_GPU_Validation.ipynb`.
2. Choose **Runtime → Change runtime type → GPU** (T4 is enough), then **Connect**.
3. Choose **Runtime → Run all**.
4. When the upload button appears under Stage 2, select `probe.bundle`.
5. Wait for the last cell: about 30–60 minutes on a T4. Keep the tab open.
6. Download the `probe-stage5-colab-….tar.gz` the notebook offers. If the browser blocks the download, use the folder icon on the left and download the file from `/content`.

The last cell prints either **STAGE 5 HARDWARE VALIDATION FINISHED** or **STAGE 5 NOT COMPLETE** with a reason. Either way, the archive holds the evidence. Return it unmodified.

If the notebook stops early, the printed message says why: no GPU attached, a build failure, or a problem found by the first hardware run. The archive is still produced.

The full procedure behind the notebook is described in [docs/gpu-validation.md](docs/gpu-validation.md).
