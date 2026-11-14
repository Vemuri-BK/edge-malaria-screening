"""
STEP 11 - COMPRESSION STUDY: evaluate every lightweight variant the same way
===========================================================================
Each variant (FP32, INT8, pruned, pruned+INT8, ...) is an ONNX file. For each:

  1. Run it on validation + test images with ONNX Runtime (exactly like the Pi)
  2. Calibrate it on VALIDATION (its own temperature + LED threshold) -
     this is what you would do before deploying any compressed model
  3. Measure it on TEST: AUROC, accuracy, sensitivity, specificity, ECE, FN, FP
  4. Count LED decisions that differ from the FP32 baseline
  5. Measure size (raw and gzip) and single-image CPU latency

The result is one row per variant -> one table per architecture.
"""
import numpy as np
import onnxruntime as ort

from compress import file_size_mb, gzip_size_mb
from config import TARGET_SENSITIVITY
from evaluate import benchmark_latency
from metrics import compute_metrics, fit_temperature, softmax_np, threshold_for_sensitivity


def onnx_logits(onnx_path, X: np.ndarray, threads: int = 4, batch: int = 256) -> np.ndarray:
    """X: float32 (N, 3, H, W) in [0, 1]."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"input": X[i:i + batch]})[0]
                           for i in range(0, len(X), batch)])


def evaluate_variant(name, onnx_path, Xv, yv, Xt, yt, led_reference=None,
                     extra: dict = None, threads: int = 4, latency_runs: int = 200) -> dict:
    lv = onnx_logits(onnx_path, Xv, threads)
    lt = onnx_logits(onnx_path, Xt, threads)

    T = fit_temperature(lv, yv)
    thr = threshold_for_sensitivity(yv, softmax_np(lv, T)[:, 1], TARGET_SENSITIVITY)
    p = softmax_np(lt, T)[:, 1]
    m = compute_metrics(yt, p, thr)
    led = p >= thr

    lat = benchmark_latency(onnx_path, threads=1, runs=latency_runs)
    row = {
        "variant": name,
        **(extra or {}),
        "onnx_mb": file_size_mb(onnx_path),
        "gzip_mb": gzip_size_mb(onnx_path),
        "auroc": m["auroc"],
        "accuracy": m["accuracy"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "ece": m["ece"],
        "fn": m["fn"],
        "fp": m["fp"],
        "led_flips_vs_fp32": int((led != led_reference).sum()) if led_reference is not None else 0,
        "temperature": T,
        "led_threshold": thr,
        "cpu_latency_ms_1thread": lat["latency_mean_ms"],
        "file": onnx_path.split("/")[-1] if isinstance(onnx_path, str) else onnx_path.name,
    }
    return row, led


def recommend(df, fp32_name: str = "FP32", max_sens_drop: float = 0.01, max_auroc_drop: float = 0.005):
    """
    Deployment rule: among variants that are clinically as safe as FP32
    (sensitivity within max_sens_drop, AUROC within max_auroc_drop),
    pick the fastest. Latency here is from THIS machine - confirm on the Pi.
    """
    ref = df.loc[fp32_name]
    safe = df[(df.sensitivity >= ref.sensitivity - max_sens_drop) & (df.auroc >= ref.auroc - max_auroc_drop)]
    return safe.sort_values("cpu_latency_ms_1thread"), safe["cpu_latency_ms_1thread"].idxmin()
