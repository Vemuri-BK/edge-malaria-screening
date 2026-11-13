"""
STEP 9 - ONNX EVALUATION AND BENCHMARKING
=========================================
Runs an ONNX model (FP32 or INT8) the same way the Raspberry Pi will, and
measures both model metrics and system metrics.

    onnx_predict()      -> logits for a list of images (uses preprocess_pil)
    benchmark_latency() -> single-image latency (mean / p50 / p95 ms), FPS,
                           RAM used by the session, CPU usage, CPU temperature
    file_size_mb()      -> model file size on disk
"""
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

from config import IMG_SIZE
from dataset import preprocess_pil
from sysmon import SystemMonitor


def make_session(onnx_path, threads: int = 4):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def file_size_mb(path) -> float:
    return os.path.getsize(path) / 1e6


def onnx_predict(session, image_paths, img_size: int = IMG_SIZE, batch: int = 64) -> np.ndarray:
    paths = list(image_paths)
    out = []
    for i in range(0, len(paths), batch):
        xs = []
        for p in paths[i:i + batch]:
            with Image.open(p) as img:
                xs.append(preprocess_pil(img, img_size))
        out.append(session.run(None, {"input": np.stack(xs)})[0])
    return np.concatenate(out)


def benchmark_latency(onnx_path, threads: int = 4, runs: int = 200, warmup: int = 20,
                      img_size: int = IMG_SIZE) -> dict:
    """Batch-1 latency, like one button press on the Pi."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    sess = make_session(onnx_path, threads)
    rss_after_load = proc.memory_info().rss

    x = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})

    mon = SystemMonitor(interval=0.5).start()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    mon.stop()
    s = mon.summary()

    times = np.array(times)
    return {
        "model": Path(onnx_path).name,
        "threads": threads,
        "size_mb": file_size_mb(onnx_path),
        "latency_mean_ms": float(times.mean()),
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "fps": float(1000 / times.mean()),
        "session_ram_mb": (rss_after_load - rss_before) / 1e6,
        "cpu_percent_mean": s.get("cpu_percent_mean"),
        "cpu_temp_max_c": s.get("cpu_temp_c_max"),
    }
