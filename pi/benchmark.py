"""
DEVICE BENCHMARK  (runs on the laptop AND on the Raspberry Pi)
=============================================================
Evaluates the exported ONNX models on THIS machine and records
model metrics and system metrics, so the same table can be produced on
every device (Windows laptop, Pi 4, Pi 5, ...).

Needs only: numpy, pillow, onnxruntime, psutil  (no PyTorch)

For every deploy folder (e.g. outputs/deploy_pretrained, outputs/deploy_scratch)
and every precision (FP32, INT8) it measures:

  model metrics : accuracy, sensitivity, specificity, AUROC, ECE,
                  using the LED threshold + temperature from deploy_config.json
  system metrics: model file size, load time, RAM used by the model (fresh process per model),
                  single-image latency (mean / p50 / p95), FPS,
                  end-to-end time (read file + preprocess + model + decision),
                  CPU %, CPU temperature (Pi)

Images evaluated
  --data_root + --split_csv  -> the full test split (recommended)
  otherwise                  -> the 20 demo images inside each deploy folder

Usage (from the project root)
  python pi/benchmark.py --deploy outputs/deploy_pretrained outputs/deploy_scratch
  python pi/benchmark.py --deploy outputs/deploy_pretrained outputs/deploy_scratch \
                         --data_root data/cell_images --split_csv splits/patient_split.csv

Results: results/benchmark_<device>.csv  (+ printed table)
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sysmon import SystemMonitor, read_cpu_temp  # noqa: E402  (psutil only, no torch)


# ----------------------------------------------------------------------
# Same preprocessing as training (src/dataset.py: preprocess_pil)
# ----------------------------------------------------------------------
def preprocess(path, img_size):
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]          # (1, 3, H, W)


def softmax(logits, t=1.0):
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# Model metrics (numpy only)
# ----------------------------------------------------------------------
def auroc(labels, scores):
    """Rank-based AUROC (Mann-Whitney U)."""
    labels = np.asarray(labels)
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = np.asarray(scores)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ece(labels, p, n_bins=15):
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    total = 0.0
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf >= lo) & (conf <= hi) if k == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def model_metrics(labels, p, thr):
    labels = np.asarray(labels).astype(int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum()); tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum()); fn = int(((pred == 0) & (labels == 1)).sum())
    return {
        "n_images": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "auroc": auroc(labels, p),
        "ece": ece(labels, p),
        "fn": fn, "fp": fp,
    }


# ----------------------------------------------------------------------
# Image list
# ----------------------------------------------------------------------
def load_images(deploy_dir, data_root=None, split_csv=None, split="test"):
    if data_root and split_csv:
        import csv
        with open(split_csv, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == split]
        return [os.path.join(data_root, r["path"]) for r in rows], [int(r["label"]) for r in rows], f"{split} split"
    import csv
    demo = Path(deploy_dir) / "demo_images"
    with open(demo / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    return [str(demo / r["file"]) for r in rows], [int(r["label"]) for r in rows], "demo images"


# ----------------------------------------------------------------------
# Benchmark one model file
# ----------------------------------------------------------------------
def run_one(onnx_path, cfg, temperature, paths, labels, threads, runs, warmup):
    proc = psutil.Process(os.getpid())
    img_size = cfg["img_size"]

    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    model_ram_mb = (proc.memory_info().rss - rss0) / 1e6

    # 1) Model metrics + end-to-end time (file -> preprocess -> model -> LED decision)
    mon = SystemMonitor(interval=0.5).start()
    probs, e2e = [], []
    for p in paths:
        t = time.perf_counter()
        x = preprocess(p, img_size)
        logit = sess.run(None, {"input": x})[0]
        prob = softmax(logit, temperature)[0, 1]
        _led_on = prob >= cfg["led_threshold"]
        e2e.append((time.perf_counter() - t) * 1000)
        probs.append(prob)
    mon.stop()
    eval_sys = mon.summary()
    m = model_metrics(labels, np.array(probs), cfg["led_threshold"])

    # 2) Pure model latency on one fixed input (like repeated button presses)
    x = preprocess(paths[0], img_size)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    mon = SystemMonitor(interval=0.5).start()
    lat = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {"input": x})
        lat.append((time.perf_counter() - t) * 1000)
    mon.stop()
    lat_sys = mon.summary()
    lat = np.array(lat)

    return {
        **m,
        "size_mb": os.path.getsize(onnx_path) / 1e6,
        "load_ms": load_ms,
        "model_ram_mb": model_ram_mb,
        "latency_mean_ms": float(lat.mean()),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "fps": float(1000 / lat.mean()),
        "end_to_end_mean_ms": float(np.mean(e2e)),
        "cpu_percent_mean": lat_sys.get("cpu_percent_mean"),
        "proc_rss_max_mb": max(eval_sys.get("proc_rss_mb_max", 0), lat_sys.get("proc_rss_mb_max", 0)),
        "cpu_temp_max_c": max([v for v in (eval_sys.get("cpu_temp_c_max"), lat_sys.get("cpu_temp_c_max")) if v] or [None]),
    }


def device_name():
    model_file = "/proc/device-tree/model"           # e.g. "Raspberry Pi 5 Model B Rev 1.0"
    if os.path.exists(model_file):
        with open(model_file) as f:
            return f.read().strip("\x00").strip()
    return f"{platform.system()} {platform.machine()} ({socket.gethostname()})"


def main():
    ap = argparse.ArgumentParser(description="Benchmark ONNX models on this device")
    ap.add_argument("--deploy", nargs="+", required=True, help="one or more deploy_<exp> folders")
    ap.add_argument("--data_root", help="folder containing Parasitized/ and Uninfected/")
    ap.add_argument("--split_csv", help="e.g. splits/patient_split.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--variants", nargs="*", help="only these variants, e.g. FP32 \"Structured 30%%\"")
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    dev = device_name()
    print(f"Device : {dev}")
    print(f"CPU    : {psutil.cpu_count(logical=True)} logical cores, "
          f"RAM {psutil.virtual_memory().total / 1e9:.1f} GB, onnxruntime {ort.__version__}")
    temp = read_cpu_temp()
    if temp is not None:
        print(f"CPU temperature at start: {temp:.1f} C")

    rows = []
    for d in args.deploy:
        d = Path(d)
        cfg = json.loads((d / "deploy_config.json").read_text())
        paths, labels, source = load_images(d, args.data_root, args.split_csv, args.split)
        exp = cfg.get("experiment", d.name)
        if "models" in cfg:
            # compression-study folder: every variant has its own calibration
            todo = [(name, info["file"], info["temperature"], info["led_threshold"])
                    for name, info in cfg["models"].items()
                    if not args.variants or name in args.variants]
        else:
            # single-model folder: INT8 uses the FP32 temperature/threshold
            todo = [("FP32", cfg["model_fp32"], cfg["temperature_fp32"], cfg["led_threshold"]),
                    ("INT8", cfg["model_int8"], cfg["temperature_fp32"], cfg["led_threshold"])]
        for name, fname, temp, thr in todo:
            print(f"\n> {exp} / {name} on {len(paths)} {source} ...")
            run_cfg = {**cfg, "led_threshold": thr}
            # Each model runs in a fresh process so its RAM usage is measured cleanly
            with mp.get_context("spawn").Pool(1) as pool:
                r = pool.apply(run_one, (d / fname, run_cfg, temp, paths, labels,
                                         args.threads, args.runs, args.warmup))
            rows.append({"device": dev, "experiment": exp, "variant": name,
                         "threads": args.threads, "images": source, **r})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in dev)[:40]
    out = out_dir / f"benchmark_{safe}.csv"
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = ["experiment", "variant", "accuracy", "sensitivity", "specificity", "auroc",
            "size_mb", "latency_mean_ms", "fps", "end_to_end_mean_ms", "model_ram_mb",
            "cpu_percent_mean", "cpu_temp_max_c"]
    widths = {c: max(len(c), 12) for c in cols}
    widths["variant"] = max(len(str(r["variant"])) for r in rows) + 1
    widths["experiment"] = max(len(str(r["experiment"])) for r in rows) + 1
    print("\n" + " | ".join(f"{c:>{widths[c]}s}" for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:{widths[c]}.4f}" if isinstance(v, float) else f"{str(v):>{widths[c]}s}")
        print(" | ".join(cells))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
