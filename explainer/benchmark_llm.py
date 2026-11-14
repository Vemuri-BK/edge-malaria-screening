"""
LLM EXPLAINER BENCHMARK  (laptop and Raspberry Pi)
=================================================
For every GGUF model (different models and quantization levels) it runs the
same set of screening cases and records:

  explanation quality : pass rate (all safety checks), correct result, correct numbers,
                        referral present, forbidden content, mentions uncertainty on
                        borderline cases, average length
  system metrics      : file size, load time, RAM used by the model (fresh process),
                        time to first token, total time per explanation (mean / p95),
                        tokens per second, CPU %, CPU temperature (Pi)

Cases: a fixed, reproducible mix of confident-positive, confident-negative and
borderline results (probabilities around the LED threshold).

Usage (from the project root)
  python explainer/benchmark_llm.py --models models/*.gguf --cases 40
  python explainer/benchmark_llm.py --models models/Qwen_Qwen3-0.6B-Q4_K_M.gguf --cases 8   # quick test

Results
  results/llm_benchmark_<device>.csv        one row per model
  results/llm_outputs_<device>.jsonl        every explanation (for qualitative examples)
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "explainer"))


def make_cases(n: int, threshold: float, seed: int = 0):
    """Reproducible probabilities: 1/4 confident positive, 1/4 confident negative, 1/2 borderline."""
    rng = np.random.default_rng(seed)
    q = max(n // 4, 1)
    probs = np.concatenate([
        rng.uniform(0.90, 0.999, q),                                   # confident positive
        rng.uniform(0.001, 0.10, q),                                   # confident negative
        rng.uniform(threshold, min(threshold + 0.14, 0.99), q),        # borderline positive
        rng.uniform(max(threshold - 0.14, 0.01), threshold - 0.005, n - 3 * q),  # borderline negative
    ])
    return [round(float(p), 3) for p in probs]


def parse_quant(name: str) -> str:
    m = re.search(r"(I?Q\d[\w]*|F16|BF16)\.gguf$", name, re.I)
    return m.group(1).upper() if m else "?"


def run_model(model_path, cases, threshold, threads, n_ctx, max_tokens, mode="hybrid"):
    """Runs in a fresh process: load the model, explain every case, return summary + per-case rows."""
    import psutil
    from explain import Explainer
    from sysmon import SystemMonitor

    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    ex = Explainer(model_path, n_ctx=n_ctx, n_threads=threads, max_tokens=max_tokens,
                   mode=mode, use_mmap=False)   # no mmap: the whole model is in RAM -> honest RAM number
    model_ram = (proc.memory_info().rss - rss0) / 1e6

    ex.explain(0.9, threshold)                     # warm-up (not counted)

    mon = SystemMonitor(interval=0.5).start()
    rows = []
    for p in cases:
        r = ex.explain(p, threshold)
        c = r["checks"]
        rows.append({"prob": p, "positive": r["facts"]["positive"], "borderline": r["facts"]["borderline"],
                     "prob_pct": r["facts"]["prob_pct"], "llm_text": r["llm_text"],
                     "final_text": r["text"], "source": r["source"], "ttft_s": r["ttft_s"],
                     "total_s": r["total_s"], "n_tokens": r["n_tokens"], **c})
    mon.stop()
    sysm = mon.summary()

    def mean(k, subset=None):
        vals = [r[k] for r in (subset or rows)]
        return float(np.mean(vals)) if vals else float("nan")

    border = [r for r in rows if r["borderline"]]
    total = np.array([r["total_s"] for r in rows])
    toks = np.array([r["n_tokens"] for r in rows])
    summary = {
        "model": Path(model_path).name,
        "mode": mode,
        "quant": parse_quant(Path(model_path).name),
        "size_mb": os.path.getsize(model_path) / 1e6,
        "load_s": ex.load_s,
        "model_ram_mb": model_ram,
        "n_cases": len(rows),
        "pass_rate": mean("passed"),
        "correct_rate": mean("correct"),
        "numbers_ok_rate": mean("numbers_ok"),
        "states_probability_rate": mean("states_probability"),
        "says_screening_rate": mean("says_screening"),
        "referral_rate": mean("referral"),
        "forbidden_rate": 1 - mean("no_forbidden"),
        "uncertainty_on_borderline_rate": mean("mentions_uncertainty", border) if border else float("nan"),
        "no_false_uncertainty_rate": mean("no_false_uncertainty"),
        "fallback_rate": 1 - mean("passed"),
        "avg_words": mean("words"),
        "ttft_mean_s": mean("ttft_s"),
        "total_mean_s": float(total.mean()),
        "total_p95_s": float(np.percentile(total, 95)),
        "tokens_per_s": float(toks.sum() / total.sum()),
        "cpu_percent_mean": sysm.get("cpu_percent_mean"),
        "proc_rss_max_mb": sysm.get("proc_rss_mb_max"),
        "cpu_temp_max_c": sysm.get("cpu_temp_c_max"),
    }
    return summary, rows


def device_name():
    f = "/proc/device-tree/model"
    if os.path.exists(f):
        with open(f) as fh:
            return fh.read().strip("\x00").strip()
    return f"{platform.system()} {platform.machine()} ({socket.gethostname()})"


def main():
    ap = argparse.ArgumentParser(description="Benchmark GGUF explainer models on this device")
    ap.add_argument("--models", nargs="+", required=True, help="GGUF files (wildcards allowed)")
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.37)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--max_tokens", type=int, default=96)
    ap.add_argument("--mode", choices=["hybrid", "prompt", "both"], default="both",
                    help="hybrid = safety sentence added by code; prompt = LLM must write it")
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. hybrid_v2 (avoids overwriting)")
    ap.add_argument("--out_dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    paths = sorted({p for pat in args.models for p in glob.glob(pat)})
    if not paths:
        sys.exit("No GGUF files found - run explainer/download_models.py first")
    cases = make_cases(args.cases, args.threshold)
    dev = device_name()
    print(f"Device: {dev} | {len(paths)} models | {len(cases)} cases each | {args.threads} threads")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in dev)[:40] + (f"_{args.tag}" if args.tag else "")
    jsonl = open(out_dir / f"llm_outputs_{safe}.jsonl", "w", encoding="utf-8")

    modes = ["prompt", "hybrid"] if args.mode == "both" else [args.mode]
    summaries = []
    for p in paths:
        for mode in modes:
            print(f"\n> {Path(p).name} [{mode}]")
            t0 = time.time()
            with mp.get_context("spawn").Pool(1) as pool:      # fresh process -> clean RAM measurement
                s, rows = pool.apply(run_model, (p, cases, args.threshold, args.threads, args.n_ctx,
                                                 args.max_tokens, mode))
            s["device"] = dev
            summaries.append(s)
            for r in rows:
                jsonl.write(json.dumps({"device": dev, "model": s["model"], "mode": mode, **r}) + "\n")
            print(f"  pass {s['pass_rate']:.0%} | correct {s['correct_rate']:.0%} | "
                  f"{s['tokens_per_s']:.1f} tok/s | {s['total_mean_s']:.1f}s per explanation | "
                  f"RAM {s['model_ram_mb']:.0f} MB | ({time.time() - t0:.0f}s)")
    jsonl.close()

    import csv
    out = out_dir / f"llm_benchmark_{safe}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    cols = ["model", "mode", "size_mb", "pass_rate", "correct_rate", "numbers_ok_rate", "uncertainty_on_borderline_rate",
            "tokens_per_s", "ttft_mean_s", "total_mean_s", "model_ram_mb", "cpu_temp_max_c"]
    w0 = max(len(s["model"]) for s in summaries) + 1
    print("\n" + " | ".join([f"{'model':>{w0}s}"] + [f"{c:>12.12s}" for c in cols[1:]]))
    for s in summaries:
        cells = [f"{s['model']:>{w0}s}"]
        for c in cols[1:]:
            v = s[c]
            cells.append(f"{v:12.3f}" if isinstance(v, float) else f"{str(v):>12s}")
        print(" | ".join(cells))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
