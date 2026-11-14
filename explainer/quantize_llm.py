"""
OUR OWN LLM QUANTIZATION  (plain vs importance-aware)
====================================================
Instead of downloading ready-made quantized files, we quantize the LLM ourselves
from the original 16-bit model with llama.cpp's own tools:

  1. PLAIN quantization        : llama-quantize base.gguf out.gguf Q4_K_M
                                 every weight is rounded the same way.
  2. IMPORTANCE-AWARE (imatrix): first run task-like text through the model
                                 (llama-imatrix) to measure which weights matter
                                 most for OUR task, then quantize those more carefully:
                                 llama-quantize --imatrix imatrix.dat base.gguf out.gguf Q4_K_M
     Same idea as calibration for the CNN (minmax vs percentile): the calibration
     data decides how the limited bits are spent.

Levels: Q8_0, Q5_K_M, Q4_K_M, Q3_K_M, Q2_K (imatrix matters most at 3-2 bits).

Requirements
  * The 16-bit base model:  python explainer/download_models.py --plan base
  * llama.cpp tools llama-quantize and llama-imatrix on PATH (or --bin_dir):
      Windows : winget install llama.cpp     (or unzip a release from github.com/ggml-org/llama.cpp/releases)
      Pi/Linux: build llama.cpp from source (cmake) - see README

Usage
  python explainer/quantize_llm.py --base models/Qwen_Qwen3-0.6B-bf16.gguf
  python explainer/benchmark_llm.py --models "models/own/*.gguf" --cases 40

Output
  models/own/qwen3-0.6b-plain-<TYPE>.gguf   and   models/own/qwen3-0.6b-imatrix-<TYPE>.gguf
  results/llm_quantization.csv              (time, size, bits per weight for each file)

Note for the report: the imatrix calibration text is built from task-like
prompts with DIFFERENT probability values (seed 123) than the benchmark cases
(seed 0), plus general malaria-screening sentences, to avoid tuning on the test set.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explainer"))
from explain import SYSTEM_PROMPT, facts_to_prompt, make_facts, template_explanation  # noqa: E402

LEVELS = ["Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
IMATRIX_LEVELS = ["Q4_K_M", "Q3_K_M", "Q2_K"]          # Q8/Q5 barely change with imatrix

GENERAL_TEXT = [
    "Malaria is caused by Plasmodium parasites that infect red blood cells.",
    "A thin blood smear stained with Giemsa shows parasites as dark purple dots or rings inside red blood cells.",
    "Microscopy by a trained laboratory technician remains the reference method for confirming malaria.",
    "An automated screening device can help prioritise samples, but it does not replace a diagnosis.",
    "Uninfected red blood cells appear evenly coloured, without dark inclusions.",
    "If a screening result is uncertain, the sample should be examined again by an expert.",
    "Health workers should follow local guidelines and refer patients for confirmation and care.",
    "A false negative means an infection was missed; a false positive means a healthy cell was flagged.",
    "The screening model reports the probability that a cell contains parasites.",
    "Results close to the decision threshold are less certain than results far from it.",
]


def build_calibration_text(path: Path, n: int = 300, threshold: float = 0.37, seed: int = 123):
    """Task-like text for llama-imatrix (prompts + reference explanations + general sentences)."""
    rng = np.random.default_rng(seed)
    probs = np.concatenate([rng.uniform(0.001, 0.2, n // 3), rng.uniform(0.2, 0.6, n // 3),
                            rng.uniform(0.6, 0.999, n - 2 * (n // 3))])
    parts = []
    for i, p in enumerate(probs):
        f = make_facts(float(p), threshold)
        parts.append(f"{SYSTEM_PROMPT}\n{facts_to_prompt(f)}\n{template_explanation(f)}")
        if i % 10 == 0:
            parts.append(" ".join(GENERAL_TEXT))
    path.write_text("\n\n".join(parts), encoding="utf-8")
    return path


def find_tool(name: str, bin_dir: str = None) -> str:
    candidates = [name, name + ".exe"]
    if bin_dir:
        for c in candidates:
            p = Path(bin_dir) / c
            if p.exists():
                return str(p)
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    sys.exit(f"Cannot find '{name}'. Install llama.cpp tools (Windows: winget install llama.cpp) "
             f"or pass --bin_dir to the folder that contains {name}.")


def run(cmd):
    print("  $ " + " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        sys.exit(f"Command failed: {cmd[0]}")
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description="Quantize the explainer LLM ourselves (plain + imatrix)")
    ap.add_argument("--base", required=True, help="16-bit GGUF (bf16/f16)")
    ap.add_argument("--name", default="qwen3-0.6b")
    ap.add_argument("--bin_dir", help="folder with llama-quantize / llama-imatrix")
    ap.add_argument("--out_dir", default=str(ROOT / "models" / "own"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--levels", nargs="+", default=LEVELS)
    ap.add_argument("--imatrix_levels", nargs="+", default=IMATRIX_LEVELS)
    ap.add_argument("--skip_imatrix", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    if not base.exists():
        sys.exit(f"Base model not found: {base} (run: python explainer/download_models.py --plan base)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    quantize = find_tool("llama-quantize", args.bin_dir)
    rows = []

    def record(method, qtype, path, seconds):
        size = path.stat().st_size
        rows.append({"model": path.name, "method": method, "type": qtype, "size_mb": size / 1e6,
                     "size_vs_base": size / base.stat().st_size, "quantize_s": seconds})
        print(f"  -> {path.name}: {size / 1e6:.0f} MB ({size / base.stat().st_size:.0%} of base), {seconds:.0f}s")

    # 1) plain quantization
    print(f"\n[1] Plain quantization of {base.name}")
    for q in args.levels:
        dst = out / f"{args.name}-plain-{q}.gguf"
        record("plain", q, dst, run([quantize, base, dst, q, args.threads]))

    # 2) importance-aware (imatrix) quantization
    if not args.skip_imatrix:
        imatrix_tool = find_tool("llama-imatrix", args.bin_dir)
        calib = build_calibration_text(out / "imatrix_calibration.txt")
        imat = out / "imatrix_task.dat"
        print(f"\n[2] Importance matrix from task-like text ({calib.stat().st_size / 1e3:.0f} kB)")
        t = run([imatrix_tool, "-m", base, "-f", calib, "-o", imat, "-c", 512, "-t", args.threads])
        print(f"  -> {imat.name} in {t:.0f}s")
        for q in args.imatrix_levels:
            dst = out / f"{args.name}-imatrix-{q}.gguf"
            record("imatrix", q, dst, run([quantize, "--imatrix", imat, base, dst, q, args.threads]))

    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    with open(res / "llm_quantization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {res / 'llm_quantization.csv'}")
    print('Next: python explainer/benchmark_llm.py --models "models/own/*.gguf" --cases 40')


if __name__ == "__main__":
    main()
