"""
Download the GGUF models used in the LLM study (from Hugging Face).

Plans
-----
  core : Qwen3-0.6B at 5 quantization levels  +  Llama-3.2-1B and Gemma-3-1B at Q4_K_M
         (~3.5 GB total)  -> answers both "how far can we quantize?" and "which model?"
  quick: Qwen3-0.6B Q4_K_M only (~0.5 GB) -> just to try the explainer

Files go to models/ (ignored by git). If a model needs a licence click-through on
Hugging Face (e.g. Gemma), log in with `huggingface-cli login` first.

Usage
  python explainer/download_models.py --plan quick
  python explainer/download_models.py --plan core
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "qwen3-0.6b":   "bartowski/Qwen_Qwen3-0.6B-GGUF",
    "llama3.2-1b":  "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "gemma3-1b":    "unsloth/gemma-3-1b-it-GGUF",
}

PLANS = {
    "quick": [("qwen3-0.6b", "Q4_K_M")],
    "core": [("qwen3-0.6b", q) for q in ("Q8_0", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K")]
            + [("llama3.2-1b", "Q4_K_M"), ("gemma3-1b", "Q4_K_M")],
}


def find_file(repo: str, quant: str) -> str:
    files = [f for f in list_repo_files(repo) if f.lower().endswith(f"-{quant.lower()}.gguf")]
    if not files:
        raise FileNotFoundError(f"No *-{quant}.gguf in {repo}")
    return sorted(files, key=len)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=list(PLANS), default="quick")
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, quant in PLANS[args.plan]:
        repo = MODELS[name]
        try:
            fname = find_file(repo, quant)
            path = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(out))
            print(f"OK   {name:12s} {quant:7s} -> {path}")
        except Exception as e:
            print(f"FAIL {name:12s} {quant:7s}: {e}")


if __name__ == "__main__":
    main()
