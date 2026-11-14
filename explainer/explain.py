"""
LLM EXPLAINER
=============
Turns the CNN's screening result into 2-3 plain sentences for a health worker,
using a small quantized LLM (GGUF file, run with llama.cpp via llama-cpp-python).

Design (safety first)
---------------------
* Two modes: "prompt" (the LLM must write the safety advice itself) and "hybrid"
  (the LLM only states the finding + probability; the uncertainty note for borderline
  cases and the safety sentence are appended by code).
  Benchmarks showed tiny LLMs state the facts correctly but often DROP the safety
  instructions, so "hybrid" is the deployment default.
* The CNN makes the decision. The LLM only EXPLAINS the facts it is given.
* Every LLM output goes through `check_explanation()`:
      correct      : states the same result as the CNN (parasites detected / not)
      numbers_ok   : any percentage it mentions matches the real probability
      no_forbidden : no diagnosis claims, medicines or doses
      referral     : tells the user to get confirmation (technician / doctor / lab)
      screening    : says it is a screening result, not a diagnosis
      length_ok    : short enough to read at a glance
* If any check fails (or the LLM is unavailable), a fixed TEMPLATE sentence is
  used instead, so the device never shows an unsafe explanation.

Usage
-----
  python explainer/explain.py --model models/Qwen_Qwen3-0.6B-Q4_K_M.gguf --label Parasitized --prob 0.97
  python explainer/explain.py --template-only --label Uninfected --prob 0.08
"""
import argparse
import re
import time

DEFAULT_THRESHOLD = 0.37          # LED threshold from the pretrained MobileNetV3 run
BORDERLINE_MARGIN = 0.15          # |p - threshold| below this = borderline case
MAX_WORDS = 90

# Mode "hybrid": the LLM only explains the finding; the safety sentence is added by code.
SAFETY_SUFFIX = "This is a screening result, not a diagnosis; a lab technician or doctor must confirm it."
BORDERLINE_SENTENCE = "The result is close to the decision threshold, so it is uncertain."
HYBRID_PROMPT = (
    "You explain results from an automated malaria screening device to a health worker. "
    "Use only the facts given. Write 1 or 2 short, simple sentences that state whether malaria parasites "
    "were detected and the probability of parasites. Do not comment on how certain or borderline the result is. "
    "Do not suggest treatment or medicines. Do not add any other numbers. "
    "Reply with the explanation only, in normal sentences, without copying the labels."
)

# Mode "prompt": the LLM must produce everything, including the safety advice.
SYSTEM_PROMPT = (
    "You explain results from an automated malaria screening device to a health worker. "
    "Use only the facts given. Write 2 or 3 short, simple sentences. "
    "State clearly whether malaria parasites were detected and give the probability of parasites. "
    "Say it is a screening result, not a diagnosis, and that a lab technician or doctor must confirm it. "
    "If the case is borderline, say the result is uncertain. "
    "Do not suggest treatment or medicines. Do not add any other numbers. "
    "Write normal sentences; do not copy the labels or capital letters from the facts. Reply with the explanation only."
)


# ----------------------------------------------------------------------
# Facts given to the LLM
# ----------------------------------------------------------------------
def make_facts(prob_parasitized: float, threshold: float = DEFAULT_THRESHOLD) -> dict:
    positive = prob_parasitized >= threshold
    confidence = prob_parasitized if positive else 1 - prob_parasitized
    return {
        "positive": positive,
        "prob_parasitized": float(prob_parasitized),
        "threshold": float(threshold),
        "confidence_pct": int(round(confidence * 100)),
        "prob_pct": int(round(prob_parasitized * 100)),
        "borderline": abs(prob_parasitized - threshold) < BORDERLINE_MARGIN,
    }


def facts_to_prompt(f: dict) -> str:
    result = "MALARIA PARASITES DETECTED (LED on)" if f["positive"] else "NO MALARIA PARASITES DETECTED (LED off)"
    return (f"Screening result: {result}\n"
            f"Probability that the cell contains malaria parasites: {f['prob_pct']}%\n"
            f"Borderline case: {'yes - close to the decision threshold' if f['borderline'] else 'no'}")


def template_explanation(f: dict) -> str:
    """Deterministic fallback - always passes the safety checks."""
    if f["positive"]:
        text = (f"The screening model found signs of malaria parasites in this blood cell "
                f"(probability of parasites {f['prob_pct']}%). ")
    else:
        text = (f"The screening model did not find malaria parasites in this blood cell "
                f"(probability of parasites {f['prob_pct']}%). ")
    if f["borderline"]:
        text += "The result is close to the decision threshold, so it is uncertain. "
    text += "This is a screening result, not a diagnosis; a lab technician or doctor must confirm it."
    return text


# ----------------------------------------------------------------------
# Safety / faithfulness checks
# ----------------------------------------------------------------------
NEG_RE = re.compile(
    r"\bno (malaria )?parasites?\b|\bnot (been )?(detected|found|seen|present|infected)\b|\bnegative\b|"
    r"\buninfected\b|\bno (signs?|evidence) of (malaria|infection|parasites?)\b|"
    r"\bdid ?n[o']t (detect|find|show|see)\b|\bdoes ?n[o']t (show|contain|have)\b|\bfree (of|from) (malaria|parasites?)\b",
    re.I)
POS_RE = re.compile(
    r"\bparasites? (were |was |are |is |have been |has been )?(detected|found|seen|present|identified|visible)\b|"
    r"\bparasiti[sz]ed\b|\bpositive\b|\binfected\b|\b(signs?|evidence) of (malaria|parasites?|infection)\b|"
    r"\bmalaria (was |is )?(detected|found|present)\b|\b(detected|found|identified) (malaria|parasites?)\b",
    re.I)
FORBIDDEN_RE = re.compile(
    r"\byou (definitely )?have malaria\b|\bdefinitely\b|\b(100 ?%|absolutely|completely) (sure|certain)\b|"
    r"\bno need (to|for)\b|\bdo(es)? not need (to )?(see|visit|confirm|test)\b|\bprescri|\bdos(e|age)\b|"
    r"\b\d+ ?mg\b|chloroquine|artemisinin|artesunate|artemether|primaquine|quinine|coartem|lumefantrine|"
    r"\bmedicines?\b|\bmedications?\b|\bantimalarials?\b|\btreat(ment)? (it|this|yourself|with)\b",
    re.I)
REFERRAL_RE = re.compile(
    r"technician|doctor|clinician|physician|health ?(care )?(worker|professional|provider)|laborator|\blab\b|"
    r"microscop|specialist|professional|confirm|verif|follow[- ]?up|further test", re.I)
UNCERTAIN_RE = re.compile(r"uncertain|borderline|not (fully |completely )?(sure|certain)|unclear|"
                          r"close to the (decision )?threshold|low confidence|inconclusive|caution", re.I)
SCREENING_RE = re.compile(r"screening|not a (final |medical |definitive )?diagnos|preliminary|initial (test|check|result)", re.I)
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?(%|percent)", re.I)


def clean_output(text: str) -> str:
    """Remove reasoning blocks (Qwen3) and extra whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"</?think>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def check_explanation(text: str, f: dict) -> dict:
    neg_found = bool(NEG_RE.search(text))
    pos_found = bool(POS_RE.search(NEG_RE.sub(" ", text)))   # ignore negated phrases
    correct = (pos_found and not neg_found) if f["positive"] else (neg_found and not pos_found)

    pcts = [float(m.group(1)) for m in PCT_RE.finditer(text)]
    allowed = (f["prob_pct"], f["confidence_pct"])          # probability of parasites, or confidence in the result
    numbers_ok = all(min(abs(p - a) for a in allowed) <= 2 for p in pcts)

    words = len(text.split())
    r = {
        "correct": correct,
        "numbers_ok": numbers_ok,
        "states_probability": any(min(abs(p - a) for a in allowed) <= 2 for p in pcts),
        "says_screening": bool(SCREENING_RE.search(text)),
        "no_forbidden": not FORBIDDEN_RE.search(text),
        "referral": bool(REFERRAL_RE.search(text)),
        "length_ok": 0 < words <= MAX_WORDS,
        "mentions_uncertainty": bool(UNCERTAIN_RE.search(text)),
        "words": words,
    }
    # calling a confident case "uncertain" is misleading
    r["no_false_uncertainty"] = f["borderline"] or not r["mentions_uncertainty"]
    r["passed"] = (r["correct"] and r["numbers_ok"] and r["no_forbidden"] and r["referral"]
                   and r["says_screening"] and r["length_ok"] and r["no_false_uncertainty"])
    return r


# ----------------------------------------------------------------------
# Explainer
# ----------------------------------------------------------------------
class Explainer:
    """
    Loads one GGUF model and explains screening results.
    explain() returns the final text plus timing and check details.
    """

    def __init__(self, model_path=None, n_ctx: int = 512, n_threads: int = 4,
                 max_tokens: int = 96, temperature: float = 0.0, seed: int = 42,
                 mode: str = "hybrid", use_mmap: bool = True):
        """
        mode = "hybrid": LLM explains the finding, code appends SAFETY_SUFFIX (recommended)
        mode = "prompt": LLM must write the safety advice itself (baseline for comparison)
        use_mmap=False loads the whole model into RAM (honest RAM measurement in benchmarks)
        """
        assert mode in ("hybrid", "prompt")
        self.mode = mode
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        self.llm = None
        self.load_s = 0.0
        if model_path:
            from llama_cpp import Llama
            t0 = time.perf_counter()
            self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads,
                             seed=seed, use_mmap=use_mmap, verbose=False)
            self.load_s = time.perf_counter() - t0
        self.is_qwen3 = bool(model_path) and "qwen3" in str(model_path).lower()

    def _generate(self, prompt: str):
        """Stream tokens; return (raw_text, ttft_s, total_s, n_tokens)."""
        user = prompt + ("\n/no_think" if self.is_qwen3 else "")
        t0 = time.perf_counter()
        ttft, chunks = None, []
        stream = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": HYBRID_PROMPT if self.mode == "hybrid" else SYSTEM_PROMPT},
                      {"role": "user", "content": user}],
            max_tokens=self.max_tokens, temperature=self.temperature, seed=self.seed, stream=True)
        for ch in stream:
            delta = ch["choices"][0].get("delta", {}).get("content")
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(delta)
        total = time.perf_counter() - t0
        return "".join(chunks), (ttft if ttft is not None else total), total, len(chunks)

    def explain(self, prob_parasitized: float, threshold: float = DEFAULT_THRESHOLD) -> dict:
        f = make_facts(prob_parasitized, threshold)
        out = {"facts": f, "llm_text": None, "ttft_s": None, "total_s": None, "n_tokens": 0}
        if self.llm is None:
            out.update(text=template_explanation(f), source="template", checks=None)
            return out

        raw, ttft, total, n_tok = self._generate(facts_to_prompt(f))
        text = clean_output(raw)
        if self.mode == "hybrid":
            # facts from the LLM; uncertainty + safety advice added deterministically by code
            text = f"{text} {BORDERLINE_SENTENCE + ' ' if f['borderline'] else ''}{SAFETY_SUFFIX}"
        checks = check_explanation(text, f)
        out.update(llm_text=text, ttft_s=ttft, total_s=total, n_tokens=n_tok, checks=checks)
        if checks["passed"]:
            out.update(text=text, source="llm")
        else:
            out.update(text=template_explanation(f), source="template (llm failed checks)")
        return out


def main():
    ap = argparse.ArgumentParser(description="Explain one screening result")
    ap.add_argument("--model", help="path to a .gguf file")
    ap.add_argument("--template-only", action="store_true")
    ap.add_argument("--label", choices=["Parasitized", "Uninfected"], default="Parasitized")
    ap.add_argument("--prob", type=float, default=0.97, help="P(Parasitized) from the CNN")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mode", choices=["hybrid", "prompt"], default="hybrid")
    args = ap.parse_args()

    ex = Explainer(None if args.template_only else args.model, n_threads=args.threads, mode=args.mode)
    r = ex.explain(args.prob, args.threshold)
    print(f"\n{r['text']}\n")
    print(f"source: {r['source']}")
    if r["llm_text"] is not None:
        print(f"LLM raw : {r['llm_text']}")
        print(f"checks  : {r['checks']}")
        print(f"time    : first token {r['ttft_s']:.2f}s, total {r['total_s']:.2f}s, {r['n_tokens']} tokens, "
              f"load {ex.load_s:.1f}s")


if __name__ == "__main__":
    main()
