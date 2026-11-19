"""
MALARIA SCREENING DEVICE  (the live demo - Raspberry Pi or laptop)
==================================================================
Press Enter -> next blood-cell image -> CNN (ONNX) -> LED ON if parasites
-> short LLM explanation (hybrid mode, safety-checked) -> everything logged.

On a Raspberry Pi with an LED on GPIO17 the real LED switches.
On a laptop (no GPIO) the LED is shown in the terminal instead.

Usage (from the project root)
  python pi/device.py                                   # defaults below
  python pi/device.py --variant "Structured 50%"        # another CNN variant
  python pi/device.py --no-llm                          # CNN + LED only
  python pi/device.py --button-pin 27                   # physical button instead of Enter

Defaults
  CNN : outputs/deploy_study_tiny_cnn_scratch, variant "Structured 50% + INT8"
  LLM : models/own/qwen3-0.6b-imatrix-Q3_K_M.gguf  (our own 3-bit quantization)
Log : results/device_log.csv  (one row per image: prediction, truth, timings, CPU, RAM, temperature)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "explainer"))
from sysmon import read_cpu_temp  # noqa: E402  (psutil only)


# ----------------------------------------------------------------------
# LED: real GPIO on the Pi, printed in the terminal elsewhere
# ----------------------------------------------------------------------
class ConsoleLED:
    def __init__(self):
        self.is_lit = False

    def on(self):
        self.is_lit = True

    def off(self):
        self.is_lit = False


def make_led(pin: int):
    try:
        from gpiozero import LED
        led = LED(pin)
        return led, f"GPIO{pin}"
    except Exception:
        return ConsoleLED(), "terminal (no GPIO on this machine)"


def wait_for_trigger(button):
    """Return False if the user wants to quit."""
    if button is not None:
        print("\nPress the button (Ctrl+C to quit) ...")
        button.wait_for_press()
        return True
    cmd = input("\n[Enter] = next image, q = quit > ").strip().lower()
    return cmd != "q"


# ----------------------------------------------------------------------
# CNN
# ----------------------------------------------------------------------
def preprocess(path, img_size):
    """Exactly the training/eval preprocessing (src/dataset.py: preprocess_pil)."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]


def softmax(z, t=1.0):
    z = z / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_cnn(deploy_dir: Path, variant: str, threads: int):
    cfg = json.loads((deploy_dir / "deploy_config.json").read_text())
    if "models" in cfg:
        if variant not in cfg["models"]:
            sys.exit(f"Variant '{variant}' not found. Available: {list(cfg['models'])}")
        info = cfg["models"][variant]
        path, temp, thr = deploy_dir / info["file"], info["temperature"], info["led_threshold"]
    else:                                   # single-model deploy folder (first notebook)
        path, temp, thr = deploy_dir / cfg["model_fp32"], cfg["temperature_fp32"], cfg["led_threshold"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess, cfg, path, temp, thr


def load_images(folder: Path):
    labels = {}
    lab = folder / "labels.csv"
    if lab.exists():
        with open(lab, newline="") as f:
            labels = {r["file"]: r["class"] for r in csv.DictReader(f)}
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return files, labels


def main():
    ap = argparse.ArgumentParser(description="Malaria screening device demo")
    ap.add_argument("--deploy", default=str(ROOT / "outputs" / "deploy_study_tiny_cnn_scratch"))
    ap.add_argument("--variant", default="Structured 50% + INT8")
    ap.add_argument("--images", help="image folder (default: <deploy>/demo_images)")
    ap.add_argument("--llm", default=str(ROOT / "models" / "own" / "qwen3-0.6b-imatrix-Q3_K_M.gguf"))
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--led-pin", type=int, default=17)
    ap.add_argument("--button-pin", type=int, help="use a physical button on this GPIO pin instead of Enter")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--log", default=str(ROOT / "results" / "device_log.csv"))
    args = ap.parse_args()

    deploy = Path(args.deploy)
    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)

    # --- load everything once (startup cost is reported, not counted per image)
    t0 = time.perf_counter()
    sess, cfg, model_path, temp, thr = load_cnn(deploy, args.variant, args.threads)
    cnn_load_s = time.perf_counter() - t0
    img_size = cfg["img_size"]

    explainer = None
    llm_load_s = 0.0
    if not args.no_llm:
        if Path(args.llm).exists():
            from explain import Explainer
            explainer = Explainer(args.llm, n_threads=args.threads, mode="hybrid")
            llm_load_s = explainer.load_s
        else:
            print(f"[warn] LLM file not found ({args.llm}) - using template explanations")
            from explain import Explainer
            explainer = Explainer(None)

    # warm-up runs, so the first demo image shows the normal speed
    sess.run(None, {"input": np.zeros((1, 3, img_size, img_size), np.float32)})
    if explainer is not None and explainer.llm is not None:
        explainer.explain(0.9, thr)

    led, led_where = make_led(args.led_pin)
    led.off()
    button = None
    if args.button_pin is not None:
        from gpiozero import Button
        button = Button(args.button_pin)

    files, labels = load_images(Path(args.images) if args.images else deploy / "demo_images")
    if not files:
        sys.exit("No images found.")

    print("=" * 72)
    print("  MALARIA SCREENING DEVICE")
    print(f"  CNN : {cfg.get('arch', '?')} / {args.variant}  ({model_path.name}, "
          f"{model_path.stat().st_size / 1e6:.2f} MB, loaded in {cnn_load_s * 1000:.0f} ms)")
    print(f"  LLM : {Path(args.llm).name if explainer and explainer.llm else 'template only'}"
          f"{f' (loaded in {llm_load_s:.1f} s)' if llm_load_s else ''}")
    print(f"  LED : {led_where}   threshold P(parasitized) >= {thr:.3f}")
    print(f"  RAM : {proc.memory_info().rss / 1e6:.0f} MB   images: {len(files)}")
    print("=" * 72)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log_path.exists()
    logf = open(log_path, "a", newline="", encoding="utf-8")
    writer = None

    n_done = n_correct = 0
    i = 0
    try:
        while wait_for_trigger(button):
            path = files[i % len(files)]
            i += 1
            led.off()

            t_start = time.perf_counter()
            x = preprocess(path, img_size)
            t_pre = time.perf_counter()
            logits = sess.run(None, {"input": x})[0]
            t_cnn = time.perf_counter()
            p = float(softmax(logits, temp)[0, 1])
            positive = p >= thr
            if positive:
                led.on()
            t_led = time.perf_counter()

            truth = labels.get(path.name)
            pred = "Parasitized" if positive else "Uninfected"
            ok = (truth == pred) if truth else None
            n_done += 1
            n_correct += bool(ok)

            print(f"\nImage {i}: {path.name}" + (f"   (true label: {truth})" if truth else ""))
            status = "*** LED ON  - parasites detected ***" if positive else "    LED OFF - no parasites detected"
            print(f"  CNN: P(parasitized) = {p:.3f}  ->  {status}"
                  + ("" if ok is None else ("   [correct]" if ok else "   [WRONG]")))
            print(f"       preprocess {1000 * (t_pre - t_start):.1f} ms | model {1000 * (t_cnn - t_pre):.1f} ms | "
                  f"image-to-LED {1000 * (t_led - t_start):.1f} ms")

            exp = explainer.explain(p, thr) if explainer else None
            if exp:
                src = "LLM" if exp["source"] == "llm" else exp["source"]
                took = f", {exp['total_s']:.1f} s" if exp["total_s"] else ""
                print(f"  Explanation ({src}{took}):\n     {exp['text']}")

            cpu = psutil.cpu_percent(interval=None)
            temp_c = read_cpu_temp()
            rss = proc.memory_info().rss / 1e6
            print(f"  Device: CPU {cpu:.0f}% | RAM {rss:.0f} MB" + (f" | temp {temp_c:.1f} C" if temp_c else ""))

            row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": path.name, "true_label": truth,
                   "prob_parasitized": round(p, 4), "threshold": round(thr, 4), "led_on": positive,
                   "correct": ok, "preprocess_ms": 1000 * (t_pre - t_start), "model_ms": 1000 * (t_cnn - t_pre),
                   "image_to_led_ms": 1000 * (t_led - t_start),
                   "explanation_s": exp["total_s"] if exp else None,
                   "explanation_source": exp["source"] if exp else None,
                   "explanation": exp["text"] if exp else None,
                   "cpu_percent": cpu, "ram_mb": rss, "cpu_temp_c": temp_c,
                   "cnn_variant": args.variant, "cnn_file": model_path.name,
                   "llm_file": Path(args.llm).name if explainer and explainer.llm else None}
            if writer is None:
                writer = csv.DictWriter(logf, fieldnames=list(row.keys()))
                if new_file:
                    writer.writeheader()
            writer.writerow(row)
            logf.flush()
    except KeyboardInterrupt:
        pass
    finally:
        led.off()
        logf.close()

    if n_done:
        print(f"\nScreened {n_done} images; {n_correct} matched the true label. Log: {log_path}")


if __name__ == "__main__":
    main()
