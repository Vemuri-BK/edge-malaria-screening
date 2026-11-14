"""
Builds the two Kaggle notebooks:

  notebooks/edai_malaria_kaggle.ipynb             (MAIN) pulls the code from GitHub
  notebooks/edai_malaria_kaggle_standalone.ipynb  (backup) code embedded with %%writefile,
                                                   runs without GitHub

Run from the project root after editing src/:  python notebooks/build_notebook.py
"""
import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO_URL = "https://github.com/Vemuri-BK/edge-malaria-screening.git"
BRANCH = "main"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


def module(name, title):
    md(f"### `src/{name}` — {title}")
    cells.append(nbf.v4.new_code_cell(f"%%writefile src/{name}\n" + (SRC / name).read_text()))


# ----------------------------------------------------------------------
# Experiments (each notebook = one experiment; outputs never overwrite each other)
# ----------------------------------------------------------------------
PRETRAINED_EXP = dict(name="pretrained", arch="mobilenet_v3_small", pretrained=True, epochs=15, lr=1e-3)
SCRATCH_EXP = dict(name="scratch", arch="mobilenet_v3_small", pretrained=False, epochs=100, lr=3e-3)

EXP_NOTE = {
    "pretrained": """
## Experiment settings — **pretrained** (transfer learning)
MobileNetV3-Small starts from ImageNet weights and is fine-tuned for 15 epochs.
Outputs go to `outputs/pretrained/` and `outputs/deploy_pretrained/`.
""",
    "scratch": """
## Experiment settings — **from scratch**
Same MobileNetV3-Small, but **random initial weights** (no ImageNet) and **100 epochs** with a higher
learning rate. Compare with the pretrained run on: test metrics, training time (GPU cost), INT8 robustness,
and speed on the laptop / Raspberry Pi (the architecture is identical, so speed should match - accuracy may not).
Outputs go to `outputs/scratch/` and `outputs/deploy_scratch/`.
""",
}

EXP_TEMPLATE = """
EXP = "{name}"            # names the output folders: outputs/<EXP>/ and outputs/deploy_<EXP>/
C.ARCH = "{arch}"
C.PRETRAINED = {pretrained}
C.EPOCHS = {epochs}
C.LR = {lr}

OUT = os.path.join(WORK, "outputs", EXP)
DEPLOY = os.path.join(WORK, "outputs", "deploy_" + EXP)
os.makedirs(OUT, exist_ok=True); os.makedirs(DEPLOY, exist_ok=True)
print(f"Experiment '{{EXP}}': {{C.ARCH}}, pretrained={{C.PRETRAINED}}, epochs={{C.EPOCHS}}, lr={{C.LR}}")
"""

INTRO = """
# EdAi Project — Malaria Screening on a Raspberry Pi (Phase 1: training on Kaggle)

**Device goal:** a Raspberry Pi classifies a blood-smear cell image; **the LED turns ON when the cell is Parasitized.**

**Research question:** when the model is compressed (INT8) for the edge, does it stay clinically reliable
(sensitivity, calibration) — and what does it cost in system resources?

This notebook:
1. {step1}
2. Builds patient-level splits (no patient in both train and test)
3. Trains MobileNetV3-Small while logging **model metrics** and **system metrics**
4. Calibrates the model (temperature scaling) and picks the **LED threshold**
5. Exports to ONNX, quantizes to INT8, compares FP32 vs INT8
6. Saves everything the Pi needs to `outputs/deploy/`

**Before running:** Add Input → *Malaria Cell Images Dataset* (iarunava) · Accelerator → **GPU** · Internet → **On**.
"""

COMMON_SETUP = r"""
# onnx / onnxruntime are needed for export + quantization
try:
    import onnx, onnxruntime
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime"], check=True)

def find_data_root(base="/kaggle/input"):
    # folder that contains Parasitized/ and Uninfected/ (shallowest copy)
    hits = sorted(glob.glob(os.path.join(base, "**", "Parasitized"), recursive=True),
                  key=lambda p: p.count(os.sep))
    if not hits:
        raise FileNotFoundError("Malaria dataset not found - add it with 'Add Input'.")
    return os.path.dirname(hits[0])

DATA_ROOT = find_data_root()
print("Working dir:", WORK)
print("Data root  :", DATA_ROOT)
"""


def github_setup(intro=None):
    md(intro or INTRO.format(step1="Pulls the latest project code from GitHub (`src/`, `splits/`)"))
    md(f"""
## 0. Get the code from GitHub
Code lives in **[{REPO_URL.removesuffix('.git')}]({REPO_URL.removesuffix('.git')})**.
Change code on your laptop → `git push` → re-run this notebook. Each run records the commit it used.

*Private repo?* Add a Kaggle secret named `GITHUB_TOKEN` (Add-ons → Secrets) with a GitHub personal access token.
""")
    code(f"""
import os, sys, glob, json, shutil, subprocess

REPO_URL = "{REPO_URL}"
BRANCH = "{BRANCH}"

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working/edge-malaria-screening" if ON_KAGGLE else os.path.abspath("edge-malaria-screening")

def git(*args, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = r.stderr.replace(TOKEN, "***") if TOKEN else r.stderr
        raise RuntimeError("git " + args[0] + " failed: " + msg)
    return r.stdout.strip()

# Optional token for a private repo (Kaggle secret GITHUB_TOKEN)
TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass
url = REPO_URL.replace("https://", "https://" + TOKEN + "@") if TOKEN else REPO_URL

if os.path.isdir(os.path.join(WORK, ".git")):          # re-run: update to latest
    git("fetch", "-q", "origin", BRANCH, cwd=WORK)
    git("reset", "-q", "--hard", "origin/" + BRANCH, cwd=WORK)
else:
    git("clone", "-q", "--depth", "1", "-b", BRANCH, url, WORK)

os.chdir(WORK)
os.makedirs("outputs", exist_ok=True)
GIT_COMMIT = git("log", "-1", "--format=%h %s (%cd)", "--date=short")
print("Code version:", GIT_COMMIT)
print(git("ls-files", "src"))
""" + COMMON_SETUP)


def standalone_setup():
    md(INTRO.format(step1="Writes the project modules to `src/` (one module per step) — no GitHub needed"))
    md("## 0. Setup")
    code(r"""
import os, sys, glob, json, shutil, subprocess

ON_KAGGLE = os.path.exists("/kaggle/working")
WORK = "/kaggle/working/EdAi_project" if ON_KAGGLE else os.path.abspath("EdAi_project_run")
for d in ("src", "splits", "outputs"):
    os.makedirs(os.path.join(WORK, d), exist_ok=True)
os.chdir(WORK)
GIT_COMMIT = "standalone notebook"
""" + COMMON_SETUP)
    md("## 1. Project modules\nEach cell below writes one file of the project. The same files are in the GitHub repo.")
    module("config.py", "all settings in one place")
    module("data_prep.py", "Step 1: patient IDs, de-duplication, splits")
    module("dataset.py", "Step 2: preprocessing and augmentation")
    module("models.py", "Step 3: model")
    module("metrics.py", "Step 5: model metrics, temperature scaling, LED threshold")
    module("sysmon.py", "system metrics monitor (Kaggle and Pi)")
    module("train.py", "Step 4: training loop")
    module("export_onnx.py", "Step 7: ONNX export")
    module("quantize.py", "Step 8: INT8 quantization")
    module("evaluate.py", "Step 9: ONNX evaluation and latency benchmark")


def pipeline(exp):
    """Cells shared by both notebooks (steps 1-9)."""
    md("## 2. Imports and system information")
    code(r'''
    sys.path.insert(0, os.path.join(WORK, "src"))
    import numpy as np, pandas as pd, torch
    import matplotlib.pyplot as plt
    from PIL import Image

    import config as C
    from dataset import build_loaders, build_transforms, preprocess_pil
    from models import EdgeClassifier, count_parameters, model_size_mb
    from metrics import compute_metrics, softmax_np, fit_temperature, threshold_for_sensitivity, expected_calibration_error
    from sysmon import system_info
    from train import set_seed, fit, predict_logits
    from export_onnx import export_onnx, check_parity
    from quantize import quantize_int8
    from evaluate import make_session, onnx_predict, benchmark_latency, file_size_mb

    SYS_TRAIN = system_info()
    for k, v in SYS_TRAIN.items():
        print(f"{k:20s}: {v}")
    ''')

    md(EXP_NOTE[exp["name"]])
    code(EXP_TEMPLATE.format(**exp))

    md("## 3. Step 1 — Data preparation (patient-level split)")
    code(r'''
    !python src/data_prep.py --data_root "{DATA_ROOT}"
    ''')

    md("## 4. Step 2 — Dataset and preprocessing check")
    code(r'''
    set_seed()
    SPLIT_CSV = os.path.join(WORK, "splits", "patient_split.csv")
    loaders, parts = build_loaders(SPLIT_CSV, DATA_ROOT)
    for k, df in parts.items():
        print(f"{k:5s}: {len(df):6d} images, {df.patient_id.nunique():3d} patients, "
              f"{(df.label == 1).mean():.1%} parasitized")

    # Sample images
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
    for row, label in enumerate([1, 0]):
        sample = parts["train"][parts["train"].label == label].sample(6, random_state=0)
        for ax, p in zip(axes[row], sample.path):
            ax.imshow(Image.open(os.path.join(DATA_ROOT, p))); ax.axis("off")
        axes[row][0].set_title(C.CLASSES[label], loc="left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "samples.png"), dpi=120); plt.show()

    # The Pi uses preprocess_pil (numpy). Check it equals the torchvision eval transform.
    p = os.path.join(DATA_ROOT, parts["test"].path[0])
    with Image.open(p) as img:
        a = preprocess_pil(img)
        b = build_transforms(train=False)(img.convert("RGB")).numpy()
    print("Max difference numpy vs torchvision preprocessing:", float(np.abs(a - b).max()))
    ''')

    md("## 5. Step 3 — Model")
    code(r'''
    model = EdgeClassifier(C.ARCH, pretrained=C.PRETRAINED)
    print(f"Architecture : {C.ARCH}")
    print(f"Parameters   : {count_parameters(model) / 1e6:.2f} M")
    print(f"FP32 size    : {model_size_mb(model):.2f} MB")
    ''')

    md("## 6. Step 4 — Training (model metrics + system metrics per epoch)")
    code(r'''
    import time
    t_start = time.time()
    model, hist = fit(model, loaders, OUT, epochs=C.EPOCHS, lr=C.LR)
    TOTAL_TRAIN_MIN = (time.time() - t_start) / 60
    print(f"\nTotal training time: {TOTAL_TRAIN_MIN:.1f} min")
    hist
    ''')

    code(r'''
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    e = hist.epoch
    ax[0,0].plot(e, hist.train_loss, label="train"); ax[0,0].plot(e, hist.val_loss, label="val"); ax[0,0].set_title("Loss"); ax[0,0].legend()
    for k in ["val_accuracy", "val_sensitivity", "val_specificity", "val_auroc"]:
        ax[0,1].plot(e, hist[k], label=k.replace("val_", ""))
    ax[0,1].set_title("Validation metrics"); ax[0,1].legend()
    ax[0,2].plot(e, hist.lr); ax[0,2].set_title("Learning rate")
    ax[1,0].plot(e, hist.epoch_time_s); ax[1,0].set_title("Epoch time (s)")
    ax[1,1].plot(e, hist.train_throughput_img_s); ax[1,1].set_title("Training throughput (img/s)")
    if hist.gpu_util_mean.notna().any():
        ax[1,2].plot(e, hist.gpu_util_mean, label="GPU util %")
    ax[1,2].plot(e, hist.cpu_percent_mean, label="CPU %"); ax[1,2].set_title("Hardware utilisation"); ax[1,2].legend()
    for a in ax.flat: a.set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120); plt.show()
    ''')

    md("""
    ## 7. Steps 5–6 — Test metrics, calibration and the LED threshold
    * **Temperature scaling** (fit on validation) fixes over/under-confidence without changing predictions at 0.5.
    * **LED threshold**: highest P(Parasitized) threshold that still catches ≥ `TARGET_SENSITIVITY` of infected cells on validation.
    """)
    code(r'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_logits, val_labels, _ = predict_logits(model, loaders["val"], device)
    test_logits, test_labels, _ = predict_logits(model, loaders["test"], device)

    T_FP32 = fit_temperature(val_logits, val_labels)
    val_p = softmax_np(val_logits, T_FP32)[:, 1]
    LED_THRESHOLD = threshold_for_sensitivity(val_labels, val_p, C.TARGET_SENSITIVITY)
    print(f"Temperature T = {T_FP32:.3f}   LED threshold = {LED_THRESHOLD:.3f}")

    p_raw = softmax_np(test_logits)[:, 1]
    p_cal = softmax_np(test_logits, T_FP32)[:, 1]
    torch_results = pd.DataFrame([
        {"setting": "raw, threshold 0.5", **compute_metrics(test_labels, p_raw, 0.5)},
        {"setting": "calibrated, threshold 0.5", **compute_metrics(test_labels, p_cal, 0.5)},
        {"setting": f"calibrated, LED threshold", **compute_metrics(test_labels, p_cal, LED_THRESHOLD)},
    ]).set_index("setting")
    torch_results.round(4)
    ''')

    code(r'''
    from sklearn.metrics import roc_curve

    def reliability(ax, labels, p, title):
        pred = (p >= 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p); corr = (pred == labels)
        bins = np.linspace(0.5, 1, 11); mids, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if m.sum() > 0: mids.append(conf[m].mean()); accs.append(corr[m].mean())
        ax.plot([0.5, 1], [0.5, 1], "k--", lw=1); ax.plot(mids, accs, "o-")
        ax.set_title(f"{title}\nECE = {expected_calibration_error(labels, p):.4f}")
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")

    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    r = torch_results.loc["calibrated, LED threshold"]
    cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
    ax[0].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): ax[0].text(j, i, int(v), ha="center", va="center", fontsize=14)
    ax[0].set_xticks([0, 1], ["LED off", "LED on"]); ax[0].set_yticks([0, 1], C.CLASSES); ax[0].set_title("Test confusion (LED threshold)")
    fpr, tpr, _ = roc_curve(test_labels, p_cal)
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], "k--", lw=1); ax[1].set_title(f"ROC (AUROC {r.auroc:.4f})"); ax[1].set_xlabel("1 - specificity"); ax[1].set_ylabel("sensitivity")
    reliability(ax[2], test_labels, p_raw, "Reliability: raw")
    reliability(ax[3], test_labels, p_cal, "Reliability: temperature-scaled")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "test_evaluation.png"), dpi=120); plt.show()
    ''')

    md("## 8. Step 7 — Export to ONNX")
    code(r'''
    FP32_ONNX = os.path.join(DEPLOY, "model_fp32.onnx")
    export_onnx(model, FP32_ONNX)
    print(f"Saved {FP32_ONNX}  ({file_size_mb(FP32_ONNX):.2f} MB)")
    print("Max |PyTorch - ONNX| logit difference:", check_parity(model, FP32_ONNX))
    ''')

    md("## 9. Step 8 — INT8 static quantization")
    code(r'''
    INT8_ONNX = os.path.join(DEPLOY, "model_int8.onnx")
    calib = parts["train"].sample(min(C.CALIB_IMAGES, len(parts["train"])), random_state=C.SEED)
    calib_paths = [os.path.join(DATA_ROOT, p) for p in calib.path]
    quantize_int8(FP32_ONNX, INT8_ONNX, calib_paths)
    print(f"FP32: {file_size_mb(FP32_ONNX):.2f} MB  ->  INT8: {file_size_mb(INT8_ONNX):.2f} MB")
    ''')

    md("""
    ## 10. Step 9 — FP32 vs INT8: model metrics and system metrics
    Both ONNX models run exactly as on the Pi (numpy preprocessing + ONNX Runtime on CPU).
    INT8 is evaluated (a) with the FP32 temperature and LED threshold — what you get if you just deploy it —
    and (b) re-calibrated with its own temperature.
    """)
    code(r'''
    val_paths = [os.path.join(DATA_ROOT, p) for p in parts["val"].path]
    test_paths = [os.path.join(DATA_ROOT, p) for p in parts["test"].path]
    yv, yt = parts["val"].label.values, parts["test"].label.values

    logits = {}
    for name, path in [("FP32", FP32_ONNX), ("INT8", INT8_ONNX)]:
        sess = make_session(path, threads=os.cpu_count())
        logits[name] = (onnx_predict(sess, val_paths), onnx_predict(sess, test_paths))

    T_INT8 = fit_temperature(logits["INT8"][0], yv)
    rows = []
    for name, T in [("FP32", T_FP32), ("INT8 (FP32 calibration)", T_FP32), ("INT8 (re-calibrated)", T_INT8)]:
        key = name.split()[0]
        p = softmax_np(logits[key][1], T)[:, 1]
        m = compute_metrics(yt, p, LED_THRESHOLD)
        m["ece_raw"] = expected_calibration_error(yt, softmax_np(logits[key][1])[:, 1])
        rows.append({"model": name, "temperature": T, **m})
    onnx_results = pd.DataFrame(rows).set_index("model")

    led_fp32 = softmax_np(logits["FP32"][1], T_FP32)[:, 1] >= LED_THRESHOLD
    led_int8 = softmax_np(logits["INT8"][1], T_FP32)[:, 1] >= LED_THRESHOLD
    print(f"T_FP32 = {T_FP32:.3f}, T_INT8 = {T_INT8:.3f}")
    print(f"LED decisions that change after INT8: {(led_fp32 != led_int8).sum()} of {len(yt)}")
    onnx_results[["temperature", "accuracy", "sensitivity", "specificity", "auroc", "ece_raw", "ece", "fn", "fp"]].round(4)
    ''')

    code(r'''
    # Latency / resources on this machine's CPU (reference only - the real numbers come from the Pi)
    bench = pd.DataFrame([benchmark_latency(p, threads=t)
                          for p in (FP32_ONNX, INT8_ONNX) for t in (1, 4)])
    bench.round(3)
    ''')

    md("## 11. Save results and deployment files")
    code(r'''
    deploy_config = {
        "experiment": EXP,
        "model_fp32": "model_fp32.onnx",
        "model_int8": "model_int8.onnx",
        "classes": C.CLASSES,
        "positive_class_index": 1,
        "img_size": C.IMG_SIZE,
        "input": "RGB, resized to img_size x img_size (bilinear), float32 in [0,1], NCHW; normalization is inside the model",
        "temperature_fp32": T_FP32,
        "temperature_int8": T_INT8,
        "led_threshold": LED_THRESHOLD,
        "led_rule": "LED ON if softmax(logits / temperature)[1] >= led_threshold",
        "target_sensitivity": C.TARGET_SENSITIVITY,
        "arch": C.ARCH,
        "pretrained": C.PRETRAINED,
        "epochs": C.EPOCHS,
    }
    with open(os.path.join(DEPLOY, "deploy_config.json"), "w") as f:
        json.dump(deploy_config, f, indent=2)

    # A few test images to copy to the Pi for the demo (balanced)
    demo_dir = os.path.join(DEPLOY, "demo_images"); os.makedirs(demo_dir, exist_ok=True)
    demo = pd.concat([parts["test"][parts["test"].label == l].sample(10, random_state=1) for l in (0, 1)])
    for p in demo.path:
        shutil.copy(os.path.join(DATA_ROOT, p), os.path.join(demo_dir, os.path.basename(p)))
    demo.assign(file=demo.path.map(os.path.basename))[["file", "class", "label", "patient_id"]].to_csv(
        os.path.join(demo_dir, "labels.csv"), index=False)

    summary = {
        "experiment": EXP,
        "code_version": GIT_COMMIT,
        "system_training": SYS_TRAIN,
        "total_training_min": TOTAL_TRAIN_MIN,
        "params_m": count_parameters(model) / 1e6,
        "torch_test_metrics": torch_results.reset_index().to_dict("records"),
        "onnx_test_metrics": onnx_results.reset_index().to_dict("records"),
        "cpu_benchmark": bench.to_dict("records"),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    onnx_results.to_csv(os.path.join(OUT, "onnx_results.csv"))
    bench.to_csv(os.path.join(OUT, "cpu_benchmark.csv"), index=False)

    zip_base = os.path.join("/kaggle/working" if ON_KAGGLE else WORK, "edai_outputs_" + EXP)
    shutil.make_archive(zip_base, "zip", os.path.join(WORK, "outputs"))
    print(f"Saved. Download {os.path.basename(zip_base)}.zip from the Output panel.")
    for root, _, files in os.walk(DEPLOY):
        if "demo_images" in root: continue
        for fn in files: print(" ", os.path.join(root, fn))
    ''')


def build(mode, filename, exp):
    cells.clear()
    github_setup() if mode == "github" else standalone_setup()
    pipeline(exp)
    nb = nbf.v4.new_notebook()
    nb["cells"] = list(cells)
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    out = ROOT / "notebooks" / filename
    nbf.write(nb, out)
    print("wrote", out)


# ======================================================================
# COMPRESSION STUDY notebooks (one per architecture)
# ======================================================================
STUDY_INTRO = """
# EdAi Project — Compression study: **{arch}**

Which lightweight method keeps a malaria screener **clinically safe** on a Raspberry Pi?

| Variant | What it does | Expected effect on a Pi CPU |
|---|---|---|
| FP32 | baseline | reference |
| INT8 (minmax / percentile) | 8-bit weights + activations (post-training) | ~4x smaller, often faster on ARM |
| Unstructured pruning 50% | zero the smallest 50% of weights + fine-tune | smaller after gzip, **not faster** |
| Structured pruning 30% / 50% | remove whole channels + fine-tune | fewer FLOPs → **faster** |
| Structured pruning + INT8 | both combined | smallest + fastest (if accuracy holds) |

Every variant is calibrated on validation (own temperature + LED threshold) and tested on 30 unseen patients.
Model metrics **and** system metrics are recorded; all ONNX files are saved for the Pi benchmark.

**Run with:** Add Input → *Malaria Cell Images Dataset* · GPU · Internet **On** · then **Save Version → Save & Run All**.
"""

STUDY_SETTINGS = """
EXP = "study_{arch}"
C.ARCH = "{arch}"
C.PRETRAINED = True
C.EPOCHS = {epochs}           # baseline training
C.LR = 1e-3
FT_EPOCHS = 5                 # fine-tuning after pruning
UNSTRUCTURED_AMOUNT = 0.5     # fraction of weights set to zero
STRUCTURED_RATIOS = [0.3, 0.5]  # fraction of channels removed

OUT = os.path.join(WORK, "outputs", EXP)
DEPLOY = os.path.join(WORK, "outputs", "deploy_" + EXP)
os.makedirs(OUT, exist_ok=True); os.makedirs(DEPLOY, exist_ok=True)
print(f"{{EXP}}: baseline {{C.EPOCHS}} epochs, fine-tune {{FT_EPOCHS}} epochs after pruning")
"""


def study_cells(arch, epochs):
    md("## 1. Imports, extra packages, system information")
    code(r"""
    try:
        import torch_pruning
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch-pruning"], check=True)

    sys.path.insert(0, os.path.join(WORK, "src"))
    import time
    import numpy as np, pandas as pd, torch
    import matplotlib.pyplot as plt

    import config as C
    from dataset import build_loaders, load_split, build_image_cache, cache_to_input
    from models import EdgeClassifier, count_parameters
    from sysmon import system_info
    from train import set_seed, fit
    from export_onnx import export_onnx, check_parity
    from quantize import quantize_int8
    from compress import (unstructured_prune, make_pruning_permanent, structured_prune,
                          count_flops_params, sparsity)
    from study import evaluate_variant, recommend

    SYS_TRAIN = system_info()
    for k, v in SYS_TRAIN.items():
        print(f"{k:20s}: {v}")
    """)

    md("## 2. Study settings")
    code(STUDY_SETTINGS.format(arch=arch, epochs=epochs))

    md("## 3. Data: patient-level split + images cached in RAM\nDecoding PNGs was the training bottleneck (GPU ~15% busy). Each image is decoded and resized once; pixels are identical to the Pi's preprocessing.")
    code(r"""
    !python src/data_prep.py --data_root "{DATA_ROOT}"
    set_seed()
    SPLIT_CSV = os.path.join(WORK, "splits", "patient_split.csv")
    parts = load_split(SPLIT_CSV)
    t0 = time.time()
    caches = {s: build_image_cache(df, DATA_ROOT) for s, df in parts.items()}
    print(f"Cached {sum(len(c) for c in caches.values())} images in {time.time() - t0:.0f}s "
          f"({sum(c.nbytes for c in caches.values()) / 1e9:.2f} GB)")
    loaders, _ = build_loaders(SPLIT_CSV, DATA_ROOT, caches=caches)
    Xv, yv = cache_to_input(caches["val"]), parts["val"].label.values
    Xt, yt = cache_to_input(caches["test"]), parts["test"].label.values
    Xcal = cache_to_input(caches["train"][np.random.default_rng(C.SEED).choice(len(caches["train"]), C.CALIB_IMAGES, replace=False)])
    """)

    md("## 4. Baseline FP32 model\nRe-uses `baseline/best_model.pth` if it already exists (e.g. after a kernel restart).")
    code(r"""
    base_dir = os.path.join(OUT, "baseline")
    model = EdgeClassifier(C.ARCH, pretrained=C.PRETRAINED)
    ckpt = os.path.join(base_dir, "best_model.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        hist = pd.read_csv(os.path.join(base_dir, "history.csv"))
        print("Loaded existing baseline")
    else:
        model, hist = fit(model, loaders, base_dir, epochs=C.EPOCHS, lr=C.LR)
    BASE_TRAIN_MIN = hist.epoch_time_s.sum() / 60
    print(f"Baseline training time: {BASE_TRAIN_MIN:.1f} min, best val AUROC {hist.val_auroc.max():.4f}")
    hist[["epoch", "train_loss", "val_loss", "val_auroc", "epoch_time_s", "train_throughput_img_s", "gpu_util_mean"]].round(4)
    """)

    md("## 5. Build and evaluate every variant")
    code(r"""
    rows, models_cfg, train_minutes = [], {}, {"FP32": BASE_TRAIN_MIN}
    led_ref = None

    def add(name, onnx_path, torch_model=None, method=""):
        global led_ref
        extra = {"method": method}
        if torch_model is not None:
            mflops, params = count_flops_params(torch_model)
            extra.update(params_m=params / 1e6, mflops=mflops, sparsity=sparsity(torch_model))
        extra["extra_train_min"] = train_minutes.get(name, 0.0)
        row, led = evaluate_variant(name, onnx_path, Xv, yv, Xt, yt, led_reference=led_ref, extra=extra)
        if name == "FP32":
            led_ref = led
        rows.append(row)
        models_cfg[name] = {"file": os.path.basename(onnx_path), "temperature": row["temperature"],
                            "led_threshold": row["led_threshold"]}
        print(f"{name:32s} AUROC {row['auroc']:.4f} sens {row['sensitivity']:.4f} "
              f"FN {row['fn']:4d} FP {row['fp']:4d} | {row['onnx_mb']:.2f} MB | {row['cpu_latency_ms_1thread']:.2f} ms")

    def export(m, name):
        path = os.path.join(DEPLOY, name + ".onnx")
        export_onnx(m, path)
        assert check_parity(m, path) < 1e-3
        return path

    def int8(src_path, name, method="percentile"):
        path = os.path.join(DEPLOY, name + ".onnx")
        quantize_int8(src_path, path, Xcal, method=method)
        return path

    # --- FP32 baseline + post-training quantization
    fp32_path = export(model, "fp32")
    add("FP32", fp32_path, model, "baseline")
    add("INT8 minmax", int8(fp32_path, "int8_minmax", "minmax"), method="quantization")
    add("INT8 percentile", int8(fp32_path, "int8_percentile"), method="quantization")

    # --- Unstructured pruning (+ fine-tune), then + INT8
    t0 = time.time()
    m_u = unstructured_prune(model, UNSTRUCTURED_AMOUNT)
    m_u, _ = fit(m_u, loaders, os.path.join(OUT, "unstructured"), epochs=FT_EPOCHS, lr=C.LR / 10)
    make_pruning_permanent(m_u)
    name = f"Unstructured {int(UNSTRUCTURED_AMOUNT * 100)}%"
    train_minutes[name] = (time.time() - t0) / 60
    p_u = export(m_u, "unstructured")
    add(name, p_u, m_u, "pruning")
    add(name + " + INT8", int8(p_u, "unstructured_int8"), method="pruning + quantization")

    # --- Structured channel pruning (+ fine-tune), then + INT8
    for r in STRUCTURED_RATIOS:
        t0 = time.time()
        m_s = structured_prune(model, r)
        m_s, _ = fit(m_s, loaders, os.path.join(OUT, f"structured_{int(r * 100)}"), epochs=FT_EPOCHS, lr=C.LR / 2)
        name = f"Structured {int(r * 100)}%"
        train_minutes[name] = (time.time() - t0) / 60
        p_s = export(m_s, f"structured_{int(r * 100)}")
        add(name, p_s, m_s, "pruning")
        add(name + " + INT8", int8(p_s, f"structured_{int(r * 100)}_int8"), method="pruning + quantization")

    results = pd.DataFrame(rows).set_index("variant")
    results.to_csv(os.path.join(OUT, "study_results.csv"))
    cols = ["method", "params_m", "mflops", "sparsity", "onnx_mb", "gzip_mb", "auroc", "accuracy",
            "sensitivity", "specificity", "ece", "fn", "fp", "led_flips_vs_fp32", "cpu_latency_ms_1thread", "extra_train_min"]
    results[cols].round(4)
    """)

    md("## 6. Trade-offs and deployment recommendation\nRule: among variants as safe as FP32 (sensitivity within 1 point, AUROC within 0.005), choose the fastest. **Latency here is Kaggle's x86 CPU — confirm on the Pi with `pi/benchmark.py`.**")
    code(r"""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for v, r in results.iterrows():
        mk = "s" if "INT8" in v else "o"
        ax[0].scatter(r.cpu_latency_ms_1thread, r.sensitivity, marker=mk, s=60); ax[0].annotate(v, (r.cpu_latency_ms_1thread, r.sensitivity), fontsize=8)
        ax[1].scatter(r.onnx_mb, r.auroc, marker=mk, s=60); ax[1].annotate(v, (r.onnx_mb, r.auroc), fontsize=8)
    ax[0].axhline(results.loc["FP32", "sensitivity"] - 0.01, ls="--", c="gray")
    ax[0].set_xlabel("CPU latency (ms, 1 thread)"); ax[0].set_ylabel("test sensitivity"); ax[0].set_title("Safety vs speed")
    ax[1].set_xlabel("ONNX size (MB)"); ax[1].set_ylabel("test AUROC"); ax[1].set_title("Quality vs size")
    results["fn"].plot.barh(ax=ax[2], color="tab:red"); ax[2].set_title("Missed infections (FN) on test"); ax[2].invert_yaxis()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "study_tradeoffs.png"), dpi=120); plt.show()

    safe, best = recommend(results)
    print("Clinically safe variants (fastest first):")
    print(safe[["onnx_mb", "auroc", "sensitivity", "fn", "cpu_latency_ms_1thread"]].round(4))
    print(f"\nRecommended for the Pi (provisional): {best}")
    """)

    md("## 7. Save everything for the laptop / Pi benchmark")
    code(r"""
    deploy_config = {
        "experiment": EXP,
        "arch": C.ARCH,
        "classes": C.CLASSES,
        "positive_class_index": 1,
        "img_size": C.IMG_SIZE,
        "input": "RGB, resized to img_size x img_size (bilinear), float32 in [0,1], NCHW; normalization is inside the model",
        "led_rule": "LED ON if softmax(logits / temperature)[1] >= led_threshold (per model)",
        "target_sensitivity": C.TARGET_SENSITIVITY,
        "recommended": best,
        "models": models_cfg,
        # backwards-compatible keys (used by older scripts)
        "model_fp32": models_cfg["FP32"]["file"], "temperature_fp32": models_cfg["FP32"]["temperature"],
        "model_int8": models_cfg["INT8 percentile"]["file"], "led_threshold": models_cfg["FP32"]["led_threshold"],
    }
    with open(os.path.join(DEPLOY, "deploy_config.json"), "w") as f:
        json.dump(deploy_config, f, indent=2)

    demo_dir = os.path.join(DEPLOY, "demo_images"); os.makedirs(demo_dir, exist_ok=True)
    demo = pd.concat([parts["test"][parts["test"].label == l].sample(10, random_state=1) for l in (0, 1)])
    for p in demo.path:
        shutil.copy(os.path.join(DATA_ROOT, p), os.path.join(demo_dir, os.path.basename(p)))
    demo.assign(file=demo.path.map(os.path.basename))[["file", "class", "label", "patient_id"]].to_csv(
        os.path.join(demo_dir, "labels.csv"), index=False)

    with open(os.path.join(OUT, "study_summary.json"), "w") as f:
        json.dump({"experiment": EXP, "code_version": GIT_COMMIT, "system_training": SYS_TRAIN,
                   "recommended": best, "results": results.reset_index().to_dict("records")}, f, indent=2, default=float)

    zip_base = os.path.join("/kaggle/working" if ON_KAGGLE else WORK, "edai_" + EXP)
    shutil.make_archive(zip_base, "zip", os.path.join(WORK, "outputs"))
    print(f"Saved. Download {os.path.basename(zip_base)}.zip from the Output panel.")
    """)


def build_study(arch, filename, epochs=15):
    cells.clear()
    github_setup(STUDY_INTRO.format(arch=arch))
    study_cells(arch, epochs)
    nb = nbf.v4.new_notebook()
    nb["cells"] = list(cells)
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    out = ROOT / "notebooks" / filename
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    build("github", "edai_malaria_kaggle.ipynb", PRETRAINED_EXP)
    build("github", "edai_malaria_kaggle_scratch.ipynb", SCRATCH_EXP)
    build("standalone", "edai_malaria_kaggle_standalone.ipynb", PRETRAINED_EXP)
    for arch in ("mobilenet_v3_small", "mobilenet_v2", "resnet18"):
        build_study(arch, f"study_{arch}.ipynb")
