# EdAi Project — Reliable Malaria Screening on a Raspberry Pi

A low-cost edge device that classifies blood-smear cell images as
**Parasitized** or **Uninfected** on a Raspberry Pi, signalling the result
with LEDs (green = negative, red = positive, yellow = unsure).

## Research question
> When a medical model is compressed for an edge device, does it stay
> clinically reliable, or does it only *look* reliable?

1. Does INT8 compression hurt **sensitivity** (missed infections) more than accuracy?
2. Does compression damage the model's **confidence**, i.e. the yellow "unsure" light?
3. Can **temperature scaling** after quantization restore it at zero speed cost?
4. Side finding: how much does a **random split** (patient leakage) inflate results
   compared with a **patient-level split**?

## Plan
| Step | Module | What it does | Runs on |
|---|---|---|---|
| 1 | `src/data_prep.py` | Patient IDs, de-duplication, patient-level + random splits | Kaggle / laptop |
| 2 | `src/dataset.py` | PyTorch Dataset, preprocessing, augmentation | Kaggle |
| 3 | `src/models.py` | MobileNetV3-Small (+ MobileNetV2, ResNet18 for comparison) | Kaggle |
| 4 | `src/train.py` | Training loop, checkpointing | Kaggle |
| 5 | `src/metrics.py` | Accuracy, sensitivity, specificity, AUROC, ECE, temperature, LED threshold | Kaggle |
| — | `src/sysmon.py` | System metrics: CPU/GPU use, RAM, temperature, throughput | Kaggle + Pi |
| 6 | `src/calibrate.py` | Temperature scaling + yellow-light threshold | Kaggle |
| 7 | `src/export_onnx.py` | Export to ONNX, check PyTorch = ONNX | Kaggle |
| 8 | `src/quantize.py` | INT8 static quantization (ONNX Runtime) | Kaggle |
| 9 | `src/evaluate.py` | ONNX evaluation (same as Pi) + latency benchmark | Kaggle |
| 10 | `pi/` | Pi inference, single LED (ON = Parasitized), system monitoring | Raspberry Pi |
| 11 | report | Tables, plots, demo | — |

## Dataset
NIH Malaria Cell Images (27,558 cells, 200 patients: 150 infected, 50 uninfected).
Kaggle: `iarunava/cell-images-for-detecting-malaria`.
The patient ID is the `C<number>` prefix of each filename.

## Project layout
```
EdAi_project/
├── src/          # all Python modules (one per step)
├── splits/       # split CSVs (committed, so everyone uses the same split)
├── notebooks/    # thin Kaggle notebook that calls src/
├── pi/           # Raspberry Pi code (later)
├── outputs/      # models, logs, plots (not committed)
└── data/         # local dataset copy (not committed)
```

## Running on Kaggle
**Run `notebooks/edai_malaria_kaggle.ipynb`.** It clones this repo, so Kaggle always uses the latest pushed code.

1. Kaggle → File → Import Notebook → `notebooks/edai_malaria_kaggle.ipynb`
2. Add Input → *Malaria Cell Images Dataset* (iarunava)
3. Settings → Accelerator **GPU**, Internet **On**
4. Run All → download `edai_outputs.zip` from Output

Workflow: edit `src/` on the laptop → `git push` → re-run the notebook on Kaggle.
Private repo: add a Kaggle secret `GITHUB_TOKEN` (GitHub personal access token, read access).

Backup: `notebooks/edai_malaria_kaggle_standalone.ipynb` has all code embedded and needs no GitHub.
After editing `src/`, regenerate both notebooks with `python notebooks/build_notebook.py`.
