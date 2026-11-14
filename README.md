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
| 10 | `src/compress.py` | Unstructured + structured (channel) pruning, FLOPs/params, sparsity | Kaggle |
| 11 | `src/study.py` | Evaluate every lightweight variant identically; deployment rule | Kaggle |
| 12 | `pi/` | Benchmark on laptop/Pi; single LED (ON = Parasitized), system monitoring | Raspberry Pi |
| 13 | report | Tables, plots, demo | — |

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

## Experiments (one Kaggle notebook each)
| Notebook | Experiment | Init | Epochs | Outputs |
|---|---|---|---|---|
| `notebooks/edai_malaria_kaggle.ipynb` | `pretrained` | ImageNet | 15 | `edai_outputs_pretrained.zip` |
| `notebooks/edai_malaria_kaggle_scratch.ipynb` | `scratch` | random | 100 | `edai_outputs_scratch.zip` |

| `notebooks/study_mobilenet_v3_small.ipynb` | `study_mobilenet_v3_small` | ImageNet | 15 + fine-tunes | `edai_study_mobilenet_v3_small.zip` |
| `notebooks/study_mobilenet_v2.ipynb` | `study_mobilenet_v2` | ImageNet | 15 + fine-tunes | `edai_study_mobilenet_v2.zip` |
| `notebooks/study_resnet18.ipynb` | `study_resnet18` | ImageNet | 15 + fine-tunes | `edai_study_resnet18.zip` |

**Compression study** (one notebook per architecture): FP32, INT8 (minmax, percentile),
unstructured pruning 50%, structured pruning 30% / 50%, and pruning + INT8 combined.
Each variant is calibrated on validation and tested on unseen patients; all ONNX files are
saved with a `deploy_config.json` so `pi/benchmark.py` can test every variant on the Pi.

All notebooks pull the code from this repo. `edai_malaria_kaggle_standalone.ipynb` is a backup of the
pretrained run with all code embedded (no GitHub needed).
After editing `src/`, regenerate the notebooks: `python notebooks/build_notebook.py`.

## Running on Kaggle
1. Kaggle → File → Import Notebook → one of the notebooks above
2. Add Input → *Malaria Cell Images Dataset* (iarunava)
3. Settings → Accelerator **GPU**, Internet **On**
4. Run All → download the zip from Output, extract it into `outputs/` on the laptop

Workflow: edit `src/` on the laptop → `git push` → re-run the notebook on Kaggle.
Private repo: add a Kaggle secret `GITHUB_TOKEN` (GitHub personal access token, read access).

## Benchmark on the laptop and the Raspberry Pi
Same script on every device (needs only `pip install -r pi/requirements.txt`):
```
python pi/benchmark.py --deploy outputs/deploy_pretrained outputs/deploy_scratch \
                       --data_root data/cell_images --split_csv splits/patient_split.csv
```
Without `--data_root/--split_csv` it uses the 20 demo images in each deploy folder.
Writes `results/benchmark_<device>.csv`: accuracy, sensitivity, specificity, AUROC, ECE,
model size, load time, RAM, latency (mean/p50/p95), FPS, end-to-end time, CPU %, CPU temperature.
