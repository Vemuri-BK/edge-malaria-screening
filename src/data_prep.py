"""
STEP 1 - DATA PREPARATION
=========================
Builds the image list for the NIH malaria cell dataset and creates
train/val/test splits.

What this module does
---------------------
1. Finds every cell image under the dataset folder.
2. Removes duplicates. (The Kaggle copy contains the whole dataset twice:
   cell_images/ and cell_images/cell_images/.)
3. Extracts the PATIENT ID from each filename.
   Example: "C33P1thinF_IMG_..._cell_179.png" -> patient "C33"
   The dataset has 200 patients (150 infected, 50 uninfected).
4. Makes TWO splits so we can compare them in the report:
   - patient_split.csv : no patient appears in more than one split (honest)
   - random_split.csv  : images shuffled randomly (what most tutorials do;
                          cells of one patient leak into train AND test)
5. Prints statistics and a leakage check.

Output: splits/manifest.csv, splits/patient_split.csv, splits/random_split.csv
Paths in the CSVs are RELATIVE to the data root, so the same CSV works on
Kaggle, on a laptop and on the Raspberry Pi.

Usage
-----
Kaggle : python src/data_prep.py --data_root /kaggle/input/cell-images-for-detecting-malaria/cell_images
Local  : python src/data_prep.py --data_root data/cell_images
Add --verify to open every image and drop corrupted files (slower).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLASSES, SEED, SPLIT_RATIOS, SPLITS_DIR

PATIENT_RE = re.compile(r"^C(\d+)")   # every filename starts with C<number>


# ----------------------------------------------------------------------
# 1. Collect images
# ----------------------------------------------------------------------
def collect_images(data_root: Path, verify: bool = False) -> pd.DataFrame:
    """Return one row per unique image: path, class name, label, patient_id."""
    rows, seen = [], set()

    for label, cls in enumerate(CLASSES):
        # All folders named like the class (there may be a duplicated nested copy).
        # Sort by depth so the shallowest copy wins and duplicates are skipped.
        class_dirs = sorted((d for d in data_root.rglob(cls) if d.is_dir()),
                            key=lambda d: len(d.parts))
        if not class_dirs:
            raise FileNotFoundError(f"No '{cls}' folder found under {data_root}")

        for d in class_dirs:
            for img in sorted(d.glob("*.png")):
                key = (cls, img.name)
                if key in seen:                 # duplicate copy -> skip
                    continue
                seen.add(key)

                m = PATIENT_RE.match(img.name)
                if m is None:
                    print(f"[warn] cannot read patient id from {img.name}, skipped")
                    continue

                rows.append({
                    "path": img.relative_to(data_root).as_posix(),
                    "class": cls,
                    "label": label,
                    "patient_id": f"C{m.group(1)}",
                })

    df = pd.DataFrame(rows)

    if verify:
        df = drop_corrupted(df, data_root)
    return df


def drop_corrupted(df: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    """Open every image once; drop files that cannot be decoded."""
    from PIL import Image
    bad = []
    for i, p in enumerate(df["path"]):
        try:
            with Image.open(data_root / p) as im:
                im.verify()
        except Exception:
            bad.append(i)
    if bad:
        print(f"[info] dropped {len(bad)} corrupted images")
    return df.drop(index=bad).reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. Splits
# ----------------------------------------------------------------------
def _cut(n: int, ratios) -> tuple:
    """Number of items for train and val (test gets the rest)."""
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return n_train, n_val


def patient_level_split(df: pd.DataFrame, ratios=SPLIT_RATIOS, seed=SEED) -> pd.DataFrame:
    """
    Split by PATIENT, not by image.
    Patients are stratified into two groups (infected / uninfected patients)
    so every split gets both kinds.
    """
    rng = np.random.default_rng(seed)
    infected = df.groupby("patient_id")["label"].max()   # 1 if patient has any parasitized cell

    assignment = {}
    for _, group in infected.groupby(infected):
        ids = np.array(sorted(group.index))
        rng.shuffle(ids)
        n_tr, n_va = _cut(len(ids), ratios)
        for pid in ids[:n_tr]:
            assignment[pid] = "train"
        for pid in ids[n_tr:n_tr + n_va]:
            assignment[pid] = "val"
        for pid in ids[n_tr + n_va:]:
            assignment[pid] = "test"

    out = df.copy()
    out["split"] = out["patient_id"].map(assignment)
    return out


def random_image_split(df: pd.DataFrame, ratios=SPLIT_RATIOS, seed=SEED) -> pd.DataFrame:
    """Common tutorial split: shuffle images (stratified by class), ignore patients."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""
    for _, idx in out.groupby("label").groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_tr, n_va = _cut(len(idx), ratios)
        out.loc[idx[:n_tr], "split"] = "train"
        out.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
        out.loc[idx[n_tr + n_va:], "split"] = "test"
    return out


# ----------------------------------------------------------------------
# 3. Reporting
# ----------------------------------------------------------------------
def summarize(df: pd.DataFrame, name: str) -> None:
    print(f"\n=== {name} ===")
    table = pd.crosstab(df["split"], df["class"], margins=True)
    table["patients"] = df.groupby("split")["patient_id"].nunique()
    table.loc["All", "patients"] = df["patient_id"].nunique()
    print(table.reindex(["train", "val", "test", "All"]))

    # Leakage check: patients appearing in more than one split
    train_p = set(df.loc[df.split == "train", "patient_id"])
    test_p = set(df.loc[df.split == "test", "patient_id"])
    leaked = train_p & test_p
    print(f"Patients in BOTH train and test: {len(leaked)} of {len(test_p)} test patients")


def main():
    ap = argparse.ArgumentParser(description="Step 1: build manifest and splits")
    ap.add_argument("--data_root", required=True,
                    help="folder that contains Parasitized/ and Uninfected/ (searched recursively)")
    ap.add_argument("--verify", action="store_true", help="open every image and drop corrupted ones")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    df = collect_images(data_root, verify=args.verify)

    print(f"Unique images : {len(df)}")
    print(df["class"].value_counts().to_string())
    n_pat = df.groupby("patient_id")["label"].max()
    print(f"Patients      : {len(n_pat)} ({int(n_pat.sum())} infected, {int((n_pat == 0).sum())} uninfected)")

    patient_df = patient_level_split(df)
    random_df = random_image_split(df)

    summarize(patient_df, "PATIENT-LEVEL split (used for all experiments)")
    summarize(random_df, "RANDOM image split (for comparison only)")

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS_DIR / "manifest.csv", index=False)
    patient_df.to_csv(SPLITS_DIR / "patient_split.csv", index=False)
    random_df.to_csv(SPLITS_DIR / "random_split.csv", index=False)
    print(f"\nSaved CSVs to {SPLITS_DIR}")


if __name__ == "__main__":
    main()
