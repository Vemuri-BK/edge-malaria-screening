"""
STEP 2 - DATASET AND PREPROCESSING
==================================
Loads cell images listed in a split CSV (from Step 1) and turns them into
tensors for training.

Preprocessing (identical on Kaggle, laptop and Raspberry Pi):
    1. Read image as RGB
    2. Resize to IMG_SIZE x IMG_SIZE (bilinear)
    3. Scale pixels to [0, 1]
    4. Channel order C x H x W
ImageNet mean/std normalization is done INSIDE the model (models.py), so the
Pi only has to do steps 1-4 with plain numpy - no torch needed on the Pi.

Augmentation (training only): flips, rotation, colour jitter. Cells have no
fixed orientation, and stain colour varies between slides and microscopes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from config import BATCH_SIZE, IMG_SIZE, NUM_WORKERS


# ----------------------------------------------------------------------
# Preprocessing shared with deployment
# ----------------------------------------------------------------------
def preprocess_pil(img: Image.Image, img_size: int = IMG_SIZE) -> np.ndarray:
    """PIL image -> float32 array (3, H, W) in [0, 1]. Same code runs on the Pi."""
    img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)


def build_transforms(train: bool, img_size: int = IMG_SIZE):
    """torchvision pipeline. The eval pipeline matches preprocess_pil exactly."""
    if train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            T.ToTensor(),
        ])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class MalariaDataset(Dataset):
    """
    Returns (image_tensor, label). label: 0 = Uninfected, 1 = Parasitized.
    If `cache` (uint8 array N x H x W x 3 of already-resized images) is given,
    images come from RAM instead of disk - identical pixels, much faster.
    """

    def __init__(self, df: pd.DataFrame, data_root, transform, cache=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.root = Path(data_root)
        self.transform = transform
        self.cache = cache

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self.cache is not None:
            img = Image.fromarray(self.cache[i])          # already IMG_SIZE x IMG_SIZE
            x = self.transform(img)
        else:
            with Image.open(self.root / self.paths[i]) as img:
                x = self.transform(img.convert("RGB"))
        return x, torch.tensor(self.labels[i], dtype=torch.long)


# ----------------------------------------------------------------------
# Optional RAM cache (speeds up training ~several x: PNG decoding is the bottleneck)
# ----------------------------------------------------------------------
def _load_resized(path, img_size):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8)


def build_image_cache(df: pd.DataFrame, data_root, img_size: int = IMG_SIZE, workers: int = 8) -> np.ndarray:
    """Decode + resize every image once. Returns uint8 array (N, img_size, img_size, 3)."""
    from concurrent.futures import ThreadPoolExecutor
    root = Path(data_root)
    with ThreadPoolExecutor(workers) as ex:
        arrs = list(ex.map(lambda p: _load_resized(root / p, img_size), df["path"]))
    return np.stack(arrs)


def cache_to_input(cache: np.ndarray) -> np.ndarray:
    """uint8 (N, H, W, 3) -> float32 (N, 3, H, W) in [0, 1] (same as preprocess_pil)."""
    return cache.astype(np.float32).transpose(0, 3, 1, 2) / 255.0


def load_split(split_csv) -> dict:
    """Read a split CSV and return {'train': df, 'val': df, 'test': df}."""
    df = pd.read_csv(split_csv)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}


def build_loaders(split_csv, data_root, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, caches=None):
    """
    DataLoaders for train (shuffled, augmented) and val/test (plain).
    caches: optional {'train': array, 'val': array, 'test': array} from build_image_cache.
    """
    parts = load_split(split_csv)
    pin = torch.cuda.is_available()
    loaders = {}
    for name, df in parts.items():
        train = name == "train"
        cache = caches.get(name) if caches else None
        ds = MalariaDataset(df, data_root, build_transforms(train), cache=cache)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=train,
                                   num_workers=num_workers, pin_memory=pin,
                                   drop_last=False)
    return loaders, parts
