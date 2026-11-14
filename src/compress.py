"""
STEP 10 - LIGHTWEIGHT METHODS (pruning; quantization lives in quantize.py)
=========================================================================
Two kinds of pruning, because they behave very differently on an edge CPU:

1. UNSTRUCTURED magnitude pruning  (torch.nn.utils.prune)
   Sets the smallest X% of individual weights to zero. The network shape is
   unchanged, so a normal CPU runtime does the SAME amount of work:
       -> no speed-up, smaller only after compression (zip/gzip)
   Included to show that "sparsity" does not mean "faster" on a Pi.

2. STRUCTURED channel pruning  (torch-pruning library)
   Removes whole channels/filters and rebuilds smaller layers, keeping all
   layer connections consistent (residual adds, SE blocks, ...).
       -> fewer parameters AND fewer FLOPs -> real speed-up on any CPU

Both are followed by a short fine-tune to recover accuracy.
Either can then be combined with INT8 quantization (quantize.py).
"""
import copy
import gzip
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from config import IMG_SIZE


def _last_layer(model):
    bb = model.backbone
    return bb.classifier[-1] if hasattr(bb, "classifier") else bb.fc


def count_flops_params(model, img_size: int = IMG_SIZE):
    """(MFLOPs/MACs, params) for one image, via torch-pruning's counter."""
    import torch_pruning as tp
    model = model.cpu().eval()
    ops, params = tp.utils.count_ops_and_params(model, torch.rand(1, 3, img_size, img_size))
    return ops / 1e6, params


def sparsity(model) -> float:
    """Fraction of conv/linear weights that are exactly zero."""
    zeros = total = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            zeros += int((m.weight == 0).sum())
            total += m.weight.numel()
    return zeros / max(total, 1)


def unstructured_prune(model, amount: float = 0.5):
    """Global magnitude pruning of conv + linear weights (classifier excluded). Returns a new model."""
    model = copy.deepcopy(model).cpu()
    last = _last_layer(model)
    params = [(m, "weight") for m in model.modules()
              if isinstance(m, (nn.Conv2d, nn.Linear)) and m is not last]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def make_pruning_permanent(model):
    """Remove pruning masks/hooks so the zeros are baked into the weights (needed before export)."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and prune.is_pruned(m):
            prune.remove(m, "weight")
    return model


def structured_prune(model, ratio: float = 0.3, img_size: int = IMG_SIZE):
    """
    Channel pruning with torch-pruning (L2-magnitude importance).
    ratio = fraction of channels removed in every prunable layer
    (params and FLOPs shrink roughly by (1 - ratio)^2).
    Returns a new, physically smaller model.
    """
    import torch_pruning as tp
    model = copy.deepcopy(model).cpu().eval()
    example = torch.rand(1, 3, img_size, img_size)
    pruner = tp.pruner.MetaPruner(
        model, example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[_last_layer(model)],      # keep the 2 output classes
    )
    pruner.step()
    return model


def gzip_size_mb(path) -> float:
    """File size after gzip - shows the storage benefit of zeros (unstructured pruning)."""
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), compresslevel=6)) / 1e6


def file_size_mb(path) -> float:
    return os.path.getsize(path) / 1e6
