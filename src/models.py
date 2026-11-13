"""
STEP 3 - MODEL
==============
A small ImageNet-pretrained CNN with a 2-class head.

EdgeClassifier wraps the backbone and adds ImageNet normalization as the
first operation, so the exported model takes plain [0, 1] RGB input.
Output: 2 logits [Uninfected, Parasitized].

Supported backbones (for the later architecture comparison):
    mobilenet_v3_small  (~1.5 M params, default)
    mobilenet_v2        (~2.2 M params)
    resnet18            (~11 M params)
"""
import io

import torch
import torch.nn as nn
import torchvision.models as tvm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_backbone(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a torchvision backbone and replace its last layer with a 2-class layer."""
    factories = {
        "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
        "mobilenet_v2": (tvm.mobilenet_v2, tvm.MobileNet_V2_Weights.IMAGENET1K_V1),
        "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    }
    if arch not in factories:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {list(factories)}")
    fn, weights = factories[arch]

    try:
        model = fn(weights=weights if pretrained else None)
        loaded = pretrained
    except Exception as e:          # e.g. no internet on Kaggle
        print(f"[warn] could not load pretrained weights ({e}); training from scratch")
        model = fn(weights=None)
        loaded = False

    if not loaded:
        # torchvision MobileNets use BatchNorm momentum 0.01, which makes the
        # running statistics converge very slowly when training from scratch
        # (eval predictions stay near-constant). 0.1 fixes that.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1

    if arch.startswith("mobilenet"):
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    else:  # resnet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class EdgeClassifier(nn.Module):
    """Backbone + built-in normalization. Input: (N, 3, H, W) in [0, 1]."""

    def __init__(self, arch: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()
        self.arch = arch
        self.backbone = build_backbone(arch, 2, pretrained)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


# ----------------------------------------------------------------------
# Model statistics (reported in the results table)
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Size of the saved FP32 weights in MB."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6
