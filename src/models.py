"""
STEP 3 - MODEL
==============
A small ImageNet-pretrained CNN with a 2-class head.

EdgeClassifier wraps the backbone and adds ImageNet normalization as the
first operation, so the exported model takes plain [0, 1] RGB input.
Output: 2 logits [Uninfected, Parasitized].

Supported backbones (for the later architecture comparison):
    tiny_cnn            (~0.29 M params, our own design, from scratch only)
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


class TinyCNN(nn.Module):
    """
    Our own small CNN, designed for the edge and always trained from scratch.
    4 blocks of [Conv3x3-BN-ReLU] x2 + MaxPool, channels 16-32-64-128,
    then global average pooling and a linear 2-class head (~0.29 M params).
    Only standard Conv/BN/ReLU layers: friendly to INT8 and channel pruning.
    """

    def __init__(self, num_classes: int = 2, widths=(16, 32, 64, 128)):
        super().__init__()
        layers, c_in = [], 3
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_backbone(arch: str, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Create a backbone and replace its last layer with a 2-class layer."""
    if arch == "tiny_cnn":
        return TinyCNN(num_classes)               # no pretrained weights exist: always from scratch
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
