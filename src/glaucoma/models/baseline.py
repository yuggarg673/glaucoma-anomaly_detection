"""Supervised baseline: ImageNet-pretrained ResNet18 or EfficientNet-B0, 2 output classes."""
from __future__ import annotations

from torch import nn


def build_baseline(backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    import torchvision.models as tvm

    try:
        if backbone == "resnet18":
            w = tvm.ResNet18_Weights.DEFAULT if pretrained else None
            m = tvm.resnet18(weights=w)
            m.fc = nn.Linear(m.fc.in_features, num_classes)
        elif backbone == "efficientnet_b0":
            w = tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            m = tvm.efficientnet_b0(weights=w)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Use resnet18 or efficientnet_b0.")
    except ValueError:
        raise
    except Exception as exc:  # most often: no internet to download pretrained weights
        raise RuntimeError(
            f"Could not build {backbone} (pretrained={pretrained}): {exc}\n"
            "If the ImageNet weights cannot be downloaded, set baseline.pretrained: false in configs/config.yaml."
        ) from exc
    return m
