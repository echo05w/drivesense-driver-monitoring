"""CNN architectures for single-frame distraction classification.

Two approaches per the Brief (§7): a small from-scratch CNN, and a
transfer-learning model fine-tuned from a pretrained ImageNet backbone.
Architectures only — no trained weights are produced by this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A small from-scratch CNN baseline for distraction classification.

    Deliberately shallow: this is meant to be a fair "from scratch" point of
    comparison against transfer learning, not a competitive architecture.
    """

    def __init__(self, num_classes: int, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TransferLearningCNN(nn.Module):
    """Wraps a pretrained torchvision backbone with a new classifier head.

    Backbone is frozen by default; call `unfreeze_backbone()` for
    fine-tuning once a baseline head-only training run has been evaluated
    (two-stage transfer learning, per the Brief's "transfer learning" phase).
    """

    def __init__(self, num_classes: int, backbone_name: str = "mobilenet_v3_small") -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = self._build_backbone(backbone_name)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.freeze_backbone()

    @staticmethod
    def _build_backbone(name: str):
        import torchvision.models as tvm

        if name == "mobilenet_v3_small":
            weights = tvm.MobileNet_V3_Small_Weights.DEFAULT
            model = tvm.mobilenet_v3_small(weights=weights)
            feature_dim = model.classifier[0].in_features
            model.classifier = nn.Identity()
            return model, feature_dim
        raise ValueError(f"Unsupported backbone: {name}")

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
