"""
Week 1 - baseline CNN trained from scratch (no pretrained weights).
I built this to compare against MobileNetV2 transfer learning later.

The architecture is 4 conv blocks (double conv + BN + pool + dropout)
followed by global average pooling and a small FC head.
~2.5M parameters, which is roughly the same as MobileNetV2 for fair comparison.
Input size: 224x224 (same as the transfer learning models).
"""

from __future__ import annotations
import torch
import torch.nn as nn


NUM_CLASSES = 7


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def _vgg_block(in_ch: int, out_ch: int, dropout: float = 0.25) -> nn.Sequential:
    return nn.Sequential(
        _conv_bn_relu(in_ch,  out_ch),
        _conv_bn_relu(out_ch, out_ch),
        nn.MaxPool2d(2, 2),
        nn.Dropout2d(dropout),
    )


class BaselineCNN(nn.Module):
    """Scratch-trained CNN, Week 1 baseline. Same 224x224 input as MobileNetV2."""

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_fc: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _vgg_block(3,    32),   # -> (B,  32, 112, 112)
            _vgg_block(32,   64),   # -> (B,  64,  56,  56)
            _vgg_block(64,  128),   # -> (B, 128,  28,  28)
            _vgg_block(128, 256),   # -> (B, 256,  14,  14)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def model_info(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    return {
        "total_params":     total,
        "trainable_params": trainable,
        "size_mb":          round(size_mb, 2),
    }


if __name__ == "__main__":
    m = BaselineCNN()
    x = torch.randn(2, 3, 224, 224)
    info = model_info(m)
    print(f"Output shape : {m(x).shape}")
    print(f"Parameters   : {info['total_params']:,}")
    print(f"Model size   : {info['size_mb']} MB")
