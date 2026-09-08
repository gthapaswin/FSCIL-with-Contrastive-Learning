"""
CIFAR-adapted ResNet18 backbone f_theta(.).

Standard torchvision resnet18 assumes 224x224 ImageNet input (7x7 stride-2
stem + maxpool), which over-downsamples 32x32 CIFAR images to nothing. This
file swaps the stem for a 3x3 stride-1 conv and drops the maxpool, which is
the standard adaptation used across CIFAR-ResNet literature.

Usage:
    backbone = CifarResNet18(out_dim=512)
    feats = backbone(images)   # (B, 512)

Per the spec, the backbone is:
  - trained with a plain softmax classifier head during Phase A (base-class
    pretraining), then
  - frozen (requires_grad=False, .eval()) for the rest of Session 0 (Phase B)
    and for all incremental sessions.
"""

import torch
import torch.nn as nn
import torchvision


class CifarResNet18(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        # CIFAR stem adaptation
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(net.children())[:-1])  # drop fc
        self.out_dim = out_dim  # resnet18 already outputs 512 after avgpool

    def forward(self, x):
        feat = self.features(x)          # (B, 512, 1, 1)
        return feat.flatten(1)             # (B, 512)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


class StandardResNet18(nn.Module):
    """Standard ImageNet-style ResNet-18 (7x7 stride-2 stem + maxpool), used
    for the larger-resolution benchmarks: miniImageNet (84x84) and CUB-200
    (224x224). `pretrained=True` loads ImageNet weights -- the FSCIL
    convention for CUB, whose 100 base classes are too few to train a strong
    backbone from scratch."""

    def __init__(self, out_dim=512, pretrained=False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])  # drop fc
        self.out_dim = out_dim  # 512 after avgpool

    def forward(self, x):
        feat = self.features(x)          # (B, 512, 1, 1)
        return feat.flatten(1)             # (B, 512)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


def build_backbone(backbone_type, out_dim=512):
    """Factory: pick the backbone stem for the active dataset."""
    if backbone_type == "cifar_resnet18":
        return CifarResNet18(out_dim=out_dim)
    if backbone_type == "resnet18":
        return StandardResNet18(out_dim=out_dim, pretrained=False)
    if backbone_type == "resnet18_pretrained":
        return StandardResNet18(out_dim=out_dim, pretrained=True)
    raise ValueError(f"Unknown backbone_type '{backbone_type}'")


class BackboneWithHead(nn.Module):
    """Used only during Phase A pretraining — backbone + a temporary linear
    classifier over the base classes. The classifier head is discarded once
    the backbone is frozen for Phase B."""

    def __init__(self, num_base_classes, out_dim=512, backbone_type="cifar_resnet18", ssl_dim=None):
        super().__init__()
        self.backbone = build_backbone(backbone_type, out_dim=out_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(out_dim, num_base_classes)
        # optional self-supervised projection head (CLOSER Phase-A mode)
        self.ssl_head = None
        if ssl_dim:
            self.ssl_head = nn.Sequential(nn.Linear(out_dim, out_dim), nn.ReLU(inplace=True),
                                          nn.Linear(out_dim, ssl_dim))

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(self.dropout(feat))
        return logits, feat

    def project_ssl(self, feat):
        return torch.nn.functional.normalize(self.ssl_head(feat), dim=-1)
