"""
CNN architectures trained in the notebooks (cnn_comparison.ipynb).
Class definitions must match exactly what was used during training so that
state dicts load without errors.
"""

import torch.nn as nn
import torchvision.models as tv_models

CLASSES = ['adware', 'banking', 'benign', 'riskware', 'sms']
N_CLASSES = len(CLASSES)


# ── CNN-V1: Baseline ──────────────────────────────────────────────────────────

class CNN_V1(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ── CNN-V2: Deep ──────────────────────────────────────────────────────────────

class CNN_V2(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(0.25))
        self.b1 = blk(3, 64);   self.b2 = blk(64, 128)
        self.b3 = blk(128, 256); self.b4 = blk(256, 256)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))


# ── CNN-V3: Residual ──────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co))
        self.skip = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(0.2)

    def forward(self, x):
        return self.drop(self.relu(self.conv(x) + self.skip(x)))


class CNN_V3(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.b1 = nn.Sequential(ResBlock(3, 64),    nn.MaxPool2d(2))
        self.b2 = nn.Sequential(ResBlock(64, 128),  nn.MaxPool2d(2))
        self.b3 = nn.Sequential(ResBlock(128, 256), nn.MaxPool2d(2))
        self.b4 = nn.Sequential(ResBlock(256, 256), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))


# ── CNN-V4: Squeeze-and-Excitation ───────────────────────────────────────────

class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // r), nn.ReLU(),
            nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class SEConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU())
        self.se   = SEBlock(co)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.25)

    def forward(self, x):
        return self.drop(self.pool(self.se(self.conv(x))))


class CNN_V4(nn.Module):
    def __init__(self, n=N_CLASSES):
        super().__init__()
        self.b1 = SEConvBlock(3, 64)
        self.b2 = SEConvBlock(64, 128)
        self.b3 = SEConvBlock(128, 256)
        self.b4 = SEConvBlock(256, 256)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))


# ── Pretrained torchvision models ─────────────────────────────────────────────

def build_pretrained(arch: str, n: int = N_CLASSES) -> nn.Module:
    """Build a torchvision backbone with the head replaced for n classes."""
    if arch == 'resnet18':
        m = tv_models.resnet18(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n))
    elif arch == 'resnet50':
        m = tv_models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, n))
    elif arch == 'efficientnet_b0':
        m = tv_models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, n))
    elif arch == 'efficientnet_b2':
        m = tv_models.efficientnet_b2(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, n))
    elif arch == 'mobilenet_v3_small':
        m = tv_models.mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n)
    elif arch == 'convnext_tiny':
        m = tv_models.convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n)
    else:
        raise ValueError(f"Unknown pretrained arch: {arch!r}")
    return m


# ── Registry ──────────────────────────────────────────────────────────────────

CUSTOM_REGISTRY = {
    'cnn-v1': CNN_V1,
    'cnn-v2': CNN_V2,
    'cnn-v3': CNN_V3,
    'cnn-v4': CNN_V4,
}

PRETRAINED_ARCHS = {
    'resnet18', 'resnet50',
    'efficientnet_b0', 'efficientnet_b2',
    'mobilenet_v3_small', 'convnext_tiny',
}
