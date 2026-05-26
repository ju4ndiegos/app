"""
Model loading and inference for image-based malware classification.

Two pipelines are supported:

  CNN pipeline (any model in CUSTOM_REGISTRY or PRETRAINED_ARCHS):
    PIL Image (64×64 RGB) → ImageNet normalise → forward() → softmax → probs

  Classical pipeline (Scaler → PCA → RF or SVM):
    PIL Image (64×64 RGB) → flatten (12288) → StandardScaler → PCA(200) → predict_proba()

Usage:
    from pipeline.core.predict import load_model, predict

    model = load_model('cnn-v3')           # uses default weight path from settings
    result = predict(model, pil_image)     # {'label': 'adware', 'probs': {...}}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pipeline.core.architectures import (
    CLASSES, CUSTOM_REGISTRY, PRETRAINED_ARCHS, build_pretrained,
)

# ImageNet stats — used for all CNN models trained in cnn_comparison.ipynb
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_DEVICE = (
    torch.device('mps')  if torch.backends.mps.is_available()  else
    torch.device('cuda') if torch.cuda.is_available()           else
    torch.device('cpu')
)

# Default weight files relative to PROYECTO root
_DEFAULT_WEIGHTS: dict[str, str] = {
    'cnn-v1':            'notebooks/output_25epocs/models/cnn-v1_baseline_sin_aug.pth',
    'cnn-v2':            'notebooks/output_25epocs/models/cnn-v2_deep_sin_aug.pth',
    'cnn-v3':            'notebooks/output_25epocs/models/cnn-v3_residual_sin_aug.pth',
    'cnn-v4':            'notebooks/output_25epocs/models/cnn-v4_se_sin_aug.pth',
    'resnet18':          'notebooks/output_cnn/models/resnet18.pth',
    'resnet50':          'notebooks/output_cnn/models/resnet50.pth',
    'efficientnet_b0':   'notebooks/output_cnn/models/efficientnet_b0.pth',
    'efficientnet_b2':   'notebooks/output_cnn/models/efficientnet_b2.pth',
    'mobilenet_v3_small':'notebooks/output_cnn/models/mobilenet_v3_small.pth',
    'convnext_tiny':     'notebooks/output_cnn/models/convnext_tiny.pth',
}

# Classical pipeline file paths
_CLASSICAL_PATHS = {
    'scaler': 'notebooks/output/models/scaler_svm.joblib',
    'pca':    'notebooks/output/models/pca.joblib',
    'rf':     'notebooks/output/models/random_forest_cv.joblib',
    'svm':    'notebooks/output/models/svm_rbf.joblib',
}

_PROYECTO_ROOT = Path(__file__).resolve().parents[3]  # …/PROYECTO


def _resolve(rel: str) -> Path:
    return _PROYECTO_ROOT / rel


# ── CNN models ────────────────────────────────────────────────────────────────

def load_model(name: str, weights_path: str | None = None) -> dict:
    """
    Load a CNN model by name and return a model bundle:
        {'type': 'cnn', 'name': name, 'model': nn.Module, 'device': device}

    Args:
        name: one of the keys in CUSTOM_REGISTRY or PRETRAINED_ARCHS.
        weights_path: override the default .pth path.
    """
    path = Path(weights_path) if weights_path else _resolve(_DEFAULT_WEIGHTS[name])
    if not path.exists():
        raise FileNotFoundError(f"Weights not found: {path}")

    if name in CUSTOM_REGISTRY:
        model = CUSTOM_REGISTRY[name]()
    elif name in PRETRAINED_ARCHS:
        model = build_pretrained(name)
    else:
        raise ValueError(f"Unknown model name: {name!r}")

    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(_DEVICE)
    return {'type': 'cnn', 'name': name, 'model': model, 'device': _DEVICE}


# ── Classical ML pipeline ─────────────────────────────────────────────────────

def load_classical(clf_name: str = 'rf') -> dict:
    """
    Load the Scaler→PCA→classifier pipeline.
    clf_name: 'rf' (random forest) or 'svm'.
    """
    import joblib
    scaler = joblib.load(_resolve(_CLASSICAL_PATHS['scaler']))
    pca    = joblib.load(_resolve(_CLASSICAL_PATHS['pca']))
    clf    = joblib.load(_resolve(_CLASSICAL_PATHS[clf_name]))
    return {'type': 'classical', 'name': clf_name, 'scaler': scaler, 'pca': pca, 'clf': clf}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image, size: int = 64) -> torch.Tensor:
    """PIL RGB → (1, 3, size, size) float32 tensor, ImageNet-normalised."""
    arr = np.array(img.convert('RGB').resize((size, size)), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)
    t = (t - _MEAN) / _STD
    return t.unsqueeze(0)                          # (1, 3, H, W)


def _pil_to_flat(img: Image.Image, size: int = 64) -> np.ndarray:
    """PIL RGB → (1, size*size*3) float32 array for classical models."""
    arr = np.array(img.convert('RGB').resize((size, size)), dtype=np.float32) / 255.0
    return arr.flatten().reshape(1, -1)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(bundle: dict, img: Image.Image) -> dict:
    """
    Run inference on a PIL image.

    Returns:
        {
          'label':  'adware',          # top-1 class name
          'index':  0,                 # class index (0-4)
          'probs':  {'adware': 0.87, 'banking': 0.03, ...}
        }
    """
    if bundle['type'] == 'cnn':
        tensor = _pil_to_tensor(img).to(bundle['device'])
        with torch.no_grad():
            logits = bundle['model'](tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().tolist()

    elif bundle['type'] == 'classical':
        flat = _pil_to_flat(img)
        flat = bundle['scaler'].transform(flat)
        flat = bundle['pca'].transform(flat)
        if hasattr(bundle['clf'], 'predict_proba'):
            probs = bundle['clf'].predict_proba(flat)[0].tolist()
        else:
            idx = int(bundle['clf'].predict(flat)[0])
            probs = [1.0 if i == idx else 0.0 for i in range(len(CLASSES))]
    else:
        raise ValueError(f"Unknown bundle type: {bundle['type']!r}")

    idx   = int(np.argmax(probs))
    return {
        'label': CLASSES[idx],
        'index': idx,
        'probs': {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)},
    }
