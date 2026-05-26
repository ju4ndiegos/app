"""
Multi-modal ensemble inference for APK malware classification.

Three modalities combined with manually-tuned weights (tabular=0.2, image=0.5, seq=0.3):
  - Tabular: Keras MLP on 345 manifest/permission features
  - Image:   PyTorch CNN-V3 (ResNet-style) on 64×64 DEX semantic image
  - Sequence: PyTorch BiLSTM+Attention on API call sequence

Weights live at: app/pipeline/weights/
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

# Suppress TF/Keras Python-level logs before the library is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
_DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)

# Lazy-loaded bundles — each populated once on first call
_TAB_BUNDLE: dict | None = None
_IMG_BUNDLE: dict | None = None
_SEQ_BUNDLE: dict | None = None

# Ensemble weights: [tabular, image, sequence]
_WEIGHTS = [0.2, 0.5, 0.3]


# ── Pickle compat ─────────────────────────────────────────────────────────────

class _NumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "numpy._core" in module:
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def _pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return _NumpyUnpickler(f).load()


# ── Model architectures (must match training exactly) ─────────────────────────

class _ResBlock(nn.Module):
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


class _CNN_V3(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.b1 = nn.Sequential(_ResBlock(3,   64),  nn.MaxPool2d(2))
        self.b2 = nn.Sequential(_ResBlock(64,  128), nn.MaxPool2d(2))
        self.b3 = nn.Sequential(_ResBlock(128, 256), nn.MaxPool2d(2))
        self.b4 = nn.Sequential(_ResBlock(256, 256), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, n_classes))

    def forward(self, x):
        return self.head(self.b4(self.b3(self.b2(self.b1(x)))))


class _AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1, bias=True))

    def forward(self, lstm_out):
        weights = F.softmax(self.attn(lstm_out), dim=1)
        return torch.sum(lstm_out * weights, dim=1)


class _DuplicateViewsEnsemble(nn.Module):
    def __init__(self, vocab_size=5000, embedding_dim=128, hidden_dim=128, num_classes=5):
        super().__init__()
        self.encoder_b = nn.ModuleDict({
            "emb":  nn.Embedding(vocab_size, embedding_dim, padding_idx=0),
            "lstm": nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.3, bidirectional=True),
            "attn": _AttentionBlock(),
        })
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(128, num_classes),
        )

    def forward(self, x_seq):
        x_emb = self.encoder_b["emb"](x_seq)
        lstm_out, _ = self.encoder_b["lstm"](x_emb)
        return self.classifier(self.encoder_b["attn"](lstm_out))


def _build_tabular_model(n_features: int, n_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(), layers.Dropout(0.3),
        layers.Dense(64,  activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return model


# ── Lazy bundle loaders ────────────────────────────────────────────────────────

def _load_tabular_bundle() -> dict:
    global _TAB_BUNDLE
    if _TAB_BUNDLE is None:
        art = _WEIGHTS_DIR / "tabular_artifacts"
        meta    = _pkl(art / "model_metadata.pkl")
        scaler  = _pkl(art / "scaler.pkl")
        le      = _pkl(art / "label_encoder.pkl")
        model   = _build_tabular_model(meta["n_features"], meta["n_classes"])
        model.load_weights(str(art / "best_model_mlp.weights.h5"))
        _TAB_BUNDLE = {"meta": meta, "scaler": scaler, "le": le, "model": model}
    return _TAB_BUNDLE


def _load_image_bundle() -> dict:
    global _IMG_BUNDLE
    if _IMG_BUNDLE is None:
        art = _WEIGHTS_DIR / "image_artifacts"
        le    = _pkl(art / "label_encoder_images.pkl")
        model = _CNN_V3(n_classes=len(le.classes_)).to(_DEVICE)
        state = torch.load(art / "cnn-v3_residual_sin_aug.pth",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _IMG_BUNDLE = {"model": model, "le": le}
    return _IMG_BUNDLE


def _load_sequence_bundle() -> dict:
    global _SEQ_BUNDLE
    if _SEQ_BUNDLE is None:
        art = _WEIGHTS_DIR / "sequence_artifacts"
        le       = _pkl(art / "label_encoder_sequence.pkl")
        tfidf    = _pkl(art / "tfidf_api_vectorizer.pkl")
        vocab    = _pkl(art / "api_vocab_dict.pkl")
        ckpt     = torch.load(art / "fase3_best.pt", map_location="cpu", weights_only=False)
        model    = _DuplicateViewsEnsemble(num_classes=len(le.classes_)).to(_DEVICE)
        clean    = {k: v for k, v in ckpt["state_dict"].items() if k != "class_w"}
        model.load_state_dict(clean, strict=True)
        model.eval()
        _SEQ_BUNDLE = {"model": model, "le": le, "tfidf": tfidf, "vocab": vocab}
    return _SEQ_BUNDLE


# ── Per-modality inference ─────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _infer_tabular(tab_row: dict) -> dict:
    b = _load_tabular_bundle()
    feat_names = b["meta"]["feature_names"]
    series = pd.Series({k: tab_row.get(k, 0) for k in feat_names})
    X = b["scaler"].transform(pd.DataFrame([series]))
    probs = b["model"].predict(X, verbose=0)[0]
    idx = int(np.argmax(probs))
    classes = list(b["le"].classes_)
    return {
        "label": classes[idx],
        "index": idx,
        "probs": np.array(probs, dtype=np.float64),
        "classes": classes,
    }


def _infer_image(image_path: str) -> dict:
    b = _load_image_bundle()
    img_np = np.array(Image.open(image_path).convert("RGB").resize((64, 64)),
                      dtype=np.float32) / 255.0
    t = torch.from_numpy(img_np).permute(2, 0, 1)
    t = ((t - _IMAGENET_MEAN) / _IMAGENET_STD).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs = F.softmax(b["model"](t), dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    classes = list(b["le"].classes_)
    return {
        "label": classes[idx],
        "index": idx,
        "probs": np.array(probs, dtype=np.float64),
        "classes": classes,
    }


def _infer_sequence(sequence: str, max_len: int = 300) -> dict:
    b = _load_sequence_bundle()
    tokens = sequence.split()
    encoded = [b["vocab"].get(tok, 1) for tok in tokens[:max_len]]
    padded = encoded + [0] * (max_len - len(encoded))
    x_seq = torch.LongTensor(padded).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        probs = F.softmax(b["model"](x_seq), dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    classes = list(b["le"].classes_)
    return {
        "label": classes[idx],
        "index": idx,
        "probs": np.array(probs, dtype=np.float64),
        "classes": classes,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def predict_ensemble(tab_row: dict, image_path: str, sequence: str) -> dict:
    """
    Run multi-modal ensemble on a single APK sample.

    Args:
        tab_row:    Full tabular row dict from extract_tabular() (400 columns).
        image_path: Path to the 64px PNG produced by extract_image().
        sequence:   Space-separated API token string from extract_sequence().

    Returns:
        {
          'label':      top-1 class name (str),
          'index':      class index (int),
          'probs':      {class_name: float, ...}  ← template-compatible,
          'modalities': {
            'tabular':  {label, probs},
            'image':    {label, probs},
            'sequence': {label, probs},
          }
        }
    """
    tab = _infer_tabular(tab_row)
    img = _infer_image(image_path)
    seq = _infer_sequence(sequence)

    classes = tab["classes"]
    w = np.array(_WEIGHTS) / np.sum(_WEIGHTS)
    combined = w[0] * tab["probs"] + w[1] * img["probs"] + w[2] * seq["probs"]

    idx = int(np.argmax(combined))
    return {
        "label":  classes[idx],
        "index":  idx,
        "probs":  {cls: round(float(p), 4) for cls, p in zip(classes, combined)},
        "modalities": {
            "tabular":  {"label": tab["label"], "probs": {c: round(float(p), 4) for c, p in zip(classes, tab["probs"])}},
            "image":    {"label": img["label"], "probs": {c: round(float(p), 4) for c, p in zip(classes, img["probs"])}},
            "sequence": {"label": seq["label"], "probs": {c: round(float(p), 4) for c, p in zip(classes, seq["probs"])}},
        },
    }
