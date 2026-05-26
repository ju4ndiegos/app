"""
Full APK preprocessing pipeline orchestrator.
Runs all three modalities, writes outputs to disk, then runs ensemble prediction.
"""

import csv
import hashlib
import json
from pathlib import Path

from pipeline.core.image import extract_image
from pipeline.core.sequence import extract_sequence
from pipeline.core.tabular import extract_tabular, load_schema


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(
    apk_path: str,
    output_dir: str,
    label: str = "",
    split: str = "new",
    schema: list[str] | None = None,
    image_sizes: list[int] | None = None,
) -> dict:
    """
    Extract all three modalities from an APK and run ensemble prediction.

    Args:
        image_sizes: pixel sizes to generate (default: [64, 256]).

    Returns a dict with:
      hash, image_paths, seq_path, tab_path, sequence, tabular_row, prediction
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if schema is None:
        schema = load_schema()
    if image_sizes is None:
        image_sizes = [64, 256]

    h = _sha256(apk_path)

    # --- Image (one file per requested size) ---
    image_paths = {}
    for size in image_sizes:
        img = extract_image(apk_path, size=size)
        img_path = out / f"{split}_{h}_{size}.png"
        img.save(img_path)
        image_paths[size] = str(img_path)

    # back-compat: image_path points to the smallest size
    img_path = image_paths[min(image_sizes)]

    # --- Sequence ---
    seq = extract_sequence(apk_path)
    seq_record = {
        "GMLnames": {"0": h},
        "Label":    {"0": seq},
        "Class":    {"0": label},
    }
    seq_path = out / f"{split}_sequence_{h[:16]}.json"
    with open(seq_path, "w") as f:
        json.dump(seq_record, f)

    # --- Tabular ---
    tab_row = extract_tabular(apk_path, schema, label)
    tab_path = out / f"{split}_tabular_{h[:16]}.csv"
    with open(tab_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["apk_name"] + schema + ["Class"])
        writer.writeheader()
        writer.writerow(tab_row)

    # --- Ensemble Prediction ---
    prediction: dict | None = None
    try:
        from pipeline.core.ensemble import predict_ensemble
        prediction = predict_ensemble(tab_row, image_paths[64], seq)
    except Exception as exc:
        prediction = {"error": str(exc)}

    return {
        "hash":         h,
        "image_path":   str(img_path),   # smallest size, back-compat
        "image_paths":  image_paths,
        "seq_path":     str(seq_path),
        "tab_path":     str(tab_path),
        "sequence":     seq,
        "tabular_row":  tab_row,
        "prediction":   prediction,
    }
