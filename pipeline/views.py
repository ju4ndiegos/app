import csv
import hashlib
import io
import json
import os
import tempfile

from django.conf import settings
from django.shortcuts import redirect, render, get_object_or_404
from PIL import Image

from pipeline.core.ensemble import predict_ensemble
from pipeline.core.pipeline import run_pipeline
from pipeline.core.tabular import load_schema
from pipeline.forms import APKUploadForm, DirectFeaturesForm
from pipeline.models import AnalysisResult

_SCHEMA = None


def _get_schema():
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = load_schema()
    return _SCHEMA


def upload(request):
    if request.method == "POST":
        form = APKUploadForm(request.POST, request.FILES)
        if form.is_valid():
            apk_file = request.FILES["apk_file"]
            label    = form.cleaned_data["label"]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".apk") as tmp:
                for chunk in apk_file.chunks():
                    tmp.write(chunk)
                tmp_path = tmp.name

            try:
                out_dir = settings.MEDIA_ROOT / "results"
                result  = run_pipeline(
                    tmp_path, str(out_dir),
                    label=label, split="new", schema=_get_schema(),
                    image_sizes=[64, 256],
                )
            finally:
                os.unlink(tmp_path)

            h   = result["hash"]
            seq = result["sequence"]
            tab = result["tabular_row"]

            # store the 256px image for display; fall back to 64px
            display_path = result["image_paths"].get(256, result["image_path"])
            rel_img = os.path.relpath(display_path, settings.MEDIA_ROOT)
            tab_features = {k: v for k, v in tab.items() if k not in ("apk_name", "Class")}

            AnalysisResult.objects.update_or_create(
                apk_hash=h,
                defaults={
                    "label":            label,
                    "image":            rel_img,
                    "sequence_preview": " ".join(seq.split()[:100]),
                    "tabular_json":     tab_features,
                    "prediction_json":  result.get("prediction"),
                },
            )
            return redirect("result", apk_hash=h)
    else:
        form = APKUploadForm()

    return render(request, "pipeline/upload.html", {
        "form": form,
        "direct_form": DirectFeaturesForm(prefix="direct"),
        "active_tab": "apk",
    })


def direct_upload(request):
    if request.method != "POST":
        return redirect("upload")

    form = DirectFeaturesForm(request.POST, request.FILES, prefix="direct")
    if not form.is_valid():
        return render(request, "pipeline/upload.html", {
            "form": APKUploadForm(),
            "direct_form": form,
            "active_tab": "features",
        })

    img_file  = request.FILES["direct-image_file"]
    seq_file  = request.FILES["direct-sequence_file"]
    tab_file  = request.FILES["direct-tabular_file"]
    label     = form.cleaned_data["label"]

    try:
        img_bytes = img_file.read()
        h = hashlib.sha256(img_bytes).hexdigest()

        out_dir = settings.MEDIA_ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{h}.png"
        Image.open(io.BytesIO(img_bytes)).convert("RGB").save(str(img_path))
        rel_img = os.path.relpath(str(img_path), settings.MEDIA_ROOT)

        seq_raw = json.loads(seq_file.read().decode())
        if isinstance(seq_raw, list):
            sequence = " ".join(str(t) for t in seq_raw)
        elif isinstance(seq_raw, dict):
            val = seq_raw.get("sequence") or seq_raw.get("tokens") or seq_raw.get("api_calls") or ""
            sequence = " ".join(val) if isinstance(val, list) else str(val)
        else:
            sequence = str(seq_raw)

        _SKIP = {"apk_name", "Class", "split"}
        reader = csv.DictReader(io.StringIO(tab_file.read().decode()))
        row = next(reader)
        tab_row = {k: int(float(v)) for k, v in row.items() if k not in _SKIP}

        prediction = predict_ensemble(tab_row, str(img_path), sequence)

    except Exception as exc:
        return render(request, "pipeline/upload.html", {
            "form": APKUploadForm(),
            "direct_form": form,
            "active_tab": "features",
            "direct_error": f"Could not process files: {exc}",
        })

    AnalysisResult.objects.update_or_create(
        apk_hash=h,
        defaults={
            "label":            label,
            "image":            rel_img,
            "sequence_preview": " ".join(sequence.split()[:100]),
            "tabular_json":     tab_row,
            "prediction_json":  prediction,
        },
    )
    return redirect("result", apk_hash=h)


def result(request, apk_hash):
    obj = get_object_or_404(AnalysisResult, apk_hash=apk_hash)
    pred = obj.prediction_json or {}
    probs_sorted = sorted(
        pred.get("probs", {}).items(), key=lambda x: x[1], reverse=True
    )
    return render(request, "pipeline/result.html", {
        "obj":          obj,
        "groups":       obj.active_features(),
        "tokens":       obj.sequence_preview.split()[:100],
        "prediction":   pred,
        "probs_sorted": probs_sorted,
    })
