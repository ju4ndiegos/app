import os
import tempfile

from django.conf import settings
from django.shortcuts import redirect, render, get_object_or_404

from pipeline.core.pipeline import run_pipeline
from pipeline.core.tabular import load_schema
from pipeline.forms import APKUploadForm
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

    return render(request, "pipeline/upload.html", {"form": form})


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
