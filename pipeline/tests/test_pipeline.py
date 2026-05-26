import pytest
from pathlib import Path
from pipeline.core.pipeline import run_pipeline
from PIL import Image


@pytest.fixture(scope="module")
def pipeline_result(apk_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("pipeline_out")
    return run_pipeline(apk_path, str(out), label="benign", split="test")


def test_creates_all_outputs(pipeline_result):
    assert Path(pipeline_result["image_path"]).exists(), "Image file not created"
    assert Path(pipeline_result["seq_path"]).exists(), "Sequence file not created"
    assert Path(pipeline_result["tab_path"]).exists(), "Tabular file not created"


def test_hash_is_sha256(pipeline_result):
    assert len(pipeline_result["hash"]) == 64


def test_output_image_is_64x64_rgb(pipeline_result):
    img = Image.open(pipeline_result["image_path"])
    assert img.size == (64, 64)
    assert img.mode == "RGB"
