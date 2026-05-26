import numpy as np
import pytest
from pipeline.core.image import extract_image


def test_image_is_64x64_rgb(apk_path):
    img = extract_image(apk_path)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_image_channels_differ(apk_path):
    img = extract_image(apk_path)
    arr = np.array(img)
    # Semantic encoding gives different byte ranges to each channel
    assert not (arr[:, :, 0] == arr[:, :, 1]).all(), "R and G channels are identical — semantic encoding broken"


def test_image_not_blank(apk_path):
    img = extract_image(apk_path)
    arr = np.array(img)
    assert arr.std() > 0, "Image is blank (all pixels equal)"
