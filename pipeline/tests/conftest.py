import django
import os
import pytest
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "apk_detector.settings")

TEST_APK = Path(__file__).parent.parent / "test_data" / "sombriyakotlin4.apk"


@pytest.fixture(scope="session")
def apk_path():
    assert TEST_APK.exists(), f"APK not found at {TEST_APK}"
    return str(TEST_APK)


@pytest.fixture(scope="session")
def schema():
    from pipeline.core.tabular import load_schema
    return load_schema()
