import pytest
from django.test import Client
from pathlib import Path

TEST_APK = Path(__file__).parent.parent / "test_data" / "sombriyakotlin4.apk"

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return Client()


def test_upload_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"APK" in response.content or b"apk" in response.content


def test_upload_apk_returns_result(client):
    with open(TEST_APK, "rb") as f:
        response = client.post("/", {"apk_file": f, "label": "benign"}, follow=True)
    assert response.status_code == 200
    assert b"result" in response.content.lower() or b"sequence" in response.content.lower()


def test_result_contains_image(client):
    with open(TEST_APK, "rb") as f:
        response = client.post("/", {"apk_file": f, "label": "benign"}, follow=True)
    assert b"<img" in response.content
