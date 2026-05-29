from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


DATA = Path(__file__).parent.parent / "tests" / "data"
NIO_MCIF = DATA / "1.6_NiO.mcif"


def test_analyze_bad_extension():
    r = client.post(
        "/analyze",
        data={"mode": "combined"},
        files={"file": ("data.txt", b"dummy content", "text/plain")},
    )
    assert r.status_code == 422
    assert "detail" in r.json()


def test_analyze_bad_mode():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "notamode"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 422


def test_analyze_magnetic():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "magnetic"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert len(r.text) > 200


def test_analyze_combined():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "combined"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert len(r.text) > 200
