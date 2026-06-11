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
        data={"mode": "magnetic"},
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


def test_analyze_default_mode_is_magnetic():
    """Issue 17: omitting mode must run the magnetic analysis (the old
    default 'combined' was removed from the UI)."""
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert "DECOMPOSITION" in r.text
    # magnetic-only output has no mechanical-representation block
    assert "Γ_mech" not in r.text


def test_analyze_combined_rejected():
    """'combined' is no longer a supported mode."""
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "combined"},
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


def test_analyze_no_filename():
    r = client.post(
        "/analyze",
        data={"mode": "magnetic"},
        files={"file": ("", b"dummy", "application/octet-stream")},
    )
    assert r.status_code == 422


def test_timeout_kills_analysis_and_recovers(monkeypatch):
    """Issue 6: a timed-out request must terminate its computation (no zombie
    blocking the single worker) and the service must serve the next request."""
    import multiprocessing
    import time

    import app as app_module

    monkeypatch.setattr(app_module, "_ANALYSIS_TIMEOUT", 0.3, raising=False)
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "displacive"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 500
    assert "timed out" in r.json()["detail"].lower()

    # The analysis child process must be gone shortly after the timeout
    deadline = time.time() + 5.0
    while multiprocessing.active_children() and time.time() < deadline:
        time.sleep(0.2)
    assert not multiprocessing.active_children()

    # Worker freed: a normal request must succeed afterwards
    monkeypatch.undo()
    with open(NIO_MCIF, "rb") as f:
        r2 = client.post(
            "/analyze",
            data={"mode": "magnetic"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r2.status_code == 200


def test_analyze_displacive():
    with open(NIO_MCIF, "rb") as f:
        r = client.post(
            "/analyze",
            data={"mode": "displacive"},
            files={"file": ("1.6_NiO.mcif", f, "application/octet-stream")},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert len(r.text) > 200
