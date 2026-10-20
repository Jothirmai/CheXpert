"""Tests for the Chest Xpert FastAPI backend.

These tests use FastAPI's TestClient and mock the heavy model loading
where appropriate to avoid GPU dependency.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["service"] == "chest-xpert"
    assert "model_loaded" in data


def test_ready_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/ready")
    assert res.status_code == 200
    data = res.json()
    assert "ready" in data
    assert "checks" in data


def test_create_analysis_rejects_bad_extension():
    client = TestClient(app)
    res = client.post(
        "/api/v1/analyses",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert res.status_code == 400
    data = res.json()
    assert "Unsupported file type" in data["detail"]


def test_create_analysis_too_large():
    client = TestClient(app)
    # Create a fake large file (> 50 MB)
    large = b"0" * (51 * 1024 * 1024)
    res = client.post(
        "/api/v1/analyses",
        files={"file": ("large.png", large, "image/png")},
    )
    assert res.status_code == 413
    data = res.json()
    assert "File too large" in data["detail"]


def test_get_nonexistent_analysis():
    client = TestClient(app)
    res = client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")
    assert res.status_code == 404


def test_report_download_nonexistent():
    client = TestClient(app)
    res = client.get(
        "/api/v1/report/00000000-0000-0000-0000-000000000000/download?format=pdf"
    )
    assert res.status_code == 404


def test_diagnostics_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/admin/diagnostics")
    assert res.status_code == 200
    data = res.json()
    assert "gpu_name" in data
    assert "host_ram_used_gb" in data


def test_cleanup_dry_run():
    client = TestClient(app)
    res = client.post("/api/v1/admin/cleanup?dry_run=true")
    assert res.status_code == 200
    data = res.json()
    assert "candidate_count" in data


def test_cache_sizes_endpoint():
    client = TestClient(app)
    res = client.get("/api/v1/admin/cache-sizes")
    assert res.status_code == 200
    data = res.json()
    assert "torchxrayvision_model_weights_mb" in data
