from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from app.api.routes_datasets import router
from app.services.datasets import DatasetRegistryService
from trainer.scanner import SampleRecord


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size, color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _seed_dataset(tmp_path: Path) -> int:
    project_root = tmp_path
    approved_root = tmp_path / "annotation_complete"
    sample_a = approved_root / "sample_a"
    sample_b = approved_root / "sample_b"

    _write_png(sample_a / "Image" / "sample_a.png", (8, 10), (255, 0, 0))
    _write_png(sample_a / "mask" / "sample_a.png", (8, 10), (0, 0, 0))
    _write_png(sample_b / "Image" / "sample_b.png", (12, 14), (0, 255, 0))
    _write_png(sample_b / "mask" / "sample_b.png", (12, 14), (0, 0, 0))

    service = DatasetRegistryService(project_root)
    record = service.create_dataset_version(
        "baseline",
        [
            SampleRecord(
                sample_id="sample_a",
                sample_path=sample_a,
                image_path=sample_a / "Image" / "sample_a.png",
                mask_path=sample_a / "mask" / "sample_a.png",
            ),
            SampleRecord(
                sample_id="sample_b",
                sample_path=sample_b,
                image_path=sample_b / "Image" / "sample_b.png",
                mask_path=sample_b / "mask" / "sample_b.png",
            ),
        ],
    )
    return record.id


def _make_client(project_root: Path) -> TestClient:
    app = FastAPI()
    app.state.project_root = project_root
    app.state.approved_source_root = project_root / "annotation_complete"
    app.include_router(router)
    return TestClient(app)


def test_approved_samples_api_returns_paginated_rows(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/samples", params={"page": 1, "page_size": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["page"] == 1
    assert payload["page_size"] == 1
    assert payload["total_count"] == 2
    assert payload["total_pages"] == 2
    assert payload["items"][0]["sample_id"] == "sample_a"
    assert payload["items"][0]["width"] == 8
    assert payload["items"][0]["height"] == 10
    assert payload["items"][0]["status"] == "ready"
    assert payload["items"][0]["selectable"] is True


def test_approved_samples_api_filters_by_query_and_resolution(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.get(
        "/api/datasets/samples",
        params={"query": "sample_b", "resolution_bucket": "<=256"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 1
    assert payload["items"][0]["sample_id"] == "sample_b"


def test_create_dataset_version_api_from_approved_samples(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.post(
        "/api/datasets/versions",
        json={"name": "baseline", "sample_ids": ["sample_a", "sample_b"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "baseline"
    assert payload["version"] == "v002"
    assert payload["sample_count"] == 2


def test_list_dataset_versions_api_returns_labels(tmp_path: Path) -> None:
    _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/versions")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["label"] == "baseline v001"


def test_dataset_samples_api_returns_paginated_samples_with_dimensions(tmp_path: Path) -> None:
    dataset_version_id = _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.get(
        f"/api/datasets/{dataset_version_id}/samples",
        params={"page": 1, "page_size": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset_version_id"] == dataset_version_id
    assert payload["page"] == 1
    assert payload["page_size"] == 1
    assert payload["total"] == 2
    assert payload["total_pages"] == 2
    assert payload["has_next"] is True
    assert payload["has_prev"] is False
    assert len(payload["items"]) == 1
    assert payload["items"][0]["sample_id"] == "sample_a"
    assert payload["items"][0]["width"] == 8
    assert payload["items"][0]["height"] == 10


def test_dataset_samples_api_filters_by_sample_id_query(tmp_path: Path) -> None:
    dataset_version_id = _seed_dataset(tmp_path)
    client = _make_client(tmp_path)

    response = client.get(
        f"/api/datasets/{dataset_version_id}/samples",
        params={"sample_id": "sample_b"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert [item["sample_id"] for item in payload["items"]] == ["sample_b"]
    assert payload["items"][0]["width"] == 12
    assert payload["items"][0]["height"] == 14


def test_dataset_samples_api_returns_404_for_missing_dataset(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    response = client.get("/api/datasets/999/samples")

    assert response.status_code == 404
