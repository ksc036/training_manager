from fastapi.testclient import TestClient

from app.main import create_app


def test_datasets_page_renders_from_non_repo_cwd(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    approved_root = tmp_path / "annotation_complete"
    sample_dir = approved_root / "sample_a"
    (sample_dir / "Image").mkdir(parents=True)
    (sample_dir / "mask").mkdir(parents=True)
    (sample_dir / "Image" / "sample_a.png").write_bytes(b"image")
    (sample_dir / "mask" / "sample_a.png").write_bytes(b"mask")

    client = TestClient(create_app(project_root=tmp_path, approved_source_root=approved_root))

    response = client.get("/datasets")

    assert response.status_code == 200
    assert "Datasets" in response.text
    assert "sample_a" in response.text
    assert 'id="select_all_samples"' in response.text


def test_datasets_page_creates_dataset_manifest(tmp_path) -> None:
    approved_root = tmp_path / "annotation_complete"
    sample_dir = approved_root / "sample_a"
    (sample_dir / "Image").mkdir(parents=True)
    (sample_dir / "mask").mkdir(parents=True)
    (sample_dir / "Image" / "sample_a.png").write_bytes(b"image")
    (sample_dir / "mask" / "sample_a.png").write_bytes(b"mask")

    client = TestClient(create_app(project_root=tmp_path, approved_source_root=approved_root))

    response = client.post(
        "/datasets",
        data={"name": "baseline", "sample_id": "sample_a"},
    )

    assert response.status_code == 200
    assert "Created dataset baseline v001 with 1 sample(s)." in response.text
    assert (tmp_path / "data" / "datasets" / "manifests" / "baseline_v001.json").is_file()


def test_static_stylesheet_is_served() -> None:
    client = TestClient(create_app())

    response = client.get("/static/styles.css")

    assert response.status_code == 200
