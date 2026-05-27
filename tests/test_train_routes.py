from fastapi.testclient import TestClient
import json

from app.main import create_app


def _seed_dataset_manifest(tmp_path) -> None:
    manifest_dir = tmp_path / "data" / "datasets" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "baseline_v001.json").write_text(
        '{"name":"baseline","version":"v001","samples":[{"sample_id":"sample_a"}]}',
        encoding="utf-8",
    )


def test_train_page_renders(tmp_path) -> None:
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/train")

    assert response.status_code == 200
    assert "Train" in response.text
    assert "U-Net" in response.text
    assert "baseline v001" in response.text
    assert "External Script Model" in response.text
    assert 'name="encoder_name"' not in response.text


def test_train_page_shows_external_script_model_when_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_EXTERNAL_TRAINER_COMMAND", "python -m trainer.worker")
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/train")

    assert response.status_code == 200
    assert "External Script Model" in response.text


def test_train_page_creates_run_directory(tmp_path) -> None:
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/train",
        data={
            "dataset_version_id": "1",
            "model_name": "unet",
            "epochs": "5",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    runs_dir = tmp_path / "data" / "runs"
    run_dirs = [entry for entry in runs_dir.iterdir() if entry.is_dir()]

    assert response.status_code == 200
    assert "Queued run" in response.text
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "config.json").is_file()


def test_train_page_uses_default_encoder_for_selected_model(tmp_path) -> None:
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/train",
        data={
            "dataset_version_id": "1",
            "model_name": "deeplabv3",
            "epochs": "5",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    runs_dir = tmp_path / "data" / "runs"
    run_dir = next(entry for entry in runs_dir.iterdir() if entry.is_dir())
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert payload["encoder_name"] == "resnet50"


def test_train_page_rejects_invalid_dataset_version(tmp_path) -> None:
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/train",
        data={
            "dataset_version_id": "999",
            "model_name": "unet",
            "epochs": "5",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    assert response.status_code == 400
    assert "Choose a valid dataset version" in response.text


def test_train_page_creates_external_script_run_when_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_EXTERNAL_TRAINER_COMMAND", "python -m trainer.worker")
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/train",
        data={
            "dataset_version_id": "1",
            "model_name": "external-script-model",
            "epochs": "3",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    runs_dir = tmp_path / "data" / "runs"
    run_dir = next(entry for entry in runs_dir.iterdir() if entry.is_dir())
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert payload["trainer_backend"] == "external-script"
    assert payload["backend_config"] == {"entry_command": "python -m trainer.worker"}
    assert payload["encoder_name"] == "custom"


def test_train_page_creates_external_script_run_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    _seed_dataset_manifest(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(
        "/train",
        data={
            "dataset_version_id": "1",
            "model_name": "external-script-model",
            "epochs": "2",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    run_dir = next(entry for entry in (tmp_path / "data" / "runs").iterdir() if entry.is_dir())
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert payload["trainer_backend"] == "external-script"
    assert "trainer.external_adapter" in payload["backend_config"]["entry_command"]
    assert payload["encoder_name"] == "custom"
