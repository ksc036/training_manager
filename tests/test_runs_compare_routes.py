from fastapi.testclient import TestClient

from app.main import create_app
from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService
from trainer.scanner import SampleRecord


def _seed_run(tmp_path, run_name: str, dice_values: list[float]) -> None:
    run_dir = tmp_path / "data" / "runs" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        (
            '{"dataset_version_id": 1, "model_name": "unet", '
            '"encoder_name": "resnet34", "trainer_backend": "simulated", "epochs": 5, '
            '"batch_size": 2, "learning_rate": 0.001}'
        ),
        encoding="utf-8",
    )
    metrics_lines = ["epoch,dice"]
    for index, value in enumerate(dice_values, start=1):
        metrics_lines.append(f"{index},{value}")
    (run_dir / "metrics.csv").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")


def _set_run_status(tmp_path, run_name: str, status: str) -> None:
    run_dir = tmp_path / "data" / "runs" / run_name
    (run_dir / "status.json").write_text(f'{{"status": "{status}"}}', encoding="utf-8")


def test_runs_page_lists_created_runs(tmp_path) -> None:
    _seed_run(tmp_path, "run_alpha", [0.52, 0.61])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/runs")

    assert response.status_code == 200
    assert "run_alpha" in response.text
    assert "resnet34" in response.text
    assert "simulated" in response.text
    assert 'name="status"' in response.text
    assert 'name="model_name"' in response.text


def test_runs_page_filters_by_status_and_model(tmp_path) -> None:
    _seed_run(tmp_path, "run_completed_unet", [0.52, 0.61])
    _seed_run(tmp_path, "run_failed_deeplab", [0.40, 0.44])
    _set_run_status(tmp_path, "run_completed_unet", "completed")
    _set_run_status(tmp_path, "run_failed_deeplab", "failed")
    failed_config = tmp_path / "data" / "runs" / "run_failed_deeplab" / "config.json"
    failed_config.write_text(
        (
            '{"dataset_version_id": 1, "model_name": "deeplabv3", '
            '"encoder_name": "resnet50", "trainer_backend": "simulated", "epochs": 5, '
            '"batch_size": 2, "learning_rate": 0.001}'
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/runs?status=failed&model_name=deeplabv3")

    assert response.status_code == 200
    assert "run_failed_deeplab" in response.text
    assert "run_completed_unet" not in response.text


def test_runs_page_filters_by_run_name_query(tmp_path) -> None:
    _seed_run(tmp_path, "run_alpha_target", [0.52, 0.61])
    _seed_run(tmp_path, "run_beta_other", [0.40, 0.44])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/runs?run_name_query=target")

    assert response.status_code == 200
    assert "run_alpha_target" in response.text
    assert "run_beta_other" not in response.text
    assert 'value="target"' in response.text


def test_runs_page_supports_pagination(tmp_path) -> None:
    for index in range(1, 13):
        _seed_run(tmp_path, f"run_{index:02d}", [0.50, 0.60])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/runs?page=2")

    assert response.status_code == 200
    assert "run_11" in response.text
    assert "run_12" in response.text
    assert "run_01" not in response.text
    assert "?page=1" in response.text


def test_compare_page_lists_ranked_runs(tmp_path) -> None:
    _seed_run(tmp_path, "run_low", [0.42, 0.51])
    _seed_run(tmp_path, "run_high", [0.66, 0.81])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/compare")

    assert response.status_code == 200
    assert "run_low" in response.text
    assert "run_high" in response.text
    assert "0.81" in response.text
    assert 'name="model_name"' in response.text
    assert 'name="sort"' in response.text


def test_compare_page_filters_and_sorts_runs(tmp_path) -> None:
    _seed_run(tmp_path, "run_unet_low", [0.42, 0.51])
    _seed_run(tmp_path, "run_unet_high", [0.66, 0.81])
    _seed_run(tmp_path, "run_deeplab", [0.77, 0.79])
    deeplab_config = tmp_path / "data" / "runs" / "run_deeplab" / "config.json"
    deeplab_config.write_text(
        (
            '{"dataset_version_id": 1, "model_name": "deeplabv3", '
            '"encoder_name": "resnet50", "trainer_backend": "simulated", "epochs": 5, '
            '"batch_size": 2, "learning_rate": 0.001}'
        ),
        encoding="utf-8",
    )
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/compare?model_name=unet&sort=best_dice_asc")

    assert response.status_code == 200
    assert "run_unet_low" in response.text
    assert "run_unet_high" in response.text
    assert "run_deeplab" not in response.text
    assert response.text.index("run_unet_low") < response.text.index("run_unet_high")
    assert 'value="unet"' in response.text


def test_compare_page_filters_by_run_name_query(tmp_path) -> None:
    _seed_run(tmp_path, "run_target_high", [0.66, 0.81])
    _seed_run(tmp_path, "run_other_low", [0.42, 0.51])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/compare?run_name_query=target")

    assert response.status_code == 200
    assert "run_target_high" in response.text
    assert "run_other_low" not in response.text
    assert 'value="target"' in response.text


def test_compare_page_supports_pagination(tmp_path) -> None:
    for index in range(1, 13):
        _seed_run(tmp_path, f"run_{index:02d}", [0.40 + (index * 0.01), 0.50 + (index * 0.01)])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/compare?page=2")

    assert response.status_code == 200
    assert "run_02" in response.text
    assert "run_01" in response.text
    assert "run_12" not in response.text
    assert "?page=1" in response.text


def test_compare_page_shows_dataset_label(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    sample_path = tmp_path / "annotation_complete" / "sample_a"
    image_path = sample_path / "Image" / "sample_a.png"
    mask_path = sample_path / "mask" / "sample_a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image")
    mask_path.write_bytes(b"mask")
    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=[
            SampleRecord(
                sample_id="sample_a",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        ],
    )
    client = TestClient(create_app(project_root=tmp_path))
    client.post(
        "/train",
        data={
            "dataset_version_id": str(dataset.id),
            "model_name": "unet",
            "encoder_name": "resnet34",
            "epochs": "2",
            "batch_size": "2",
            "learning_rate": "0.001",
        },
    )

    response = client.get("/compare")

    assert response.status_code == 200
    assert "baseline v001" in response.text


def test_run_detail_page_shows_metrics_and_log(tmp_path) -> None:
    _seed_run(tmp_path, "run_detail", [0.55, 0.72])
    run_dir = tmp_path / "data" / "runs" / "run_detail"
    (run_dir / "train.log").write_text("starting training\nepoch 2: dice=0.72\n", encoding="utf-8")
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "best.ckpt").write_text("checkpoint", encoding="utf-8")
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/runs/run_detail")

    assert response.status_code == 200
    assert "run_detail" in response.text
    assert "0.72" in response.text
    assert "starting training" in response.text
    assert "baseline v001" in response.text
    assert "/runs/run_detail/artifacts/config" in response.text
    assert "/runs/run_detail/artifacts/log" in response.text
    assert "/runs/run_detail/artifacts/metrics" in response.text
    assert "/runs/run_detail/artifacts/checkpoint" in response.text


def test_run_detail_page_can_retry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    _seed_run(tmp_path, "run_retry", [0.55, 0.72])
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post("/runs/run_retry/retry")

    runs_dir = tmp_path / "data" / "runs"
    run_names = sorted(entry.name for entry in runs_dir.iterdir() if entry.is_dir())

    assert response.status_code == 200
    assert "Queued retry run" in response.text
    assert "run_retry" in response.text
    assert len(run_names) == 2


def test_run_detail_page_can_stop_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_EPOCH_DELAY_SEC", "0.3")
    service = RunService(project_root=tmp_path)
    run = service.start_run(
        dataset_version_id=1,
        model_name="unet",
        encoder_name="resnet34",
        epochs=10,
        batch_size=2,
        learning_rate=1e-3,
    )
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post(f"/runs/{run.run_name}/stop")

    assert response.status_code == 200
    assert "Stopped run" in response.text


def test_run_artifact_routes_return_files(tmp_path) -> None:
    _seed_run(tmp_path, "run_files", [0.55, 0.72])
    run_dir = tmp_path / "data" / "runs" / "run_files"
    (run_dir / "train.log").write_text("log line\n", encoding="utf-8")
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "best.ckpt").write_text("checkpoint", encoding="utf-8")
    client = TestClient(create_app(project_root=tmp_path))

    config_response = client.get("/runs/run_files/artifacts/config")
    metrics_response = client.get("/runs/run_files/artifacts/metrics")
    log_response = client.get("/runs/run_files/artifacts/log")
    checkpoint_response = client.get("/runs/run_files/artifacts/checkpoint")

    assert config_response.status_code == 200
    assert '"model_name": "unet"' in config_response.text
    assert metrics_response.status_code == 200
    assert "epoch,dice" in metrics_response.text
    assert log_response.status_code == 200
    assert "log line" in log_response.text
    assert checkpoint_response.status_code == 200
    assert "checkpoint" in checkpoint_response.text
