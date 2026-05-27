import os
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from app.db import get_session_factory
from app.models import RunMetric, TrainingRun
from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService
from trainer.runners import resolve_backend_command
from trainer.scanner import SampleRecord


def test_start_run_creates_run_directory_and_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    service = RunService(project_root=tmp_path)

    run = service.start_run(
        dataset_version_id=1,
        model_name="unet",
        encoder_name="resnet34",
        epochs=5,
        batch_size=2,
        learning_rate=1e-3,
    )

    assert run.id == 1
    assert (Path(run.run_dir) / "config.json").is_file()
    assert (Path(run.run_dir) / "metrics.csv").is_file()
    assert (Path(run.run_dir) / "train.log").is_file()
    assert run.status == "queued"
    payload = json.loads((Path(run.run_dir) / "config.json").read_text(encoding="utf-8"))
    assert payload["trainer_backend"] == "external-script"
    assert "trainer.external_adapter" in payload["backend_config"]["entry_command"]


def test_list_runs_reads_worker_status_and_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    service = RunService(project_root=tmp_path)

    service.start_run(
        dataset_version_id=1,
        model_name="unet",
        encoder_name="resnet34",
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
    )

    runs = service.list_runs()

    assert len(runs) == 1
    assert runs[0].status == "completed"
    assert runs[0].best_dice is not None


def test_start_run_persists_status_and_metrics_to_database(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    service = RunService(project_root=tmp_path)

    run = service.start_run(
        dataset_version_id=1,
        model_name="unet",
        encoder_name="resnet34",
        epochs=4,
        batch_size=2,
        learning_rate=1e-3,
    )

    session_factory = get_session_factory(tmp_path)
    with session_factory() as session:
        training_run = session.query(TrainingRun).filter_by(id=run.id).one()
        metrics = (
            session.query(RunMetric)
            .filter_by(training_run_id=run.id)
            .order_by(RunMetric.epoch.asc())
            .all()
        )

    assert training_run.status == "completed"
    assert training_run.best_epoch == 4
    assert training_run.best_checkpoint_path.endswith("checkpoints/best.ckpt")
    assert [metric.epoch for metric in metrics] == [1, 2, 3, 4]
    assert metrics[-1].dice >= metrics[0].dice


def test_resolve_backend_command_supports_external_script_backend() -> None:
    command = resolve_backend_command(
        trainer_backend="external-script",
        run_dir=Path("/tmp/run-dir"),
        backend_config={"entry_command": "python -m custom_trainer"},
    )

    assert command == [os.sys.executable, "-m", "custom_trainer", "/tmp/run-dir"]


def test_resolve_backend_command_rejects_missing_external_script_command() -> None:
    with pytest.raises(ValueError, match="entry_command"):
        resolve_backend_command(
            trainer_backend="external-script",
            run_dir=Path("/tmp/run-dir"),
            backend_config={},
        )


def test_default_external_adapter_command_executes_sync_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    service = RunService(project_root=tmp_path)

    run = service.start_run(
        dataset_version_id=1,
        model_name="external-script-model",
        encoder_name="custom",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    run_dir = Path(run.run_dir)

    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "train.log").is_file()
    assert (run_dir / "checkpoints" / "best.ckpt").is_file()
    assert "external adapter" in (run_dir / "train.log").read_text(encoding="utf-8").lower()


def test_default_external_adapter_writes_dataset_summary_and_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    samples: list[SampleRecord] = []
    for index in range(1, 5):
        sample_path = tmp_path / "annotation_complete" / f"sample_{index:02d}"
        image_path = sample_path / "Image" / f"sample_{index:02d}.png"
        mask_path = sample_path / "mask" / f"sample_{index:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{index}".encode("utf-8"))
        mask_path.write_bytes(f"mask-{index}".encode("utf-8"))
        samples.append(
            SampleRecord(
                sample_id=f"sample_{index:02d}",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=samples,
    )
    service = RunService(project_root=tmp_path)

    run = service.start_run(
        dataset_version_id=dataset.id,
        model_name="external-script-model",
        encoder_name="custom",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    run_dir = Path(run.run_dir)
    summary_payload = json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    split_payload = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))

    assert summary_payload["dataset_name"] == "baseline"
    assert summary_payload["dataset_version"] == "v001"
    assert summary_payload["sample_count"] == 4
    assert summary_payload["train_count"] == 3
    assert summary_payload["val_count"] == 1
    assert len(split_payload["train_samples"]) == 3
    assert len(split_payload["val_samples"]) == 1


def test_default_external_adapter_builds_workspace_links(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    samples: list[SampleRecord] = []
    for index in range(1, 4):
        sample_path = tmp_path / "annotation_complete" / f"sample_{index:02d}"
        image_path = sample_path / "Image" / f"sample_{index:02d}.png"
        mask_path = sample_path / "mask" / f"sample_{index:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(f"image-{index}".encode("utf-8"))
        mask_path.write_bytes(f"mask-{index}".encode("utf-8"))
        samples.append(
            SampleRecord(
                sample_id=f"sample_{index:02d}",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=samples,
    )
    service = RunService(project_root=tmp_path)
    run = service.start_run(
        dataset_version_id=dataset.id,
        model_name="external-script-model",
        encoder_name="custom",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    run_dir = Path(run.run_dir)
    workspace_dir = run_dir / "workspace"
    train_dirs = sorted((workspace_dir / "train").iterdir())
    val_dirs = sorted((workspace_dir / "val").iterdir())

    assert len(train_dirs) == 2
    assert len(val_dirs) == 1
    assert (train_dirs[0] / "Image").is_dir()
    assert (train_dirs[0] / "mask").is_dir()
    train_image_links = list((train_dirs[0] / "Image").iterdir())
    train_mask_links = list((train_dirs[0] / "mask").iterdir())
    assert len(train_image_links) == 1
    assert len(train_mask_links) == 1
    assert train_image_links[0].is_symlink()
    assert train_mask_links[0].is_symlink()


def test_external_adapter_runs_actual_torch_training_on_valid_images(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    samples: list[SampleRecord] = []
    for index in range(1, 5):
        sample_path = tmp_path / "annotation_complete" / f"sample_{index:02d}"
        image_path = sample_path / "Image" / f"sample_{index:02d}.png"
        mask_path = sample_path / "mask" / f"sample_{index:02d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (16, 16), color=(index * 30, 0, 0))
        mask = Image.new("L", (16, 16), color=255 if index % 2 == 0 else 0)
        image.save(image_path)
        mask.save(mask_path)
        samples.append(
            SampleRecord(
                sample_id=f"sample_{index:02d}",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=samples,
    )
    service = RunService(project_root=tmp_path)
    run = service.start_run(
        dataset_version_id=dataset.id,
        model_name="external-script-model",
        encoder_name="custom",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    run_dir = Path(run.run_dir)
    metrics_lines = (run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    checkpoint_payload = json.loads(
        (run_dir / "checkpoints" / "best.ckpt").read_text(encoding="utf-8")
    )

    assert metrics_lines[0] == "epoch,train_loss,val_loss,dice,iou,precision,recall"
    assert len(metrics_lines) == 3
    assert checkpoint_payload["best_epoch"] in (1, 2)
    assert checkpoint_payload["best_dice"] >= 0.0


def test_stop_run_marks_async_run_as_stopped(tmp_path: Path, monkeypatch) -> None:
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

    assert (Path(run.run_dir) / "run.pid").is_file()

    stopped = service.stop_run(run.run_name)
    assert stopped is True

    for _ in range(30):
        detail = service.get_run_detail(run.run_name)
        if detail is not None and detail.summary.status == "stopped":
            break
        time.sleep(0.1)

    detail = service.get_run_detail(run.run_name)
    assert detail is not None
    assert detail.summary.status == "stopped"
