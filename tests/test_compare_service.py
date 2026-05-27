from pathlib import Path

from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService
from app.services.compare import rank_runs_by_best_dice
from app.services.compare import list_ranked_runs
from trainer.scanner import SampleRecord


def test_rank_runs_by_best_dice_orders_highest_first() -> None:
    ranked = rank_runs_by_best_dice(
        [
            {"run_name": "run_b", "best_dice": 0.81},
            {"run_name": "run_a", "best_dice": 0.92},
        ]
    )

    assert [row["run_name"] for row in ranked] == ["run_a", "run_b"]


def test_list_ranked_runs_reads_metrics_csv(tmp_path: Path) -> None:
    runs_dir = tmp_path / "data" / "runs" / "run_001"
    runs_dir.mkdir(parents=True)
    (runs_dir / "config.json").write_text(
        (
            '{"dataset_version_id": 1, "model_name": "unet", '
            '"encoder_name": "resnet34", "epochs": 5, '
            '"batch_size": 2, "learning_rate": 0.001}'
        ),
        encoding="utf-8",
    )
    (runs_dir / "metrics.csv").write_text(
        "epoch,dice\n1,0.61\n2,0.77\n",
        encoding="utf-8",
    )

    ranked = list_ranked_runs(tmp_path)

    assert ranked[0]["run_name"] == "run_001"
    assert ranked[0]["best_dice"] == 0.77


def test_list_ranked_runs_reads_database_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    service = RunService(project_root=tmp_path)

    service.start_run(
        dataset_version_id=7,
        model_name="unet",
        encoder_name="resnet34",
        epochs=3,
        batch_size=2,
        learning_rate=1e-3,
    )
    run_dir = next((tmp_path / "data" / "runs").iterdir())
    (run_dir / "metrics.csv").unlink()

    ranked = list_ranked_runs(tmp_path)

    assert len(ranked) == 1
    assert ranked[0]["dataset_version_id"] == 7
    assert ranked[0]["best_dice"] == 0.65


def test_list_ranked_runs_includes_dataset_label(tmp_path: Path, monkeypatch) -> None:
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

    service = RunService(project_root=tmp_path)
    service.start_run(
        dataset_version_id=dataset.id,
        model_name="unet",
        encoder_name="resnet34",
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
    )

    ranked = list_ranked_runs(tmp_path)

    assert ranked[0]["dataset_label"] == "baseline v001"
