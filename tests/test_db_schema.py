from sqlalchemy import create_engine, inspect

from app.db import Base
from app.models import DatasetSample, DatasetVersion, RunEvent, RunMetric, TrainingRun


def test_schema_contains_core_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)

    table_names = inspector.get_table_names()

    assert "dataset_versions" in table_names
    assert "dataset_samples" in table_names
    assert "training_runs" in table_names
    assert "run_metrics" in table_names
    assert "run_events" in table_names

    assert {column["name"] for column in inspector.get_columns("dataset_versions")} >= {
        "id",
        "name",
        "version",
        "source_root",
        "snapshot_mode",
        "sample_count",
        "manifest_path",
    }
    assert {column["name"] for column in inspector.get_columns("dataset_samples")} >= {
        "id",
        "dataset_version_id",
        "sample_id",
        "sample_path",
        "image_path",
        "mask_path",
        "image_hash",
        "mask_hash",
    }
    assert {column["name"] for column in inspector.get_columns("training_runs")} >= {
        "id",
        "dataset_version_id",
        "run_name",
        "model_name",
        "encoder_name",
        "trainer_backend",
        "status",
        "best_epoch",
        "best_checkpoint_path",
        "run_dir",
        "config_path",
        "log_path",
    }
    assert {column["name"] for column in inspector.get_columns("run_metrics")} >= {
        "id",
        "training_run_id",
        "epoch",
        "train_loss",
        "val_loss",
        "dice",
        "iou",
        "precision",
        "recall",
    }
    assert {column["name"] for column in inspector.get_columns("run_events")} >= {
        "id",
        "training_run_id",
        "event_type",
        "message",
    }

    assert any(
        fk["constrained_columns"] == ["dataset_version_id"]
        and fk["referred_table"] == "dataset_versions"
        and fk["referred_columns"] == ["id"]
        for fk in inspector.get_foreign_keys("dataset_samples")
    )
    assert any(
        fk["constrained_columns"] == ["dataset_version_id"]
        and fk["referred_table"] == "dataset_versions"
        and fk["referred_columns"] == ["id"]
        for fk in inspector.get_foreign_keys("training_runs")
    )
    assert any(
        fk["constrained_columns"] == ["training_run_id"]
        and fk["referred_table"] == "training_runs"
        and fk["referred_columns"] == ["id"]
        for fk in inspector.get_foreign_keys("run_metrics")
    )
    assert any(
        fk["constrained_columns"] == ["training_run_id"]
        and fk["referred_table"] == "training_runs"
        and fk["referred_columns"] == ["id"]
        for fk in inspector.get_foreign_keys("run_events")
    )

    assert any(
        set(constraint["column_names"]) == {"name", "version"}
        for constraint in inspector.get_unique_constraints("dataset_versions")
    )
    assert any(
        set(constraint["column_names"]) == {"dataset_version_id", "sample_id"}
        for constraint in inspector.get_unique_constraints("dataset_samples")
    )
    assert any(
        set(constraint["column_names"]) == {"training_run_id", "epoch"}
        for constraint in inspector.get_unique_constraints("run_metrics")
    )
