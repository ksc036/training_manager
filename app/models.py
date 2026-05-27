from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(40))
    source_root: Mapped[str] = mapped_column(Text)
    snapshot_mode: Mapped[str] = mapped_column(String(20))
    sample_count: Mapped[int] = mapped_column(Integer)
    manifest_path: Mapped[str] = mapped_column(Text)


class DatasetSample(Base):
    __tablename__ = "dataset_samples"
    __table_args__ = (UniqueConstraint("dataset_version_id", "sample_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    sample_id: Mapped[str] = mapped_column(String(255))
    sample_path: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text)
    mask_path: Mapped[str] = mapped_column(Text)
    image_hash: Mapped[str] = mapped_column(String(64))
    mask_hash: Mapped[str] = mapped_column(String(64))


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (UniqueConstraint("run_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    run_name: Mapped[str] = mapped_column(String(120))
    model_name: Mapped[str] = mapped_column(String(80))
    encoder_name: Mapped[str] = mapped_column(String(80))
    trainer_backend: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20))
    best_epoch: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_checkpoint_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    run_dir: Mapped[str] = mapped_column(Text)
    config_path: Mapped[str] = mapped_column(Text)
    log_path: Mapped[str] = mapped_column(Text)


class RunMetric(Base):
    __tablename__ = "run_metrics"
    __table_args__ = (UniqueConstraint("training_run_id", "epoch"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float] = mapped_column(Float)
    val_loss: Mapped[float] = mapped_column(Float)
    dice: Mapped[float] = mapped_column(Float)
    iou: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)


class RunEvent(Base):
    __tablename__ = "run_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    event_type: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
