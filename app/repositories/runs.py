from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.models import RunMetric, TrainingRun


def create_training_run_row(
    session: Session,
    *,
    dataset_version_id: int,
    run_name: str,
    model_name: str,
    encoder_name: str,
    trainer_backend: str,
    status: str,
    run_dir: str,
    config_path: str,
    log_path: str,
) -> TrainingRun:
    training_run = TrainingRun(
        dataset_version_id=dataset_version_id,
        run_name=run_name,
        model_name=model_name,
        encoder_name=encoder_name,
        trainer_backend=trainer_backend,
        status=status,
        run_dir=run_dir,
        config_path=config_path,
        log_path=log_path,
    )
    session.add(training_run)
    session.flush()
    return training_run


def update_training_run_status(
    session: Session,
    *,
    run_name: str,
    status: str,
) -> None:
    training_run = session.query(TrainingRun).filter_by(run_name=run_name).one_or_none()
    if training_run is None:
        return
    training_run.status = status
    session.flush()


def update_training_run_checkpoint(
    session: Session,
    *,
    run_name: str,
    best_epoch: int,
    best_checkpoint_path: str,
) -> None:
    training_run = session.query(TrainingRun).filter_by(run_name=run_name).one_or_none()
    if training_run is None:
        return
    training_run.best_epoch = best_epoch
    training_run.best_checkpoint_path = best_checkpoint_path
    session.flush()


def replace_run_metric(
    session: Session,
    *,
    run_name: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    dice: float,
    iou: float,
    precision: float,
    recall: float,
) -> None:
    training_run = session.query(TrainingRun).filter_by(run_name=run_name).one_or_none()
    if training_run is None:
        return

    session.execute(
        delete(RunMetric).where(
            RunMetric.training_run_id == training_run.id,
            RunMetric.epoch == epoch,
        )
    )
    session.add(
        RunMetric(
            training_run_id=training_run.id,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            dice=dice,
            iou=iou,
            precision=precision,
            recall=recall,
        )
    )
    session.flush()
