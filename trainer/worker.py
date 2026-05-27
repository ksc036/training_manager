from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

from app.db import get_session_factory
from app.repositories.runs import (
    replace_run_metric,
    update_training_run_checkpoint,
    update_training_run_status,
)

_STOP_REQUESTED = False


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status.json").write_text(
        json.dumps({"status": status}, indent=2),
        encoding="utf-8",
    )


def _project_root_from_run_dir(run_dir: Path) -> Path:
    return run_dir.parents[2]


def _persist_status(run_dir: Path, status: str) -> None:
    _write_status(run_dir, status)
    session_factory = get_session_factory(_project_root_from_run_dir(run_dir))
    with session_factory() as session:
        update_training_run_status(session, run_name=run_dir.name, status=status)
        session.commit()


def _persist_metric(run_dir: Path, epoch: int, dice: float) -> None:
    _persist_metric_full(run_dir, epoch=epoch, train_loss=None, val_loss=None, dice=dice, iou=None, precision=None, recall=None)


def _persist_metric_full(
    run_dir: Path,
    *,
    epoch: int,
    train_loss: float | None,
    val_loss: float | None,
    dice: float,
    iou: float | None,
    precision: float | None,
    recall: float | None,
) -> None:
    session_factory = get_session_factory(_project_root_from_run_dir(run_dir))
    train_loss_value = (
        round(max(0.05, 0.9 - (epoch * 0.08)), 4)
        if train_loss is None
        else round(float(train_loss), 4)
    )
    val_loss_value = (
        round(max(0.04, 1.0 - (epoch * 0.07)), 4)
        if val_loss is None
        else round(float(val_loss), 4)
    )
    iou_value = (
        round(max(0.0, dice - 0.08), 4)
        if iou is None
        else round(float(iou), 4)
    )
    precision_value = (
        round(min(0.99, dice + 0.04), 4)
        if precision is None
        else round(float(precision), 4)
    )
    recall_value = (
        round(max(0.0, dice - 0.03), 4)
        if recall is None
        else round(float(recall), 4)
    )
    with session_factory() as session:
        replace_run_metric(
            session,
            run_name=run_dir.name,
            epoch=epoch,
            train_loss=train_loss_value,
            val_loss=val_loss_value,
            dice=dice,
            iou=iou_value,
            precision=precision_value,
            recall=recall_value,
        )
        session.commit()


def _persist_best_checkpoint(run_dir: Path, epoch: int, dice: float) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "best.ckpt"
    checkpoint_path.write_text(
        json.dumps({"best_epoch": epoch, "best_dice": dice}, indent=2),
        encoding="utf-8",
    )
    session_factory = get_session_factory(_project_root_from_run_dir(run_dir))
    with session_factory() as session:
        update_training_run_checkpoint(
            session,
            run_name=run_dir.name,
            best_epoch=epoch,
            best_checkpoint_path=str(checkpoint_path),
        )
        session.commit()


def _handle_termination(signum, frame) -> None:  # type: ignore[no-untyped-def]
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def run_training(run_dir: Path) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _handle_termination)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    epochs = int(config["epochs"])
    epoch_delay_sec = float(os.environ.get("TRAINING_MANAGER_EPOCH_DELAY_SEC", "0") or "0")
    metrics_path = run_dir / "metrics.csv"
    log_path = run_dir / "train.log"

    _persist_status(run_dir, "running")
    log_path.write_text("starting training\n", encoding="utf-8")

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "dice"])
        writer.writeheader()
        best_dice = -1.0
        best_epoch = 0
        for epoch in range(1, epochs + 1):
            if _STOP_REQUESTED:
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write("training stopped\n")
                _persist_status(run_dir, "stopped")
                return
            dice = round(min(0.5 + (epoch * 0.05), 0.99), 4)
            writer.writerow({"epoch": epoch, "dice": dice})
            _persist_metric(run_dir, epoch=epoch, dice=dice)
            if dice >= best_dice:
                best_dice = dice
                best_epoch = epoch
                _persist_best_checkpoint(run_dir, epoch=best_epoch, dice=best_dice)
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"epoch {epoch}: dice={dice}\n")
            if epoch_delay_sec > 0:
                time.sleep(epoch_delay_sec)

    _persist_status(run_dir, "completed")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("Usage: python -m trainer.worker <run_dir>")
    run_training(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
