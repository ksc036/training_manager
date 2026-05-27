from __future__ import annotations

import json
import csv
import os
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.db import get_session_factory
from app.models import RunMetric, TrainingRun
from app.repositories.runs import create_training_run_row, update_training_run_status
from trainer.model_registry import get_registered_model
from trainer.runners import launch_training_worker

@dataclass
class RunRecord:
    id: int
    run_name: str
    run_dir: str
    status: str
    dataset_version_id: int
    model_name: str


@dataclass(frozen=True)
class RunSummary:
    run_name: str
    run_dir: str
    status: str
    dataset_version_id: int
    model_name: str
    encoder_name: str
    trainer_backend: str
    epochs: int
    batch_size: int
    learning_rate: float
    best_dice: float | None


@dataclass(frozen=True)
class RunMetricRecord:
    epoch: int
    dice: float


@dataclass(frozen=True)
class RunDetail:
    summary: RunSummary
    dataset_label: str
    metrics: list[RunMetricRecord]
    log_text: str


class RunService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._session_factory = get_session_factory(project_root)

    def start_run(
        self,
        dataset_version_id: int,
        model_name: str,
        encoder_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> RunRecord:
        registered_model = get_registered_model(model_name)
        trainer_backend = (
            registered_model.trainer_backend if registered_model is not None else "simulated"
        )
        backend_config = (
            registered_model.backend_config if registered_model is not None else {}
        )
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        run_dir = self.project_root / "data" / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_version_id": dataset_version_id,
                    "model_name": model_name,
                    "encoder_name": encoder_name,
                    "trainer_backend": trainer_backend,
                    "backend_config": backend_config,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            json.dumps({"status": "queued"}, indent=2),
            encoding="utf-8",
        )
        with self._session_factory() as session:
            training_run = create_training_run_row(
                session,
                dataset_version_id=dataset_version_id,
                run_name=run_name,
                model_name=model_name,
                encoder_name=encoder_name,
                trainer_backend=trainer_backend,
                status="queued",
                run_dir=str(run_dir),
                config_path=str(config_path),
                log_path=str(run_dir / "train.log"),
            )
            session.commit()
        launch_training_worker(
            self.project_root,
            run_dir,
            trainer_backend=trainer_backend,
            backend_config=backend_config,
        )
        return RunRecord(
            id=training_run.id,
            run_name=run_name,
            run_dir=str(run_dir),
            status="queued",
            dataset_version_id=dataset_version_id,
            model_name=model_name,
        )

    def list_runs(self) -> list[RunSummary]:
        runs_dir = self.project_root / "data" / "runs"
        if not runs_dir.is_dir():
            return []

        summaries: list[RunSummary] = []
        seen_run_names: set[str] = set()
        with self._session_factory() as session:
            rows = session.query(TrainingRun).order_by(TrainingRun.id.asc()).all()
            for row in rows:
                summary = self._summary_from_run_dir(
                    run_dir=Path(row.run_dir),
                    run_name=row.run_name,
                    dataset_version_id=row.dataset_version_id,
                    model_name=row.model_name,
                    encoder_name=row.encoder_name,
                    trainer_backend=row.trainer_backend,
                    best_dice=self._best_dice_from_db(session, row.id),
                    status=row.status,
                )
                if summary is not None:
                    summaries.append(summary)
                    seen_run_names.add(summary.run_name)

        for run_dir in sorted(entry for entry in runs_dir.iterdir() if entry.is_dir()):
            if run_dir.name in seen_run_names:
                continue
            summary = self._summary_from_run_dir(run_dir=run_dir)
            if summary is not None:
                summaries.append(summary)
        return summaries

    def get_run_detail(self, run_name: str) -> RunDetail | None:
        dataset_label_lookup = self._dataset_label_lookup()
        for summary in self.list_runs():
            if summary.run_name != run_name:
                continue
            run_dir = Path(summary.run_dir)
            return RunDetail(
                summary=summary,
                dataset_label=dataset_label_lookup.get(
                    summary.dataset_version_id,
                    f"dataset {summary.dataset_version_id}",
                ),
                metrics=self._metrics(run_dir / "metrics.csv"),
                log_text=self._log_text(run_dir / "train.log"),
            )
        return None

    def retry_run(self, run_name: str) -> RunRecord | None:
        detail = self.get_run_detail(run_name)
        if detail is None:
            return None
        summary = detail.summary
        return self.start_run(
            dataset_version_id=summary.dataset_version_id,
            model_name=summary.model_name,
            encoder_name=summary.encoder_name,
            epochs=summary.epochs,
            batch_size=summary.batch_size,
            learning_rate=summary.learning_rate,
        )

    def stop_run(self, run_name: str) -> bool:
        detail = self.get_run_detail(run_name)
        if detail is None:
            return False
        run_dir = Path(detail.summary.run_dir)
        pid_path = run_dir / "run.pid"
        if not pid_path.is_file():
            return False
        try:
            pid = int(pid_path.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            (run_dir / "status.json").write_text(
                json.dumps({"status": "stopped"}, indent=2),
                encoding="utf-8",
            )
            with self._session_factory() as session:
                update_training_run_status(session, run_name=run_name, status="stopped")
                session.commit()
            return True
        except (ProcessLookupError, ValueError, OSError):
            return False

    def _summary_from_run_dir(
        self,
        run_dir: Path,
        run_name: str | None = None,
        dataset_version_id: int | None = None,
        model_name: str | None = None,
        encoder_name: str | None = None,
        trainer_backend: str | None = None,
        best_dice: float | None = None,
        status: str | None = None,
    ) -> RunSummary | None:
        config_path = run_dir / "config.json"
        if not config_path.is_file():
            return None

        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return RunSummary(
            run_name=run_name or run_dir.name,
            run_dir=str(run_dir),
            status=status or self._status(run_dir / "status.json"),
            dataset_version_id=int(
                dataset_version_id
                if dataset_version_id is not None
                else payload["dataset_version_id"]
            ),
            model_name=model_name or str(payload["model_name"]),
            encoder_name=encoder_name or str(payload["encoder_name"]),
            trainer_backend=trainer_backend or str(payload.get("trainer_backend", "simulated")),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            best_dice=(
                best_dice
                if best_dice is not None
                else self._best_dice(run_dir / "metrics.csv")
            ),
        )

    def _best_dice(self, metrics_path: Path) -> float | None:
        if not metrics_path.is_file():
            return None
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            dice_values = [
                float(row["dice"])
                for row in reader
                if row.get("dice") not in (None, "")
            ]
        if not dice_values:
            return None
        return max(dice_values)

    def _status(self, status_path: Path) -> str:
        if not status_path.is_file():
            return "queued"
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        return str(payload.get("status", "queued"))

    def _metrics(self, metrics_path: Path) -> list[RunMetricRecord]:
        if not metrics_path.is_file():
            return []
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [
                RunMetricRecord(epoch=int(row["epoch"]), dice=float(row["dice"]))
                for row in reader
                if row.get("epoch") not in (None, "") and row.get("dice") not in (None, "")
            ]

    def _log_text(self, log_path: Path) -> str:
        if not log_path.is_file():
            return ""
        return log_path.read_text(encoding="utf-8")

    def _best_dice_from_db(self, session, training_run_id: int) -> float | None:
        rows = (
            session.query(RunMetric.dice)
            .filter(RunMetric.training_run_id == training_run_id)
            .all()
        )
        if not rows:
            return None
        return max(float(row[0]) for row in rows)

    def _dataset_label_lookup(self) -> dict[int, str]:
        manifests_dir = self.project_root / "data" / "datasets" / "manifests"
        if not manifests_dir.is_dir():
            return {1: "baseline v001"}

        labels: dict[int, str] = {}
        for index, manifest_path in enumerate(sorted(manifests_dir.glob("*.json")), start=1):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels[index] = f'{payload.get("name", manifest_path.stem)} {payload.get("version", "")}'.strip()
        if not labels:
            labels[1] = "baseline v001"
        return labels
