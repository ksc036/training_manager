from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.config import PROJECT_ROOT
from app.services.runs import RunService
from trainer.model_registry import get_registered_model

router = APIRouter()


def _project_root(request: Request) -> Path:
    return getattr(request.app.state, "project_root", PROJECT_ROOT)


def _run_service(request: Request) -> RunService:
    return RunService(_project_root(request))


@router.get("/api/runs")
def list_runs(
    request: Request,
    status: str = "",
    model_name: str = "",
    query: str = "",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in _run_service(request).list_runs():
        if status and run.status != status:
            continue
        if model_name and run.model_name != model_name:
            continue
        if query and query.lower() not in run.run_name.lower():
            continue
        rows.append(
            {
                "run_name": run.run_name,
                "run_dir": run.run_dir,
                "status": run.status,
                "dataset_version_id": run.dataset_version_id,
                "model_name": run.model_name,
                "encoder_name": run.encoder_name,
                "trainer_backend": run.trainer_backend,
                "epochs": run.epochs,
                "batch_size": run.batch_size,
                "learning_rate": run.learning_rate,
                "best_dice": run.best_dice,
            }
        )
    return rows


@router.post("/api/runs")
async def create_run(request: Request) -> dict[str, object]:
    payload = await request.json()
    dataset_version_id = int(payload.get("dataset_version_id", 0))
    model_name = str(payload.get("model_name", "")).strip()
    epochs = int(payload.get("epochs", 1))
    batch_size = int(payload.get("batch_size", 1))
    learning_rate = float(payload.get("learning_rate", 0.001))
    registered_model = get_registered_model(model_name)
    if registered_model is None:
        raise HTTPException(status_code=400, detail="Choose a valid registered model.")
    run = _run_service(request).start_run(
        dataset_version_id=dataset_version_id,
        model_name=model_name,
        encoder_name=registered_model.default_encoder,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    return {
        "id": run.id,
        "run_name": run.run_name,
        "run_dir": run.run_dir,
        "status": run.status,
        "dataset_version_id": run.dataset_version_id,
        "model_name": run.model_name,
    }


@router.get("/api/runs/{run_name}")
def get_run_detail(run_name: str, request: Request) -> dict[str, object]:
    detail = _run_service(request).get_run_detail(run_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "summary": {
            "run_name": detail.summary.run_name,
            "run_dir": detail.summary.run_dir,
            "status": detail.summary.status,
            "dataset_version_id": detail.summary.dataset_version_id,
            "model_name": detail.summary.model_name,
            "encoder_name": detail.summary.encoder_name,
            "trainer_backend": detail.summary.trainer_backend,
            "epochs": detail.summary.epochs,
            "batch_size": detail.summary.batch_size,
            "learning_rate": detail.summary.learning_rate,
            "best_dice": detail.summary.best_dice,
        },
        "dataset_label": detail.dataset_label,
        "metrics": [
            {"epoch": metric.epoch, "dice": metric.dice}
            for metric in detail.metrics
        ],
        "log_text": detail.log_text,
    }


@router.post("/api/runs/{run_name}/retry")
def retry_run(run_name: str, request: Request) -> dict[str, object]:
    run = _run_service(request).retry_run(run_name)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": run.id,
        "run_name": run.run_name,
        "status": run.status,
        "dataset_version_id": run.dataset_version_id,
        "model_name": run.model_name,
    }


@router.post("/api/runs/{run_name}/stop")
def stop_run(run_name: str, request: Request) -> dict[str, object]:
    stopped = _run_service(request).stop_run(run_name)
    if not stopped:
        raise HTTPException(status_code=400, detail="Run could not be stopped")
    return {"run_name": run_name, "status": "stopped"}
