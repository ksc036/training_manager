from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.config import PROJECT_ROOT
from app.services.test_inference import TestInferenceService

router = APIRouter()


def _project_root(request: Request) -> Path:
    return getattr(request.app.state, "project_root", PROJECT_ROOT)


def _service(request: Request) -> TestInferenceService:
    return TestInferenceService(_project_root(request))


@router.get("/api/test/runs")
def list_testable_runs(request: Request) -> list[dict[str, object]]:
    return [
        {
            "run_name": run.run_name,
            "run_dir": run.run_dir,
            "model_name": run.model_name,
            "encoder_name": run.encoder_name,
            "dataset_version_id": run.dataset_version_id,
        }
        for run in _service(request).list_testable_runs()
    ]


@router.get("/api/test/runs/{run_name}/samples")
def list_test_samples(run_name: str, request: Request) -> list[dict[str, object]]:
    try:
        samples = _service(request).list_samples_for_run(run_name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return [
        {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
        }
        for sample in samples
    ]


@router.post("/api/test/infer")
async def infer_sample(request: Request) -> dict[str, object]:
    payload = await request.json()
    run_name = str(payload.get("run_name", "")).strip()
    sample_id = str(payload.get("sample_id", "")).strip()
    try:
        result = _service(request).run_inference(run_name, sample_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "run_name": result.run_name,
        "model_name": result.model_name,
        "encoder_name": result.encoder_name,
        "dataset_label": result.dataset_label,
        "checkpoint_path": result.checkpoint_path,
        "sample_id": result.sample_id,
        "original_data_url": result.original_data_url,
        "ground_truth_data_url": result.ground_truth_data_url,
        "prediction_data_url": result.prediction_data_url,
        "overlay_data_url": result.overlay_data_url,
    }
