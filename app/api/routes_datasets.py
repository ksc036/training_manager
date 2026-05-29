from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from app.config import PROJECT_ROOT
from app.services.datasets import DatasetRegistryService

router = APIRouter()


def _project_root(request: Request) -> Path:
    return getattr(request.app.state, "project_root", PROJECT_ROOT)


def _approved_source_root(request: Request) -> Path:
    return getattr(request.app.state, "approved_source_root", PROJECT_ROOT / "annotation_complete")


@router.get("/api/datasets/samples")
def list_approved_samples(
    request: Request,
    query: str = "",
    resolution_bucket: str = "",
    width_min: Optional[int] = None,
    width_max: Optional[int] = None,
    height_min: Optional[int] = None,
    height_max: Optional[int] = None,
    status: str = "",
    sort: str = "sample_id",
    page: int = 1,
    page_size: int = 25,
) -> dict[str, object]:
    service = DatasetRegistryService(_project_root(request))
    page_result = service.list_approved_samples(
        _approved_source_root(request),
        query=query,
        resolution_bucket=resolution_bucket,
        width_min=width_min,
        width_max=width_max,
        height_min=height_min,
        height_max=height_max,
        status=status,
        sort=sort,
        page=page,
        page_size=page_size,
    )
    payload = page_result.to_payload()
    payload["approved_source_root"] = str(_approved_source_root(request))
    return payload


@router.get("/api/datasets/versions")
def list_dataset_versions(request: Request) -> list[dict[str, object]]:
    service = DatasetRegistryService(_project_root(request))
    return [
        {
            "id": dataset.id,
            "name": dataset.name,
            "version": dataset.version,
            "manifest_path": dataset.manifest_path,
            "sample_count": dataset.sample_count,
            "label": f"{dataset.name} {dataset.version}".strip(),
        }
        for dataset in service.list_dataset_versions()
    ]


@router.post("/api/datasets/versions")
async def create_dataset_version(request: Request) -> dict[str, object]:
    payload = await request.json()
    name = str(payload.get("name", "")).strip()
    sample_ids = {
        str(sample_id).strip()
        for sample_id in payload.get("sample_ids", [])
        if str(sample_id).strip()
    }
    service = DatasetRegistryService(_project_root(request))
    selected_samples = service.approved_records_by_ids(_approved_source_root(request), sample_ids)
    if not selected_samples:
        raise HTTPException(status_code=400, detail="Select at least one valid approved sample.")
    try:
        record = service.create_dataset_version(name=name, selected_samples=selected_samples)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {
        "id": record.id,
        "name": record.name,
        "version": record.version,
        "sample_count": record.sample_count,
        "manifest_path": record.manifest_path,
        "label": f"{record.name} {record.version}".strip(),
    }


@router.get("/api/datasets/{dataset_version_id}/samples")
def list_dataset_samples(
    dataset_version_id: int,
    request: Request,
    sample_id: str = "",
    page: int = 1,
    page_size: int = 20,
) -> dict[str, object]:
    service = DatasetRegistryService(_project_root(request))
    page_result = service.list_dataset_samples(
        dataset_version_id,
        sample_id_query=sample_id,
        page=page,
        page_size=page_size,
    )
    if page_result is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return page_result.to_payload()
