from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.datasets import DatasetRegistryService
from trainer.scanner import scan_approved_samples

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def _datasets_context(request: Request, created_message: str | None = None) -> dict:
    scan_result = scan_approved_samples(request.app.state.approved_source_root)
    return {
        "page_title": "Datasets",
        "samples": scan_result.samples,
        "issues": scan_result.issues,
        "created_message": created_message,
    }


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context=_datasets_context(request),
    )


@router.post("/datasets", response_class=HTMLResponse)
async def create_dataset_version(
    request: Request,
) -> HTMLResponse:
    raw_body = (await request.body()).decode("utf-8")
    form_data = parse_qs(raw_body, keep_blank_values=True)
    name = form_data.get("name", [""])[0].strip()
    selected_ids = set(form_data.get("sample_id", []))
    scan_result = scan_approved_samples(request.app.state.approved_source_root)
    selected_samples = [
        sample for sample in scan_result.samples if sample.sample_id in selected_ids
    ]
    if not selected_samples:
        context = _datasets_context(request)
        context["error_message"] = "Select at least one sample before creating a dataset."
        return templates.TemplateResponse(
            request=request,
            name="datasets.html",
            context=context,
            status_code=400,
        )

    service = DatasetRegistryService(project_root=request.app.state.project_root)
    record = service.create_dataset_version(name=name, selected_samples=selected_samples)
    message = (
        f"Created dataset {record.name} {record.version} with "
        f"{record.sample_count} sample(s)."
    )
    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context=_datasets_context(request, created_message=message),
    )
