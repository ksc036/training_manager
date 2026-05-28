from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.test_inference import TestInferenceService


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def _test_context(
    request: Request,
    *,
    selected_run_name: str = "",
    selected_sample_id: str = "",
    error_message: str | None = None,
    result=None,
) -> dict:
    service = TestInferenceService(project_root=request.app.state.project_root)
    runs = service.list_testable_runs()
    samples = service.list_samples_for_run(selected_run_name) if selected_run_name else []
    return {
        "page_title": "Test",
        "runs": runs,
        "samples": samples,
        "selected_run_name": selected_run_name,
        "selected_sample_id": selected_sample_id,
        "error_message": error_message,
        "result": result,
    }


@router.get("/test", response_class=HTMLResponse)
def render_test_page(request: Request) -> HTMLResponse:
    selected_run_name = request.query_params.get("run_name", "").strip()
    context = _test_context(request, selected_run_name=selected_run_name)
    return templates.TemplateResponse(
        request=request,
        name="test.html",
        context=context,
    )


@router.post("/test", response_class=HTMLResponse)
async def run_test_inference(request: Request) -> HTMLResponse:
    raw_body = (await request.body()).decode("utf-8")
    form_data = parse_qs(raw_body, keep_blank_values=True)
    run_name = form_data.get("run_name", [""])[0].strip()
    sample_id = form_data.get("sample_id", [""])[0].strip()
    service = TestInferenceService(project_root=request.app.state.project_root)
    runs = service.list_testable_runs()
    selected_run = next((run for run in runs if run.run_name == run_name), None)
    if selected_run is None:
        return templates.TemplateResponse(
            request=request,
            name="test.html",
            context=_test_context(
                request,
                error_message="Choose a valid completed run.",
            ),
            status_code=400,
        )
    try:
        result = service.run_inference(run_name, sample_id)
    except ValueError as exc:
        return templates.TemplateResponse(
            request=request,
            name="test.html",
            context=_test_context(
                request,
                selected_run_name=run_name,
                selected_sample_id=sample_id,
                error_message=str(exc),
            ),
            status_code=400,
        )
    return templates.TemplateResponse(
        request=request,
        name="test.html",
        context=_test_context(
            request,
            selected_run_name=run_name,
            selected_sample_id=sample_id,
            result=result,
        ),
    )
