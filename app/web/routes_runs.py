from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.runs import RunService
from trainer.model_registry import list_registered_models

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")
PAGE_SIZE = 10


def _page_window(items: list, page: int, page_size: int) -> tuple[list, int]:
    total_pages = max(1, ((len(items) - 1) // page_size) + 1) if items else 1
    page = min(max(page, 1), total_pages)
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total_pages


def _page_link(base_path: str, params: dict[str, str], page: int) -> str:
    query = dict(params)
    query["page"] = str(page)
    return f"{base_path}?{urlencode(query)}"


@router.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request) -> HTMLResponse:
    run_service = RunService(project_root=request.app.state.project_root)
    status_filter = request.query_params.get("status", "").strip()
    model_filter = request.query_params.get("model_name", "").strip()
    run_name_query = request.query_params.get("run_name_query", "").strip()
    page = int(request.query_params.get("page", "1") or "1")
    runs = run_service.list_runs()
    if status_filter:
        runs = [run for run in runs if run.status == status_filter]
    if model_filter:
        runs = [run for run in runs if run.model_name == model_filter]
    if run_name_query:
        lowered_query = run_name_query.lower()
        runs = [run for run in runs if lowered_query in run.run_name.lower()]
    current_page_runs, total_pages = _page_window(runs, page, PAGE_SIZE)
    pagination_params = {
        key: value
        for key, value in {
            "status": status_filter,
            "model_name": model_filter,
            "run_name_query": run_name_query,
        }.items()
        if value
    }
    return templates.TemplateResponse(
        request=request,
        name="runs.html",
        context={
            "page_title": "Runs",
            "runs": current_page_runs,
            "selected_status": status_filter,
            "selected_model_name": model_filter,
            "run_name_query": run_name_query,
            "statuses": ("queued", "running", "completed", "failed", "stopped"),
            "models": list_registered_models(),
            "page": min(max(page, 1), total_pages),
            "total_pages": total_pages,
            "prev_page_link": _page_link("/runs", pagination_params, page - 1) if page > 1 else None,
            "next_page_link": _page_link("/runs", pagination_params, page + 1) if page < total_pages else None,
        },
    )


@router.get("/runs/{run_name}", response_class=HTMLResponse)
def run_detail_page(request: Request, run_name: str) -> HTMLResponse:
    run_service = RunService(project_root=request.app.state.project_root)
    detail = run_service.get_run_detail(run_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "page_title": detail.summary.run_name,
            "detail": detail,
        },
    )


@router.post("/runs/{run_name}/retry", response_class=HTMLResponse)
def retry_run_page(request: Request, run_name: str) -> HTMLResponse:
    run_service = RunService(project_root=request.app.state.project_root)
    retried_run = run_service.retry_run(run_name)
    if retried_run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    detail = run_service.get_run_detail(run_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "page_title": detail.summary.run_name,
            "detail": detail,
            "created_message": f"Queued retry run {retried_run.run_name} from {run_name}.",
        },
    )


@router.post("/runs/{run_name}/stop", response_class=HTMLResponse)
def stop_run_page(request: Request, run_name: str) -> HTMLResponse:
    run_service = RunService(project_root=request.app.state.project_root)
    stopped = run_service.stop_run(run_name)
    if not stopped:
        raise HTTPException(status_code=404, detail="Run not found")
    detail = run_service.get_run_detail(run_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        request=request,
        name="run_detail.html",
        context={
            "page_title": detail.summary.run_name,
            "detail": detail,
            "created_message": f"Stopped run {run_name}.",
        },
    )


@router.get("/runs/{run_name}/artifacts/{artifact_name}")
def run_artifact(request: Request, run_name: str, artifact_name: str) -> FileResponse:
    detail = RunService(project_root=request.app.state.project_root).get_run_detail(run_name)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")

    run_dir = Path(detail.summary.run_dir)
    artifact_map = {
        "config": run_dir / "config.json",
        "metrics": run_dir / "metrics.csv",
        "log": run_dir / "train.log",
        "checkpoint": run_dir / "checkpoints" / "best.ckpt",
    }
    artifact_path = artifact_map.get(artifact_name)
    if artifact_path is None or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=artifact_path, filename=artifact_path.name)
