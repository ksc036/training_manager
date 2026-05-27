from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.compare import list_ranked_runs
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


@router.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request) -> HTMLResponse:
    model_filter = request.query_params.get("model_name", "").strip()
    run_name_query = request.query_params.get("run_name_query", "").strip()
    sort = request.query_params.get("sort", "best_dice_desc").strip() or "best_dice_desc"
    page = int(request.query_params.get("page", "1") or "1")
    runs = list_ranked_runs(
        request.app.state.project_root,
        model_name=model_filter or None,
        run_name_query=run_name_query or None,
        sort=sort,
    )
    current_page_runs, total_pages = _page_window(runs, page, PAGE_SIZE)
    pagination_params = {
        key: value
        for key, value in {
            "model_name": model_filter,
            "run_name_query": run_name_query,
            "sort": sort if sort != "best_dice_desc" else "",
        }.items()
        if value
    }
    return templates.TemplateResponse(
        request=request,
        name="compare.html",
        context={
            "page_title": "Compare",
            "runs": current_page_runs,
            "selected_model_name": model_filter,
            "run_name_query": run_name_query,
            "selected_sort": sort,
            "models": list_registered_models(),
            "page": min(max(page, 1), total_pages),
            "total_pages": total_pages,
            "prev_page_link": _page_link("/compare", pagination_params, page - 1) if page > 1 else None,
            "next_page_link": _page_link("/compare", pagination_params, page + 1) if page < total_pages else None,
        },
    )
