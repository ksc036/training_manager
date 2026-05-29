from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request

from app.config import PROJECT_ROOT
from app.services.compare import list_ranked_runs

router = APIRouter()


def _project_root(request: Request) -> Path:
    return getattr(request.app.state, "project_root", PROJECT_ROOT)


@router.get("/api/compare")
def compare_runs(
    request: Request,
    model_name: str = "",
    query: str = "",
    sort: str = "best_dice_desc",
) -> list[dict[str, object]]:
    return list_ranked_runs(
        _project_root(request),
        model_name=model_name or None,
        run_name_query=query or None,
        sort=sort,
    )
