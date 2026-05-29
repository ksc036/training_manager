from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from frontend.app_shell import build_shell_html


router = APIRouter()
_STATIC_ROOT = Path(__file__).resolve().parent / "static" / "app"
_INDEX_PATH = _STATIC_ROOT / "index.html"


def _fallback_index() -> str:
    return build_shell_html()


def _render_index() -> HTMLResponse:
    if _INDEX_PATH.is_file():
        return HTMLResponse(_INDEX_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(_fallback_index())


@router.get("/app", response_class=HTMLResponse)
def dashboard_app() -> HTMLResponse:
    return _render_index()


@router.get("/app/{path:path}", response_class=HTMLResponse)
def dashboard_app_nested(path: str) -> HTMLResponse:
    if path.startswith("api/"):
        return HTMLResponse(status_code=404, content="Not Found")
    return _render_index()
