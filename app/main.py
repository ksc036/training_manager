from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.bootstrap import ensure_runtime_dirs
from app.config import APPROVED_SOURCE_ROOT, PROJECT_ROOT
from app.db import init_database
from app.api.routes_compare import router as compare_api_router
from app.api.routes_datasets import router as datasets_api_router
from app.api.routes_models import router as models_api_router
from app.api.routes_runs import router as runs_api_router
from app.api.routes_test import router as test_api_router
from app.web.routes_compare import router as compare_router
from app.web.routes_dashboard import router as dashboard_router
from app.web.routes_datasets import router as datasets_router
from app.web.routes_runs import router as runs_router
from app.web.routes_test import router as test_router
from app.web.routes_train import router as train_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    ensure_runtime_dirs(app.state.project_root)
    init_database(app.state.project_root)
    yield


def create_app(
    project_root: Path = PROJECT_ROOT,
    approved_source_root: Path = APPROVED_SOURCE_ROOT,
) -> FastAPI:
    app = FastAPI(title="Training Manager", lifespan=lifespan)
    app.state.project_root = project_root
    app.state.approved_source_root = approved_source_root
    app.include_router(datasets_api_router)
    app.include_router(models_api_router)
    app.include_router(runs_api_router)
    app.include_router(compare_api_router)
    app.include_router(test_api_router)
    app.include_router(dashboard_router)
    app.include_router(datasets_router)
    app.include_router(train_router)
    app.include_router(runs_router)
    app.include_router(compare_router)
    app.include_router(test_router)
    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "web" / "static"), name="static")

    return app


app = create_app()
