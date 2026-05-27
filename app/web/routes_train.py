from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService
from trainer.model_registry import list_registered_models


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def _train_context(request: Request, created_message: str | None = None) -> dict:
    dataset_service = DatasetRegistryService(project_root=request.app.state.project_root)
    return {
        "page_title": "Train",
        "models": list_registered_models(),
        "dataset_versions": dataset_service.list_dataset_versions(),
        "created_message": created_message,
    }


@router.get("/train", response_class=HTMLResponse)
def train_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context=_train_context(request),
    )


@router.post("/train", response_class=HTMLResponse)
async def start_train_run(request: Request) -> HTMLResponse:
    raw_body = (await request.body()).decode("utf-8")
    form_data = parse_qs(raw_body, keep_blank_values=True)
    dataset_version_id = int(form_data.get("dataset_version_id", ["0"])[0])
    model_name = form_data.get("model_name", [""])[0]
    epochs = int(form_data.get("epochs", ["1"])[0])
    batch_size = int(form_data.get("batch_size", ["1"])[0])
    learning_rate = float(form_data.get("learning_rate", ["0.001"])[0])

    context = _train_context(request)
    dataset_versions = context["dataset_versions"]
    valid_dataset_ids = {dataset.id for dataset in dataset_versions}
    registered_models = {model.key: model for model in context["models"]}

    if dataset_version_id not in valid_dataset_ids:
        context["error_message"] = "Choose a valid dataset version before starting a run."
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context=context,
            status_code=400,
        )
    if model_name not in registered_models:
        context["error_message"] = "Choose a valid registered model before starting a run."
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context=context,
            status_code=400,
        )
    encoder_name = registered_models[model_name].default_encoder

    run_service = RunService(project_root=request.app.state.project_root)
    run = run_service.start_run(
        dataset_version_id=dataset_version_id,
        model_name=model_name,
        encoder_name=encoder_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    message = f"Queued run {run.run_name} for model {run.model_name}."
    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context=_train_context(request, created_message=message),
    )
