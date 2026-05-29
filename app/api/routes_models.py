from __future__ import annotations

from fastapi import APIRouter

from trainer.model_registry import list_registered_models

router = APIRouter()


@router.get("/api/models")
def list_models() -> list[dict[str, object]]:
    return [
        {
            "key": model.key,
            "display_name": model.display_name,
            "default_encoder": model.default_encoder,
            "trainer_backend": model.trainer_backend,
        }
        for model in list_registered_models()
    ]
