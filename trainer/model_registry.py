from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RegisteredModel:
    key: str
    display_name: str
    default_encoder: str
    trainer_backend: str
    backend_config: dict[str, str] = field(default_factory=dict)

def build_registered_models(
    external_trainer_command: str | None = None,
) -> tuple[RegisteredModel, ...]:
    resolved_external_command = (
        external_trainer_command
        or os.environ.get("TRAINING_MANAGER_EXTERNAL_TRAINER_COMMAND")
        or f"{sys.executable} -m trainer.external_adapter"
    )
    external_backend_config = {"entry_command": resolved_external_command}
    models = [
        RegisteredModel(
            key="unet",
            display_name="U-Net",
            default_encoder="resnet34",
            trainer_backend="external-script",
            backend_config=external_backend_config,
        ),
        RegisteredModel(
            key="unetplusplus",
            display_name="U-Net++",
            default_encoder="resnet34",
            trainer_backend="external-script",
            backend_config=external_backend_config,
        ),
        RegisteredModel(
            key="deeplabv3",
            display_name="DeepLabV3",
            default_encoder="resnet50",
            trainer_backend="external-script",
            backend_config=external_backend_config,
        ),
    ]
    models.append(
        RegisteredModel(
            key="external-script-model",
            display_name="External Script Model",
            default_encoder="custom",
            trainer_backend="external-script",
            backend_config={"entry_command": resolved_external_command},
        )
    )
    return tuple(models)


def list_registered_models() -> tuple[RegisteredModel, ...]:
    return build_registered_models()


def get_registered_model(model_key: str) -> RegisteredModel | None:
    for model in list_registered_models():
        if model.key == model_key:
            return model
    return None
