from trainer.model_registry import build_registered_models, list_registered_models


def test_model_registry_exposes_supported_models() -> None:
    models = list_registered_models()

    assert [model.key for model in models] == [
        "unet",
        "unetplusplus",
        "deeplabv3",
        "external-script-model",
    ]
    assert models[0].display_name == "U-Net"
    assert models[0].default_encoder == "resnet34"
    assert models[0].trainer_backend == "external-script"
    assert "trainer.external_adapter" in models[0].backend_config["entry_command"]
    assert models[1].trainer_backend == "external-script"
    assert models[2].trainer_backend == "external-script"
    assert models[-1].trainer_backend == "external-script"
    assert "trainer.external_adapter" in models[-1].backend_config["entry_command"]


def test_build_registered_models_adds_external_script_model_when_configured() -> None:
    models = build_registered_models("python -m custom_trainer")

    assert [model.key for model in models][-1] == "external-script-model"
    assert models[-1].trainer_backend == "external-script"
    assert models[-1].backend_config == {"entry_command": "python -m custom_trainer"}
