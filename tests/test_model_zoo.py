import torch

from trainer.model_zoo import build_segmentation_model


def test_supported_models_produce_single_channel_logits() -> None:
    inputs = torch.rand(1, 3, 64, 64)

    for model_name in ["unet", "unetplusplus", "deeplabv3", "external-script-model"]:
        model = build_segmentation_model(model_name)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
        assert tuple(outputs.shape) == (1, 1, 64, 64)
