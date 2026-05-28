import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from app.services.datasets import DatasetRegistryService
from app.services import test_inference as test_inference_module
from app.services.test_inference import TestInferenceService
from trainer.scanner import SampleRecord


def test_list_testable_runs_returns_completed_real_checkpoint_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "data" / "runs" / "run_real"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        (
            '{"dataset_version_id": 1, "model_name": "unet", "encoder_name": "resnet34", '
            '"trainer_backend": "external-script", "epochs": 5, "batch_size": 2, "learning_rate": 0.001}'
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    (checkpoint_dir / "best.ckpt").write_bytes(b"checkpoint")

    service = TestInferenceService(project_root=tmp_path)

    runs = service.list_testable_runs()

    assert [run.run_name for run in runs] == ["run_real"]


def test_run_inference_returns_four_preview_images(tmp_path: Path) -> None:
    sample_path = tmp_path / "annotation_complete" / "sample_a"
    image_path = sample_path / "Image" / "sample_a.png"
    mask_path = sample_path / "mask" / "sample_a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    Image.new("L", (16, 16), color=255).save(mask_path)

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=[
            SampleRecord(
                sample_id="sample_a",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        ],
    )

    run_dir = tmp_path / "data" / "runs" / "run_real"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_version_id": dataset.id,
                "model_name": "external-script-model",
                "encoder_name": "custom",
                "trainer_backend": "external-script",
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    torch.save(
        {
            "epoch": 1,
            "best_dice": 0.5,
            "model_name": "external-script-model",
            "encoder_name": "custom",
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint_dir / "best.ckpt",
    )

    service = TestInferenceService(project_root=tmp_path)

    result = service.run_inference("run_real", "sample_a")

    assert result.sample_id == "sample_a"
    assert result.original_data_url.startswith("data:image/png;base64,")
    assert result.ground_truth_data_url.startswith("data:image/png;base64,")
    assert result.prediction_data_url.startswith("data:image/png;base64,")
    assert result.overlay_data_url.startswith("data:image/png;base64,")


def test_list_samples_for_run_is_limited_to_run_dataset(tmp_path: Path) -> None:
    sample_a_path = tmp_path / "annotation_complete" / "sample_a"
    sample_b_path = tmp_path / "annotation_complete" / "sample_b"
    for sample_path, color in ((sample_a_path, 255), (sample_b_path, 0)):
        image_path = sample_path / "Image" / f"{sample_path.name}.png"
        mask_path = sample_path / "mask" / f"{sample_path.name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
        Image.new("L", (16, 16), color=color).save(mask_path)

    dataset_a = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=[
            SampleRecord(
                sample_id="sample_a",
                sample_path=sample_a_path,
                image_path=sample_a_path / "Image" / "sample_a.png",
                mask_path=sample_a_path / "mask" / "sample_a.png",
            )
        ],
    )
    DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=[
            SampleRecord(
                sample_id="sample_b",
                sample_path=sample_b_path,
                image_path=sample_b_path / "Image" / "sample_b.png",
                mask_path=sample_b_path / "mask" / "sample_b.png",
            )
        ],
    )

    run_dir = tmp_path / "data" / "runs" / "run_real"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_version_id": dataset_a.id,
                "model_name": "external-script-model",
                "encoder_name": "custom",
                "trainer_backend": "external-script",
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    torch.save(
        {
            "epoch": 1,
            "best_dice": 0.5,
            "model_name": "external-script-model",
            "encoder_name": "custom",
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint_dir / "best.ckpt",
    )

    service = TestInferenceService(project_root=tmp_path)

    samples = service.list_samples_for_run("run_real")

    assert [sample.sample_id for sample in samples] == ["sample_a"]


def test_run_inference_uses_shared_device_helper(tmp_path: Path, monkeypatch) -> None:
    sample_path = tmp_path / "annotation_complete" / "sample_a"
    image_path = sample_path / "Image" / "sample_a.png"
    mask_path = sample_path / "mask" / "sample_a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
    Image.new("L", (16, 16), color=255).save(mask_path)

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
        name="baseline",
        selected_samples=[
            SampleRecord(
                sample_id="sample_a",
                sample_path=sample_path,
                image_path=image_path,
                mask_path=mask_path,
            )
        ],
    )

    run_dir = tmp_path / "data" / "runs" / "run_real"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "dataset_version_id": dataset.id,
                "model_name": "external-script-model",
                "encoder_name": "custom",
                "trainer_backend": "external-script",
                "epochs": 5,
                "batch_size": 2,
                "learning_rate": 0.001,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    torch.save(
        {
            "epoch": 1,
            "best_dice": 0.5,
            "model_name": "external-script-model",
            "encoder_name": "custom",
            "model_state_dict": {},
            "optimizer_state_dict": {},
        },
        checkpoint_dir / "best.ckpt",
    )

    class TrackingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.loaded_device: str | None = None
            self.forward_device: str | None = None

        def load_state_dict(self, state_dict, strict=True):  # type: ignore[override]
            del strict
            return state_dict

        def to(self, device):  # type: ignore[override]
            self.loaded_device = str(device)
            return self

        def eval(self):
            return self

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            self.forward_device = tensor.device.type
            return torch.zeros((tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]), device=tensor.device)

    tracking_model = TrackingModel()
    monkeypatch.setattr(
        test_inference_module,
        "build_segmentation_model",
        lambda model_name: tracking_model,
    )
    monkeypatch.setattr(
        test_inference_module,
        "resolve_torch_device",
        lambda: torch.device("cpu"),
    )

    service = TestInferenceService(project_root=tmp_path)

    result = service.run_inference("run_real", "sample_a")

    assert result.sample_id == "sample_a"
    assert tracking_model.loaded_device == "cpu"
    assert tracking_model.forward_device == "cpu"
