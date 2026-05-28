from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService
from trainer.device import resolve_torch_device
from trainer.model_zoo import build_segmentation_model


@dataclass(frozen=True)
class TestableRunSummary:
    __test__ = False

    run_name: str
    run_dir: str
    model_name: str
    encoder_name: str
    dataset_version_id: int


@dataclass(frozen=True)
class TestSampleSummary:
    __test__ = False

    sample_id: str
    image_path: str
    mask_path: str


@dataclass(frozen=True)
class InferencePreviewResult:
    __test__ = False

    run_name: str
    model_name: str
    encoder_name: str
    dataset_label: str
    checkpoint_path: str
    sample_id: str
    original_data_url: str
    ground_truth_data_url: str
    prediction_data_url: str
    overlay_data_url: str


def _to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class TestInferenceService:
    __test__ = False

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._run_service = RunService(project_root)
        self._dataset_service = DatasetRegistryService(project_root)

    def list_testable_runs(self) -> list[TestableRunSummary]:
        runs: list[TestableRunSummary] = []
        for run in self._run_service.list_runs():
            checkpoint_path = Path(run.run_dir) / "checkpoints" / "best.ckpt"
            if run.status != "completed":
                continue
            if run.trainer_backend != "external-script":
                continue
            if not checkpoint_path.is_file():
                continue
            runs.append(
                TestableRunSummary(
                    run_name=run.run_name,
                    run_dir=run.run_dir,
                    model_name=run.model_name,
                    encoder_name=run.encoder_name,
                    dataset_version_id=run.dataset_version_id,
                )
            )
        return runs

    def list_samples_for_run(self, run_name: str) -> list[TestSampleSummary]:
        run = self._require_testable_run(run_name)
        return self._samples_for_dataset_version(run.dataset_version_id)

    def run_inference(self, run_name: str, sample_id: str) -> InferencePreviewResult:
        run = self._require_testable_run(run_name)
        sample = self._require_sample(run.dataset_version_id, sample_id)
        checkpoint_path = Path(run.run_dir) / "checkpoints" / "best.ckpt"
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        device = resolve_torch_device()
        model = build_segmentation_model(run.model_name).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        original = Image.open(sample.image_path).convert("RGB")
        ground_truth = Image.open(sample.mask_path).convert("L")
        resized = original.resize((64, 64))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            prediction = (torch.sigmoid(logits) > 0.5).float()[0, 0].cpu().numpy()

        prediction_image = Image.fromarray((prediction * 255).astype("uint8")).resize(
            original.size
        )
        ground_truth_display = ground_truth.resize(original.size)
        overlay = original.copy()
        overlay_pixels = overlay.load()
        prediction_pixels = prediction_image.load()
        for y in range(original.size[1]):
            for x in range(original.size[0]):
                if prediction_pixels[x, y] > 127:
                    red, green, blue = overlay_pixels[x, y]
                    overlay_pixels[x, y] = (255, green // 2, blue // 2)

        dataset = self._dataset_service.get_dataset_version(run.dataset_version_id)
        dataset_label = "unknown"
        if dataset is not None:
            dataset_label = f"{dataset.name} {dataset.version}".strip()

        return InferencePreviewResult(
            run_name=run.run_name,
            model_name=run.model_name,
            encoder_name=run.encoder_name,
            dataset_label=dataset_label,
            checkpoint_path=str(checkpoint_path),
            sample_id=sample.sample_id,
            original_data_url=_to_data_url(original),
            ground_truth_data_url=_to_data_url(ground_truth_display.convert("RGB")),
            prediction_data_url=_to_data_url(prediction_image.convert("RGB")),
            overlay_data_url=_to_data_url(overlay),
        )

    def _require_testable_run(self, run_name: str) -> TestableRunSummary:
        for run in self.list_testable_runs():
            if run.run_name == run_name:
                return run
        raise ValueError("Choose a valid completed run with a real checkpoint.")

    def _require_sample(self, dataset_version_id: int, sample_id: str) -> TestSampleSummary:
        for sample in self._samples_for_dataset_version(dataset_version_id):
            if sample.sample_id == sample_id:
                return sample
        raise ValueError("Choose a valid sample from the selected run dataset.")

    def _samples_for_dataset_version(self, dataset_version_id: int) -> list[TestSampleSummary]:
        dataset = self._dataset_service.get_dataset_version(dataset_version_id)
        if dataset is None:
            return []
        payload = json.loads(Path(dataset.manifest_path).read_text(encoding="utf-8"))
        return [
            TestSampleSummary(
                sample_id=str(sample["sample_id"]),
                image_path=str(sample["image_path"]),
                mask_path=str(sample["mask_path"]),
            )
            for sample in payload.get("samples", [])
            if sample.get("sample_id") and sample.get("image_path") and sample.get("mask_path")
        ]
