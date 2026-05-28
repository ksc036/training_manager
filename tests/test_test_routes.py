import json

import torch
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.services.datasets import DatasetRegistryService
from trainer.scanner import SampleRecord


def _seed_test_run(tmp_path):
    sample_a_path = tmp_path / "annotation_complete" / "sample_a"
    sample_b_path = tmp_path / "annotation_complete" / "sample_b"
    for sample_path, color in ((sample_a_path, 255), (sample_b_path, 0)):
        image_path = sample_path / "Image" / f"{sample_path.name}.png"
        mask_path = sample_path / "mask" / f"{sample_path.name}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), color=(255, 0, 0)).save(image_path)
        Image.new("L", (16, 16), color=color).save(mask_path)

    dataset = DatasetRegistryService(project_root=tmp_path).create_dataset_version(
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


def test_test_page_renders_navigation_and_title(tmp_path) -> None:
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/test")

    assert response.status_code == 200
    assert "Test" in response.text
    assert 'name="run_name"' in response.text


def test_test_page_limits_samples_to_selected_run_dataset(tmp_path) -> None:
    _seed_test_run(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.get("/test?run_name=run_real")

    assert response.status_code == 200
    assert "sample_a" in response.text
    assert "sample_b" not in response.text


def test_test_page_runs_inference_and_renders_four_panels(tmp_path) -> None:
    _seed_test_run(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post("/test", data={"run_name": "run_real", "sample_id": "sample_a"})

    assert response.status_code == 200
    assert "Original" in response.text
    assert "Ground Truth" in response.text
    assert "Prediction" in response.text
    assert "Overlay" in response.text
    assert "data:image/png;base64," in response.text


def test_test_page_rejects_invalid_run(tmp_path) -> None:
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post("/test", data={"run_name": "missing", "sample_id": "sample_a"})

    assert response.status_code == 400
    assert "Choose a valid completed run." in response.text


def test_test_page_rejects_invalid_sample(tmp_path) -> None:
    _seed_test_run(tmp_path)
    client = TestClient(create_app(project_root=tmp_path))

    response = client.post("/test", data={"run_name": "run_real", "sample_id": "sample_b"})

    assert response.status_code == 400
    assert "Choose a valid sample from the selected run dataset." in response.text
