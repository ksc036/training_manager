import json
from pathlib import Path

import pytest

from app.services.datasets import DatasetRegistryService
from trainer.scanner import SampleRecord


def _make_sample(tmp_path: Path) -> SampleRecord:
    sample_path = tmp_path / "annotation_complete" / "sample_a"
    image_path = sample_path / "Image" / "sample_a.png"
    mask_path = sample_path / "mask" / "sample_a.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"image-bytes")
    mask_path.write_bytes(b"mask-bytes")
    return SampleRecord(
        sample_id="sample_a",
        sample_path=sample_path,
        image_path=image_path,
        mask_path=mask_path,
    )


def test_create_dataset_version_writes_manifest_and_returns_record(
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path)
    service = DatasetRegistryService(project_root=tmp_path)

    record = service.create_dataset_version(name="baseline", selected_samples=[sample])

    assert record.id == 1
    assert record.name == "baseline"
    assert record.version == "v001"
    assert record.sample_count == 1
    manifest_path = Path(record.manifest_path)

    assert manifest_path == tmp_path / "data" / "datasets" / "manifests" / "baseline_v001.json"
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "name": "baseline",
        "version": "v001",
        "samples": [
            {
                "sample_id": "sample_a",
                "sample_path": str(sample.sample_path),
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            }
        ],
    }
    assert service.list_dataset_versions() == [
        type(service.list_dataset_versions()[0])(
            id=1,
            name="baseline",
            version="v001",
            manifest_path=str(manifest_path),
            sample_count=1,
        )
    ]


def test_create_dataset_version_increments_version_for_repeated_name(
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path)
    service = DatasetRegistryService(project_root=tmp_path)

    first_record = service.create_dataset_version(name="baseline", selected_samples=[sample])
    second_record = service.create_dataset_version(name="baseline", selected_samples=[sample])

    assert first_record.id == 1
    assert second_record.id == 2
    assert first_record.version == "v001"
    assert second_record.version == "v002"
    assert first_record.manifest_path != second_record.manifest_path
    assert Path(first_record.manifest_path).is_file()
    assert Path(second_record.manifest_path).is_file()


def test_create_dataset_version_rejects_invalid_names(
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path)
    service = DatasetRegistryService(project_root=tmp_path)

    for invalid_name in ("../outside", "a?b", "***"):
        with pytest.raises(ValueError, match="Dataset name must contain only"):
            service.create_dataset_version(
                name=invalid_name,
                selected_samples=[sample],
            )


def test_create_dataset_version_accepts_lowercase_name_and_rejects_uppercase(
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path)
    service = DatasetRegistryService(project_root=tmp_path)

    record = service.create_dataset_version(name="baseline", selected_samples=[sample])

    assert record.name == "baseline"

    with pytest.raises(ValueError, match="Dataset name must contain only"):
        service.create_dataset_version(name="BASELINE", selected_samples=[sample])


def test_create_dataset_version_uses_next_version_after_legacy_uppercase_manifest(
    tmp_path: Path,
) -> None:
    sample = _make_sample(tmp_path)
    manifest_dir = tmp_path / "data" / "datasets" / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "BASELINE_v001.json").write_text("{}", encoding="utf-8")
    service = DatasetRegistryService(project_root=tmp_path)

    record = service.create_dataset_version(name="baseline", selected_samples=[sample])

    assert record.version == "v002"
    assert Path(record.manifest_path) == manifest_dir / "baseline_v002.json"
    assert Path(record.manifest_path).is_file()
