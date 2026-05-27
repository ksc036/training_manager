import json
from pathlib import Path

from trainer.datasets import build_manifest_payload, write_manifest
from trainer.scanner import ScanResult, SampleRecord, scan_approved_samples


def test_scan_approved_samples_returns_valid_samples(tmp_path: Path) -> None:
    sample_dir = tmp_path / "annotation_complete" / "sample_a"
    image_dir = sample_dir / "Image"
    mask_dir = sample_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "sample_a.png").write_bytes(b"image")
    (mask_dir / "sample_a.png").write_bytes(b"mask")

    result = scan_approved_samples(tmp_path / "annotation_complete")

    assert len(result.samples) == 1
    assert result.samples[0].sample_id == "sample_a"
    assert result.issues == []


def test_scan_approved_samples_reports_missing_directories(tmp_path: Path) -> None:
    sample_dir = tmp_path / "annotation_complete" / "sample_a"
    sample_dir.mkdir(parents=True)
    (sample_dir / "Image").mkdir()

    result = scan_approved_samples(tmp_path / "annotation_complete")

    assert result.samples == []
    assert result.issues == ["sample_a: missing Image or mask directory"]


def test_scan_approved_samples_rejects_ambiguous_files(tmp_path: Path) -> None:
    sample_dir = tmp_path / "annotation_complete" / "sample_a"
    image_dir = sample_dir / "Image"
    mask_dir = sample_dir / "mask"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (image_dir / "sample_a_a.png").write_bytes(b"image")
    (image_dir / "sample_a_b.png").write_bytes(b"image")
    (mask_dir / "sample_a.png").write_bytes(b"mask")

    result = scan_approved_samples(tmp_path / "annotation_complete")

    assert result.samples == []
    assert result.issues == ["sample_a: expected exactly one file in Image and mask"]


def test_scan_approved_samples_handles_missing_source_root(tmp_path: Path) -> None:
    result = scan_approved_samples(tmp_path / "annotation_complete")

    assert result.samples == []
    assert result.issues == [f"{tmp_path / 'annotation_complete'}: source root does not exist"]


def test_build_manifest_payload_serializes_scan_result(tmp_path: Path) -> None:
    sample = SampleRecord(
        sample_id="sample_a",
        sample_path=tmp_path / "annotation_complete" / "sample_a",
        image_path=tmp_path / "annotation_complete" / "sample_a" / "Image" / "sample_a.png",
        mask_path=tmp_path / "annotation_complete" / "sample_a" / "mask" / "sample_a.png",
    )
    scan_result = ScanResult(samples=[sample], issues=["sample_a: warning"])

    payload = build_manifest_payload(scan_result)

    assert payload == {
        "samples": [
            {
                "sample_id": "sample_a",
                "sample_path": str(sample.sample_path),
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            }
        ],
        "issues": ["sample_a: warning"],
    }


def test_write_manifest_writes_json_file(tmp_path: Path) -> None:
    sample = SampleRecord(
        sample_id="sample_a",
        sample_path=tmp_path / "annotation_complete" / "sample_a",
        image_path=tmp_path / "annotation_complete" / "sample_a" / "Image" / "sample_a.png",
        mask_path=tmp_path / "annotation_complete" / "sample_a" / "mask" / "sample_a.png",
    )
    scan_result = ScanResult(samples=[sample], issues=[])
    manifest_path = tmp_path / "data" / "datasets" / "manifests" / "baseline.json"

    written_path = write_manifest(manifest_path, scan_result)

    assert written_path == manifest_path
    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == build_manifest_payload(
        scan_result
    )
