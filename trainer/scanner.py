from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SampleRecord:
    sample_id: str
    sample_path: Path
    image_path: Path
    mask_path: Path


@dataclass
class ScanResult:
    samples: list[SampleRecord] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _single_file(path: Path) -> Path | None:
    files = [entry for entry in sorted(path.iterdir()) if entry.is_file()]
    if len(files) != 1:
        return None
    return files[0]


def scan_approved_samples(source_root: Path) -> ScanResult:
    result = ScanResult()
    if not source_root.is_dir():
        result.issues.append(f"{source_root}: source root does not exist")
        return result

    for sample_dir in sorted(source_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        image_dir = sample_dir / "Image"
        mask_dir = sample_dir / "mask"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            result.issues.append(f"{sample_dir.name}: missing Image or mask directory")
            continue

        image_path = _single_file(image_dir)
        mask_path = _single_file(mask_dir)
        if image_path is None or mask_path is None:
            result.issues.append(
                f"{sample_dir.name}: expected exactly one file in Image and mask"
            )
            continue

        result.samples.append(
            SampleRecord(
                sample_id=sample_dir.name,
                sample_path=sample_dir,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    return result
