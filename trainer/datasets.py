from __future__ import annotations

import json
from pathlib import Path

from trainer.scanner import ScanResult


def build_manifest_payload(scan_result: ScanResult) -> dict[str, object]:
    return {
        "samples": [
            {
                "sample_id": sample.sample_id,
                "sample_path": str(sample.sample_path),
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            }
            for sample in scan_result.samples
        ],
        "issues": list(scan_result.issues),
    }


def write_manifest(manifest_path: Path, scan_result: ScanResult) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(build_manifest_payload(scan_result), indent=2),
        encoding="utf-8",
    )
    return manifest_path
