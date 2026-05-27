from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import DatasetSample, DatasetVersion


def write_dataset_manifest(
    manifest_path: Path,
    *,
    name: str,
    version: str,
    samples: list[dict[str, str]],
) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "version": version,
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_dataset_version_row(
    session: Session,
    *,
    name: str,
    version: str,
    source_root: str,
    snapshot_mode: str,
    sample_count: int,
    manifest_path: str,
    samples: list[dict[str, str]],
) -> DatasetVersion:
    dataset_version = DatasetVersion(
        name=name,
        version=version,
        source_root=source_root,
        snapshot_mode=snapshot_mode,
        sample_count=sample_count,
        manifest_path=manifest_path,
    )
    session.add(dataset_version)
    session.flush()

    for sample in samples:
        image_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])
        session.add(
            DatasetSample(
                dataset_version_id=dataset_version.id,
                sample_id=sample["sample_id"],
                sample_path=sample["sample_path"],
                image_path=sample["image_path"],
                mask_path=sample["mask_path"],
                image_hash=_sha256(image_path),
                mask_hash=_sha256(mask_path),
            )
        )

    session.flush()
    return dataset_version
