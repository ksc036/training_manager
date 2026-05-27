from __future__ import annotations

import re
from dataclasses import dataclass
import json
from pathlib import Path

from app.db import get_session_factory
from app.models import DatasetVersion
from app.repositories.datasets import create_dataset_version_row, write_dataset_manifest
from trainer.scanner import SampleRecord


@dataclass
class DatasetVersionRecord:
    id: int
    name: str
    version: str
    sample_count: int
    manifest_path: str


@dataclass(frozen=True)
class DatasetVersionSummary:
    id: int
    name: str
    version: str
    manifest_path: str
    sample_count: int


class DatasetRegistryService:
    _VALID_NAME_PATTERN = re.compile(r"^(?=.*[a-z0-9])[a-z0-9_-]+$")

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._session_factory = get_session_factory(project_root)

    def _manifest_dir(self) -> Path:
        return self.project_root / "data" / "datasets" / "manifests"

    def _validate_name(self, name: str) -> str:
        if not self._VALID_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Dataset name must contain only lowercase ASCII letters, digits, "
                "underscores, and hyphens, with at least one alphanumeric character."
            )
        return name

    def _next_version(self, name: str) -> str:
        existing_versions: list[int] = []
        normalized_prefix = f"{name}_v"
        manifest_dir = self._manifest_dir()
        if not manifest_dir.is_dir():
            return "v001"

        for manifest_path in manifest_dir.iterdir():
            if not manifest_path.is_file() or manifest_path.suffix.lower() != ".json":
                continue
            stem = manifest_path.stem
            stem_lower = stem.lower()
            if not stem_lower.startswith(normalized_prefix):
                continue
            suffix = stem_lower[len(normalized_prefix) :]
            if suffix.isdigit():
                existing_versions.append(int(suffix))
        next_version = max(existing_versions, default=0) + 1
        return f"v{next_version:03d}"

    def create_dataset_version(
        self,
        name: str,
        selected_samples: list[SampleRecord],
    ) -> DatasetVersionRecord:
        validated_name = self._validate_name(name)
        version = self._next_version(validated_name)
        manifest_path = self._manifest_dir() / f"{validated_name}_{version}.json"
        samples = [
            {
                "sample_id": sample.sample_id,
                "sample_path": str(sample.sample_path),
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            }
            for sample in selected_samples
        ]
        written_path = write_dataset_manifest(
            manifest_path,
            name=validated_name,
            version=version,
            samples=samples,
        )
        with self._session_factory() as session:
            dataset_version = create_dataset_version_row(
                session,
                name=validated_name,
                version=version,
                source_root=str(self.project_root),
                snapshot_mode="manifest",
                sample_count=len(selected_samples),
                manifest_path=str(written_path),
                samples=samples,
            )
            session.commit()
        return DatasetVersionRecord(
            id=dataset_version.id,
            name=validated_name,
            version=version,
            sample_count=len(selected_samples),
            manifest_path=str(written_path),
        )

    def list_dataset_versions(self) -> list[DatasetVersionSummary]:
        summaries: list[DatasetVersionSummary] = []
        with self._session_factory() as session:
            rows = session.query(DatasetVersion).order_by(DatasetVersion.id.asc()).all()
            for row in rows:
                summaries.append(
                    DatasetVersionSummary(
                        id=row.id,
                        name=row.name,
                        version=row.version,
                        manifest_path=row.manifest_path,
                        sample_count=row.sample_count,
                    )
                )
        if summaries:
            return summaries

        manifest_dir = self._manifest_dir()
        if not manifest_dir.is_dir():
            return []

        for index, manifest_path in enumerate(sorted(manifest_dir.glob("*.json")), start=1):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            summaries.append(
                DatasetVersionSummary(
                    id=index,
                    name=str(payload.get("name", manifest_path.stem)),
                    version=str(payload.get("version", "")),
                    manifest_path=str(manifest_path),
                    sample_count=len(payload.get("samples", [])),
                )
            )
        return summaries

    def get_dataset_version(self, dataset_version_id: int) -> DatasetVersionSummary | None:
        for dataset in self.list_dataset_versions():
            if dataset.id == dataset_version_id:
                return dataset
        return None
