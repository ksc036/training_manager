from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from app.db import get_session_factory
from app.models import DatasetSample, DatasetVersion
from app.repositories.datasets import create_dataset_version_row, write_dataset_manifest
from trainer.scanner import SampleRecord, scan_approved_samples


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


@dataclass(frozen=True)
class DatasetSampleSummary:
    id: int
    sample_id: str
    sample_path: str
    image_path: str
    mask_path: str
    width: int | None
    height: int | None


@dataclass(frozen=True)
class DatasetSamplePage:
    dataset_version_id: int
    page: int
    page_size: int
    total: int
    total_pages: int
    has_prev: bool
    has_next: bool
    items: list[DatasetSampleSummary]

    def to_payload(self) -> dict[str, object]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "page": self.page,
            "page_size": self.page_size,
            "total": self.total,
            "total_pages": self.total_pages,
            "has_prev": self.has_prev,
            "has_next": self.has_next,
            "items": [
                {
                    "id": item.id,
                    "sample_id": item.sample_id,
                    "sample_path": item.sample_path,
                    "image_path": item.image_path,
                    "mask_path": item.mask_path,
                    "width": item.width,
                    "height": item.height,
                }
                for item in self.items
            ],
        }


@dataclass(frozen=True)
class ApprovedSampleSummary:
    sample_id: str
    sample_path: str
    image_path: str | None
    mask_path: str | None
    width: int | None
    height: int | None
    status: str
    selectable: bool


@dataclass(frozen=True)
class ApprovedSamplePage:
    page: int
    page_size: int
    total_count: int
    total_pages: int
    has_prev: bool
    has_next: bool
    items: list[ApprovedSampleSummary]

    def to_payload(self) -> dict[str, object]:
        return {
            "page": self.page,
            "page_size": self.page_size,
            "total_count": self.total_count,
            "total_pages": self.total_pages,
            "has_prev": self.has_prev,
            "has_next": self.has_next,
            "items": [
                {
                    "sample_id": item.sample_id,
                    "sample_path": item.sample_path,
                    "image_path": item.image_path,
                    "mask_path": item.mask_path,
                    "width": item.width,
                    "height": item.height,
                    "status": item.status,
                    "selectable": item.selectable,
                }
                for item in self.items
            ],
        }


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

    def _read_image_dimensions(self, image_path: Path) -> tuple[int | None, int | None]:
        try:
            from PIL import Image
        except Exception:
            return None, None

        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            return None, None
        return width, height

    def _load_dataset_sample_rows(
        self,
        dataset_version_id: int,
    ) -> list[DatasetSampleSummary]:
        rows: list[DatasetSampleSummary] = []
        with self._session_factory() as session:
            dataset_rows = (
                session.query(DatasetSample)
                .filter(DatasetSample.dataset_version_id == dataset_version_id)
                .order_by(DatasetSample.id.asc())
                .all()
            )
            for row in dataset_rows:
                width, height = self._read_image_dimensions(Path(row.image_path))
                rows.append(
                    DatasetSampleSummary(
                        id=row.id,
                        sample_id=row.sample_id,
                        sample_path=row.sample_path,
                        image_path=row.image_path,
                        mask_path=row.mask_path,
                        width=width,
                        height=height,
                    )
                )
        if rows:
            return rows

        dataset_version = self.get_dataset_version(dataset_version_id)
        if dataset_version is None:
            return []

        manifest_path = Path(dataset_version.manifest_path)
        if not manifest_path.is_file():
            return []

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for index, sample in enumerate(payload.get("samples", []), start=1):
            image_path = Path(str(sample.get("image_path", "")))
            width, height = self._read_image_dimensions(image_path)
            rows.append(
                DatasetSampleSummary(
                    id=index,
                    sample_id=str(sample.get("sample_id", "")),
                    sample_path=str(sample.get("sample_path", "")),
                    image_path=str(sample.get("image_path", "")),
                    mask_path=str(sample.get("mask_path", "")),
                    width=width,
                    height=height,
                )
            )
        return rows

    def list_dataset_samples(
        self,
        dataset_version_id: int,
        *,
        sample_id_query: str = "",
        page: int = 1,
        page_size: int = 20,
    ) -> DatasetSamplePage | None:
        dataset_version = self.get_dataset_version(dataset_version_id)
        if dataset_version is None:
            return None

        page = max(page, 1)
        page_size = max(page_size, 1)
        sample_id_query = sample_id_query.strip().lower()

        samples = self._load_dataset_sample_rows(dataset_version_id)
        if sample_id_query:
            samples = [
                sample
                for sample in samples
                if sample_id_query in sample.sample_id.lower()
            ]

        total = len(samples)
        total_pages = max((total + page_size - 1) // page_size, 1) if total else 0
        if total == 0:
            page = 1
        elif page > total_pages:
            page = total_pages

        start = (page - 1) * page_size
        end = start + page_size
        items = samples[start:end]
        return DatasetSamplePage(
            dataset_version_id=dataset_version_id,
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            has_prev=page > 1 and total > 0,
            has_next=page < total_pages,
            items=items,
        )

    def list_approved_samples(
        self,
        source_root: Path,
        *,
        query: str = "",
        resolution_bucket: str = "",
        width_min: int | None = None,
        width_max: int | None = None,
        height_min: int | None = None,
        height_max: int | None = None,
        status: str = "",
        sort: str = "sample_id",
        page: int = 1,
        page_size: int = 25,
    ) -> ApprovedSamplePage:
        page = max(page, 1)
        page_size = max(page_size, 1)
        query = query.strip().lower()
        status = status.strip().lower()

        rows = self._load_approved_sample_rows(source_root)
        if query:
            rows = [row for row in rows if query in row.sample_id.lower()]
        if resolution_bucket:
            rows = [row for row in rows if self._matches_resolution_bucket(row, resolution_bucket)]
        if width_min is not None:
            rows = [row for row in rows if row.width is not None and row.width >= width_min]
        if width_max is not None:
            rows = [row for row in rows if row.width is not None and row.width <= width_max]
        if height_min is not None:
            rows = [row for row in rows if row.height is not None and row.height >= height_min]
        if height_max is not None:
            rows = [row for row in rows if row.height is not None and row.height <= height_max]
        if status:
            rows = [row for row in rows if row.status == status]

        rows = self._sort_approved_rows(rows, sort)

        total_count = len(rows)
        total_pages = max((total_count + page_size - 1) // page_size, 1) if total_count else 0
        if total_count == 0:
            page = 1
        elif page > total_pages:
            page = total_pages

        start = (page - 1) * page_size
        end = start + page_size
        return ApprovedSamplePage(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
            has_prev=page > 1 and total_count > 0,
            has_next=page < total_pages,
            items=rows[start:end],
        )

    def approved_records_by_ids(
        self,
        source_root: Path,
        sample_ids: set[str],
    ) -> list[SampleRecord]:
        if not sample_ids:
            return []
        scan_result = scan_approved_samples(source_root)
        return [
            sample
            for sample in scan_result.samples
            if sample.sample_id in sample_ids
        ]

    def _matches_resolution_bucket(
        self,
        row: ApprovedSampleSummary,
        bucket: str,
    ) -> bool:
        if row.width is None or row.height is None:
            return False
        longest_edge = max(row.width, row.height)
        if bucket == "<=256":
            return longest_edge <= 256
        if bucket == "257-512":
            return 257 <= longest_edge <= 512
        if bucket == "513-1024":
            return 513 <= longest_edge <= 1024
        if bucket == "1025+":
            return longest_edge >= 1025
        return True

    def _sort_approved_rows(
        self,
        rows: list[ApprovedSampleSummary],
        sort: str,
    ) -> list[ApprovedSampleSummary]:
        if sort == "width_desc":
            return sorted(rows, key=lambda row: (row.width or -1, row.sample_id), reverse=True)
        if sort == "height_desc":
            return sorted(rows, key=lambda row: (row.height or -1, row.sample_id), reverse=True)
        if sort == "width_asc":
            return sorted(rows, key=lambda row: (row.width is None, row.width or 0, row.sample_id))
        if sort == "height_asc":
            return sorted(rows, key=lambda row: (row.height is None, row.height or 0, row.sample_id))
        return sorted(rows, key=lambda row: row.sample_id)

    def _single_file_in_dir(self, path: Path) -> Path | None:
        if not path.is_dir():
            return None
        files = [entry for entry in sorted(path.iterdir()) if entry.is_file()]
        if len(files) != 1:
            return None
        return files[0]

    def _load_approved_sample_rows(self, source_root: Path) -> list[ApprovedSampleSummary]:
        if not source_root.is_dir():
            return []

        rows: list[ApprovedSampleSummary] = []
        for sample_dir in sorted(entry for entry in source_root.iterdir() if entry.is_dir()):
            image_dir = sample_dir / "Image"
            mask_dir = sample_dir / "mask"
            image_path = self._single_file_in_dir(image_dir)
            mask_path = self._single_file_in_dir(mask_dir)
            width, height = (None, None)
            if image_path is not None:
                width, height = self._read_image_dimensions(image_path)

            if image_path is not None and mask_path is not None:
                row_status = "ready"
            elif not image_dir.is_dir() or image_path is None:
                row_status = "missing_image"
            else:
                row_status = "missing_mask"

            rows.append(
                ApprovedSampleSummary(
                    sample_id=sample_dir.name,
                    sample_path=str(sample_dir),
                    image_path=str(image_path) if image_path is not None else None,
                    mask_path=str(mask_path) if mask_path is not None else None,
                    width=width,
                    height=height,
                    status=row_status,
                    selectable=row_status == "ready",
                )
            )
        return rows
