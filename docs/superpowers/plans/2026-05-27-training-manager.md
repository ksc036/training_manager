# Training Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone FastAPI-based training manager that scans approved samples, creates dataset versions, launches pre-registered model training runs, tracks metrics, and compares results.

**Architecture:** The app uses FastAPI with server-rendered templates and JSON endpoints, SQLite for metadata, and the filesystem for manifests, workspaces, logs, and checkpoints. Dataset creation flows from a read-only source root into manifest-backed dataset versions, then into run directories managed by a subprocess-based training runner.

**Tech Stack:** Python, FastAPI, Jinja2, SQLite, SQLAlchemy, Pydantic, Pytest, imageio, hashlib, subprocess

---

### Task 1: Scaffold the Project Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/config.py`
- Create: `app/bootstrap.py`
- Create: `app/web/__init__.py`
- Create: `app/web/templates/base.html`
- Create: `app/web/static/styles.css`
- Create: `tests/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from app.bootstrap import ensure_runtime_dirs


def test_ensure_runtime_dirs_creates_expected_structure(tmp_path: Path) -> None:
    ensure_runtime_dirs(tmp_path)

    assert (tmp_path / "data" / "datasets" / "manifests").is_dir()
    assert (tmp_path / "data" / "datasets" / "workspaces").is_dir()
    assert (tmp_path / "data" / "datasets" / "snapshots").is_dir()
    assert (tmp_path / "data" / "runs").is_dir()
    assert (tmp_path / "data" / "artifacts").is_dir()
    assert (tmp_path / "db").is_dir()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError` for `app.bootstrap`

- [ ] **Step 3: Write minimal implementation**

```python
# app/bootstrap.py
from pathlib import Path


RUNTIME_DIRS = (
    "data/datasets/manifests",
    "data/datasets/workspaces",
    "data/datasets/snapshots",
    "data/runs",
    "data/artifacts",
    "db",
)


def ensure_runtime_dirs(project_root: Path) -> None:
    for relative_dir in RUNTIME_DIRS:
        (project_root / relative_dir).mkdir(parents=True, exist_ok=True)
```

```python
# app/main.py
from fastapi import FastAPI

from app.bootstrap import ensure_runtime_dirs


def create_app() -> FastAPI:
    app = FastAPI(title="Training Manager")

    @app.on_event("startup")
    def startup() -> None:
        ensure_runtime_dirs(app.state.project_root)

    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bootstrap.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml app/__init__.py app/main.py app/config.py app/bootstrap.py app/web/__init__.py app/web/templates/base.html app/web/static/styles.css tests/test_bootstrap.py
git commit -m "feat: scaffold training manager project"
```

### Task 2: Add Database Schema and Session Setup

**Files:**
- Create: `app/db.py`
- Create: `app/models.py`
- Create: `app/repositories/__init__.py`
- Create: `tests/test_db_schema.py`

- [ ] **Step 1: Write the failing test**

```python
from sqlalchemy import create_engine, inspect

from app.db import Base
from app.models import DatasetVersion, TrainingRun


def test_schema_contains_core_tables() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    inspector = inspect(engine)

    assert "dataset_versions" in inspector.get_table_names()
    assert "dataset_samples" in inspector.get_table_names()
    assert "training_runs" in inspector.get_table_names()
    assert "run_metrics" in inspector.get_table_names()
    assert "run_events" in inspector.get_table_names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_schema.py -v`
Expected: FAIL with `ModuleNotFoundError` for `app.db`

- [ ] **Step 3: Write minimal implementation**

```python
# app/db.py
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
```

```python
# app/models.py
from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    version: Mapped[str] = mapped_column(String(40))
    source_root: Mapped[str] = mapped_column(Text)
    snapshot_mode: Mapped[str] = mapped_column(String(20))
    sample_count: Mapped[int] = mapped_column(Integer)
    manifest_path: Mapped[str] = mapped_column(Text)


class DatasetSample(Base):
    __tablename__ = "dataset_samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    sample_id: Mapped[str] = mapped_column(String(255))
    sample_path: Mapped[str] = mapped_column(Text)
    image_path: Mapped[str] = mapped_column(Text)
    mask_path: Mapped[str] = mapped_column(Text)
    image_hash: Mapped[str] = mapped_column(String(64))
    mask_hash: Mapped[str] = mapped_column(String(64))


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_version_id: Mapped[int] = mapped_column(ForeignKey("dataset_versions.id"))
    run_name: Mapped[str] = mapped_column(String(120))
    model_name: Mapped[str] = mapped_column(String(80))
    encoder_name: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20))
    run_dir: Mapped[str] = mapped_column(Text)
    config_path: Mapped[str] = mapped_column(Text)
    log_path: Mapped[str] = mapped_column(Text)


class RunMetric(Base):
    __tablename__ = "run_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float] = mapped_column(Float)
    val_loss: Mapped[float] = mapped_column(Float)
    dice: Mapped[float] = mapped_column(Float)
    iou: Mapped[float] = mapped_column(Float)
    precision: Mapped[float] = mapped_column(Float)
    recall: Mapped[float] = mapped_column(Float)


class RunEvent(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id"))
    event_type: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/db.py app/models.py app/repositories/__init__.py tests/test_db_schema.py
git commit -m "feat: add core sqlite schema"
```

### Task 3: Build the Dataset Scanner and Manifest Writer

**Files:**
- Create: `trainer/scanner.py`
- Create: `trainer/datasets.py`
- Create: `tests/test_scanner.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from trainer.scanner import scan_approved_samples


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL with `ModuleNotFoundError` for `trainer.scanner`

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/scanner.py
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


def _first_file(path: Path) -> Path | None:
    files = [entry for entry in sorted(path.iterdir()) if entry.is_file()]
    return files[0] if files else None


def scan_approved_samples(source_root: Path) -> ScanResult:
    result = ScanResult()
    for sample_dir in sorted(source_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        image_dir = sample_dir / "Image"
        mask_dir = sample_dir / "mask"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            result.issues.append(f"{sample_dir.name}: missing Image or mask directory")
            continue
        image_path = _first_file(image_dir)
        mask_path = _first_file(mask_dir)
        if image_path is None or mask_path is None:
            result.issues.append(f"{sample_dir.name}: missing image or mask file")
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/scanner.py trainer/datasets.py tests/test_scanner.py
git commit -m "feat: add approved sample scanner"
```

### Task 4: Implement Dataset Registry Persistence

**Files:**
- Create: `app/repositories/datasets.py`
- Create: `app/services/datasets.py`
- Create: `tests/test_dataset_registry.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from app.services.datasets import DatasetRegistryService
from trainer.scanner import SampleRecord


def test_create_dataset_version_writes_manifest_and_returns_record(tmp_path: Path) -> None:
    sample = SampleRecord(
        sample_id="sample_a",
        sample_path=tmp_path / "annotation_complete" / "sample_a",
        image_path=tmp_path / "annotation_complete" / "sample_a" / "Image" / "sample_a.png",
        mask_path=tmp_path / "annotation_complete" / "sample_a" / "mask" / "sample_a.png",
    )
    service = DatasetRegistryService(project_root=tmp_path)

    record = service.create_dataset_version(name="baseline", selected_samples=[sample])

    assert record.name == "baseline"
    assert record.sample_count == 1
    assert Path(record.manifest_path).is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset_registry.py -v`
Expected: FAIL with `ModuleNotFoundError` for `app.services.datasets`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/datasets.py
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetVersionRecord:
    name: str
    version: str
    sample_count: int
    manifest_path: str


class DatasetRegistryService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def create_dataset_version(self, name: str, selected_samples: list) -> DatasetVersionRecord:
        version = "v001"
        manifest_dir = self.project_root / "data" / "datasets" / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{name}_{version}.json"
        payload = {
            "name": name,
            "version": version,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "sample_path": str(sample.sample_path),
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                }
                for sample in selected_samples
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return DatasetVersionRecord(
            name=name,
            version=version,
            sample_count=len(selected_samples),
            manifest_path=str(manifest_path),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/repositories/datasets.py app/services/datasets.py tests/test_dataset_registry.py
git commit -m "feat: add dataset registry service"
```

### Task 5: Expose the Datasets Screen and Dataset Creation API

**Files:**
- Create: `app/web/routes_datasets.py`
- Create: `app/web/templates/datasets.html`
- Create: `tests/test_datasets_routes.py`

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_datasets_page_renders() -> None:
    client = TestClient(create_app())

    response = client.get("/datasets")

    assert response.status_code == 200
    assert "Datasets" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_datasets_routes.py -v`
Expected: FAIL with `404 != 200`

- [ ] **Step 3: Write minimal implementation**

```python
# app/web/routes_datasets.py
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory="app/web/templates")


@router.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="datasets.html",
        context={"page_title": "Datasets"},
    )
```

```html
<!-- app/web/templates/datasets.html -->
{% extends "base.html" %}
{% block content %}
<h1>Datasets</h1>
<p>Scan approved samples and create dataset versions.</p>
{% endblock %}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_datasets_routes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/web/routes_datasets.py app/web/templates/datasets.html tests/test_datasets_routes.py
git commit -m "feat: add datasets page"
```

### Task 6: Add the Pre-Registered Model Catalog and Train Screen

**Files:**
- Create: `trainer/model_registry.py`
- Create: `app/web/routes_train.py`
- Create: `app/web/templates/train.html`
- Create: `tests/test_model_registry.py`

- [ ] **Step 1: Write the failing test**

```python
from trainer.model_registry import list_registered_models


def test_model_registry_exposes_supported_models() -> None:
    models = list_registered_models()

    assert [model.key for model in models][:2] == ["unet", "unetplusplus"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_registry.py -v`
Expected: FAIL with `ModuleNotFoundError` for `trainer.model_registry`

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/model_registry.py
from dataclasses import dataclass


@dataclass(frozen=True)
class RegisteredModel:
    key: str
    display_name: str
    default_encoder: str


REGISTERED_MODELS = [
    RegisteredModel(key="unet", display_name="U-Net", default_encoder="resnet34"),
    RegisteredModel(key="unetplusplus", display_name="U-Net++", default_encoder="resnet34"),
    RegisteredModel(key="deeplabv3", display_name="DeepLabV3", default_encoder="resnet50"),
]


def list_registered_models() -> list[RegisteredModel]:
    return REGISTERED_MODELS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_registry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/model_registry.py app/web/routes_train.py app/web/templates/train.html tests/test_model_registry.py
git commit -m "feat: add registered model catalog"
```

### Task 7: Implement the Training Runner and Run Tracking

**Files:**
- Create: `trainer/runners.py`
- Create: `trainer/worker.py`
- Create: `app/services/runs.py`
- Create: `tests/test_training_runner.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from app.services.runs import RunService


def test_start_run_creates_run_directory_and_config(tmp_path: Path) -> None:
    service = RunService(project_root=tmp_path)

    run = service.start_run(
        dataset_version_id=1,
        model_name="unet",
        encoder_name="resnet34",
        epochs=5,
        batch_size=2,
        learning_rate=1e-3,
    )

    assert (Path(run.run_dir) / "config.json").is_file()
    assert run.status == "queued"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_training_runner.py -v`
Expected: FAIL with `ModuleNotFoundError` for `app.services.runs`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/runs.py
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunRecord:
    run_name: str
    run_dir: str
    status: str


class RunService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def start_run(
        self,
        dataset_version_id: int,
        model_name: str,
        encoder_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
    ) -> RunRecord:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = self.project_root / "data" / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_version_id": dataset_version_id,
                    "model_name": model_name,
                    "encoder_name": encoder_name,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return RunRecord(run_name=run_name, run_dir=str(run_dir), status="queued")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_training_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/runners.py trainer/worker.py app/services/runs.py tests/test_training_runner.py
git commit -m "feat: add run creation service"
```

### Task 8: Add Runs and Compare Screens with Summary Queries

**Files:**
- Create: `app/web/routes_runs.py`
- Create: `app/web/routes_compare.py`
- Create: `app/web/templates/runs.html`
- Create: `app/web/templates/compare.html`
- Create: `app/services/compare.py`
- Create: `tests/test_compare_service.py`

- [ ] **Step 1: Write the failing test**

```python
from app.services.compare import rank_runs_by_best_dice


def test_rank_runs_by_best_dice_orders_highest_first() -> None:
    ranked = rank_runs_by_best_dice(
        [
            {"run_name": "run_b", "best_dice": 0.81},
            {"run_name": "run_a", "best_dice": 0.92},
        ]
    )

    assert [row["run_name"] for row in ranked] == ["run_a", "run_b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare_service.py -v`
Expected: FAIL with `ModuleNotFoundError` for `app.services.compare`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/compare.py
def rank_runs_by_best_dice(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: row["best_dice"], reverse=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/web/routes_runs.py app/web/routes_compare.py app/web/templates/runs.html app/web/templates/compare.html app/services/compare.py tests/test_compare_service.py
git commit -m "feat: add run comparison views"
```

### Task 9: Wire the App Together and Verify End-to-End Basics

**Files:**
- Modify: `app/main.py`
- Modify: `app/web/templates/base.html`
- Create: `tests/test_app_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from fastapi.testclient import TestClient

from app.main import create_app


def test_core_pages_are_available() -> None:
    client = TestClient(create_app())

    assert client.get("/datasets").status_code == 200
    assert client.get("/train").status_code == 200
    assert client.get("/runs").status_code == 200
    assert client.get("/compare").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app_smoke.py -v`
Expected: FAIL because routes are not all mounted

- [ ] **Step 3: Write minimal implementation**

```python
# app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.web.routes_compare import router as compare_router
from app.web.routes_datasets import router as datasets_router
from app.web.routes_runs import router as runs_router
from app.web.routes_train import router as train_router


def create_app() -> FastAPI:
    app = FastAPI(title="Training Manager")
    app.include_router(datasets_router)
    app.include_router(train_router)
    app.include_router(runs_router)
    app.include_router(compare_router)
    app.mount("/static", StaticFiles(directory="app/web/static"), name="static")
    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/web/templates/base.html tests/test_app_smoke.py
git commit -m "feat: wire training manager mvp"
```
