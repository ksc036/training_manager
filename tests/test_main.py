import asyncio
import sqlite3
from pathlib import Path

from sqlalchemy import create_engine, inspect

from app.db import init_database
from app.main import create_app


def test_create_app_uses_project_root_and_bootstraps_runtime_dirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    project_root = tmp_path / "project_root"
    project_root.mkdir()
    app = create_app(project_root=project_root)

    assert app.state.project_root == project_root

    async def run_lifespan() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(run_lifespan())

    assert (project_root / "data" / "datasets" / "manifests").is_dir()
    assert (project_root / "data" / "datasets" / "workspaces").is_dir()
    assert (project_root / "data" / "datasets" / "snapshots").is_dir()
    assert (project_root / "data" / "runs").is_dir()
    assert (project_root / "data" / "artifacts").is_dir()
    assert (project_root / "db").is_dir()


def test_init_database_adds_missing_trainer_backend_column(tmp_path: Path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "training_manager.sqlite"

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE training_runs (
                id INTEGER PRIMARY KEY,
                dataset_version_id INTEGER NOT NULL,
                run_name VARCHAR(120) NOT NULL,
                model_name VARCHAR(80) NOT NULL,
                encoder_name VARCHAR(80) NOT NULL,
                status VARCHAR(20) NOT NULL,
                run_dir TEXT NOT NULL,
                config_path TEXT NOT NULL,
                log_path TEXT NOT NULL
            )
            """
        )
        connection.commit()

    init_database(tmp_path)

    inspector = inspect(create_engine(f"sqlite:///{db_path}"))
    columns = {column["name"] for column in inspector.get_columns("training_runs")}
    assert "trainer_backend" in columns
    assert "best_epoch" in columns
    assert "best_checkpoint_path" in columns
