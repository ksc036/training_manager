from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_ENGINES: dict[str, Engine] = {}
_SESSION_FACTORIES: dict[str, sessionmaker[Session]] = {}


def _database_url(project_root: Path) -> str:
    db_path = project_root / "db" / "training_manager.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


def get_engine(project_root: Path) -> Engine:
    key = str(project_root.resolve())
    if key not in _ENGINES:
        _ENGINES[key] = create_engine(_database_url(project_root), future=True)
    return _ENGINES[key]


def get_session_factory(project_root: Path) -> sessionmaker[Session]:
    key = str(project_root.resolve())
    if key not in _SESSION_FACTORIES:
        init_database(project_root)
        _SESSION_FACTORIES[key] = sessionmaker(
            bind=get_engine(project_root),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SESSION_FACTORIES[key]


def init_database(project_root: Path) -> None:
    from app import models  # noqa: F401

    engine = get_engine(project_root)
    Base.metadata.create_all(engine)
    _ensure_compatibility_columns(engine)


def _ensure_compatibility_columns(engine: Engine) -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "training_runs" not in table_names:
        return

    columns = {column["name"] for column in inspector.get_columns("training_runs")}
    if "trainer_backend" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE training_runs "
                    "ADD COLUMN trainer_backend VARCHAR(40) NOT NULL DEFAULT 'simulated'"
                )
            )
    if "best_epoch" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE training_runs "
                    "ADD COLUMN best_epoch INTEGER"
                )
            )
    if "best_checkpoint_path" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE training_runs "
                    "ADD COLUMN best_checkpoint_path TEXT"
                )
            )
