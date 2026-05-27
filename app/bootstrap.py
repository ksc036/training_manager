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
