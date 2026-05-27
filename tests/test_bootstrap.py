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
