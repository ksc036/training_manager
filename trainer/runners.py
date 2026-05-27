from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from app.config import PROJECT_ROOT


def resolve_backend_command(
    *,
    trainer_backend: str,
    run_dir: Path,
    backend_config: dict[str, str] | None = None,
) -> list[str]:
    if trainer_backend == "simulated":
        return [
            sys.executable,
            "-m",
            "trainer.worker",
            str(run_dir),
        ]
    if trainer_backend == "external-script":
        entry_command = (backend_config or {}).get("entry_command", "").strip()
        if not entry_command:
            raise ValueError(
                "external-script backend requires backend_config['entry_command']."
            )
        command = shlex.split(entry_command)
        if command and command[0] == "python":
            command[0] = sys.executable
        return [*command, str(run_dir)]
    raise ValueError(f"Unsupported trainer backend: {trainer_backend}")


def launch_training_worker(
    project_root: Path,
    run_dir: Path,
    trainer_backend: str,
    backend_config: dict[str, str] | None = None,
) -> None:
    command = resolve_backend_command(
        trainer_backend=trainer_backend,
        run_dir=run_dir,
        backend_config=backend_config,
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(PROJECT_ROOT)
    )
    if env.get("TRAINING_MANAGER_SYNC_RUNS") == "1":
        subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=env)
        return
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    (run_dir / "run.pid").write_text(str(process.pid), encoding="utf-8")
