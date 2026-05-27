from __future__ import annotations

from pathlib import Path

from app.services.datasets import DatasetRegistryService
from app.services.runs import RunService


def rank_runs_by_best_dice(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: row["best_dice"], reverse=True)


def list_ranked_runs(
    project_root: Path,
    *,
    model_name: str | None = None,
    run_name_query: str | None = None,
    sort: str = "best_dice_desc",
) -> list[dict]:
    run_service = RunService(project_root=project_root)
    dataset_service = DatasetRegistryService(project_root=project_root)
    dataset_lookup = {
        dataset.id: f"{dataset.name} {dataset.version}"
        for dataset in dataset_service.list_dataset_versions()
    }
    rows = []
    for run in run_service.list_runs():
        if model_name and run.model_name != model_name:
            continue
        if run_name_query and run_name_query.lower() not in run.run_name.lower():
            continue
        rows.append(
            {
                "run_name": run.run_name,
                "dataset_version_id": run.dataset_version_id,
                "dataset_label": dataset_lookup.get(
                    run.dataset_version_id,
                    f"dataset {run.dataset_version_id}",
                ),
                "model_name": run.model_name,
                "encoder_name": run.encoder_name,
                "best_dice": run.best_dice if run.best_dice is not None else -1.0,
            }
        )
    if sort == "best_dice_asc":
        return sorted(rows, key=lambda row: row["best_dice"])
    return rank_runs_by_best_dice(rows)
