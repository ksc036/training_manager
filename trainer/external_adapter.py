from __future__ import annotations

import json
import shutil
import signal
import sys
import time
from pathlib import Path

from app.services.datasets import DatasetRegistryService
from trainer.device import describe_torch_device, resolve_torch_device
from trainer.model_zoo import build_segmentation_model
from trainer.worker import (
    _persist_best_checkpoint,
    _persist_metric_full,
    _persist_status,
    run_training,
)

_STOP_REQUESTED = False


def _load_dataset_manifest(run_dir: Path) -> tuple[dict[str, object] | None, Path | None]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset_version_id = int(config["dataset_version_id"])
    project_root = run_dir.parents[2]
    dataset = DatasetRegistryService(project_root=project_root).get_dataset_version(
        dataset_version_id
    )
    if dataset is None:
        return None, None
    manifest_path = Path(dataset.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload, manifest_path


def _split_samples(samples: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(samples) <= 1:
        return samples, []
    val_count = max(1, round(len(samples) * 0.2))
    train_count = max(1, len(samples) - val_count)
    train_samples = samples[:train_count]
    val_samples = samples[train_count:]
    return train_samples, val_samples


def _write_prepared_dataset_files(run_dir: Path) -> None:
    manifest_payload, manifest_path = _load_dataset_manifest(run_dir)
    if manifest_payload is None or manifest_path is None:
        (run_dir / "dataset_summary.json").write_text(
            json.dumps(
                {
                    "dataset_name": "unknown",
                    "dataset_version": "",
                    "manifest_path": None,
                    "sample_count": 0,
                    "train_count": 0,
                    "val_count": 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (run_dir / "split_manifest.json").write_text(
            json.dumps({"train_samples": [], "val_samples": []}, indent=2),
            encoding="utf-8",
        )
        return
    samples = list(manifest_payload.get("samples", []))
    train_samples, val_samples = _split_samples(samples)
    dataset_summary = {
        "dataset_name": manifest_payload.get("name", manifest_path.stem),
        "dataset_version": manifest_payload.get("version", ""),
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
    }
    split_manifest = {
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    (run_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2),
        encoding="utf-8",
    )
    (run_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    _build_workspace(run_dir, train_samples=train_samples, val_samples=val_samples)


def _materialize_file_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_sample(split_dir: Path, sample: dict[str, object]) -> None:
    sample_dir = split_dir / str(sample["sample_id"])
    image_path = sample.get("image_path")
    mask_path = sample.get("mask_path")
    if not image_path or not mask_path:
        return
    image_source = Path(str(image_path))
    mask_source = Path(str(mask_path))
    _materialize_file_link(
        image_source,
        sample_dir / "Image" / image_source.name,
    )
    _materialize_file_link(
        mask_source,
        sample_dir / "mask" / mask_source.name,
    )


def _build_workspace(
    run_dir: Path,
    *,
    train_samples: list[dict[str, object]],
    val_samples: list[dict[str, object]],
) -> None:
    workspace_dir = run_dir / "workspace"
    train_dir = workspace_dir / "train"
    val_dir = workspace_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    for sample in train_samples:
        _materialize_sample(train_dir, sample)
    for sample in val_samples:
        _materialize_sample(val_dir, sample)


def _handle_termination(signum, frame) -> None:  # type: ignore[no-untyped-def]
    del signum, frame
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _can_run_actual_training(run_dir: Path) -> bool:
    try:
        from PIL import Image
        import numpy  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False

    split_manifest_path = run_dir / "split_manifest.json"
    if not split_manifest_path.is_file():
        return False
    payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    train_samples = list(payload.get("train_samples", []))
    if not train_samples:
        return False
    first_sample = train_samples[0]
    image_path = first_sample.get("image_path")
    mask_path = first_sample.get("mask_path")
    if not image_path or not mask_path:
        return False
    try:
        Image.open(image_path).verify()
        Image.open(mask_path).verify()
    except Exception:
        return False
    return True


def _run_actual_training(run_dir: Path) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    signal.signal(signal.SIGTERM, _handle_termination)

    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class WorkspaceDataset(Dataset):
        def __init__(self, samples: list[dict[str, object]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            sample = self.samples[index]
            image = (
                Image.open(str(sample["image_path"]))
                .convert("RGB")
                .resize((64, 64))
            )
            mask = (
                Image.open(str(sample["mask_path"]))
                .convert("L")
                .resize((64, 64))
            )
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            mask_array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask_array[None, :, :])
            return image_tensor, mask_tensor

    def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float, float]:
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        targets = targets.float()
        intersection = float((preds * targets).sum().item())
        pred_sum = float(preds.sum().item())
        target_sum = float(targets.sum().item())
        union = float(((preds + targets) > 0).float().sum().item())
        dice = (2.0 * intersection) / max(pred_sum + target_sum, 1.0)
        iou = intersection / max(union, 1.0)
        precision = intersection / max(pred_sum, 1.0)
        recall = intersection / max(target_sum, 1.0)
        return dice, iou, precision, recall

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
    train_samples = list(split_manifest.get("train_samples", []))
    val_samples = list(split_manifest.get("val_samples", []))
    if not val_samples:
        val_samples = train_samples

    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    epochs = int(config["epochs"])
    model_name = str(config.get("model_name", "external-script-model"))

    train_loader = DataLoader(WorkspaceDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WorkspaceDataset(val_samples), batch_size=batch_size, shuffle=False)

    device = resolve_torch_device()
    model = build_segmentation_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    metrics_path = run_dir / "metrics.csv"
    log_path = run_dir / "train.log"
    _persist_status(run_dir, "running")
    log_path.write_text(
        (
            f"starting actual torch training with model={model_name}\n"
            f"device={describe_torch_device(device)}\n"
        ),
        encoding="utf-8",
    )

    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "dice", "iou", "precision", "recall"],
        )
        writer.writeheader()
        best_dice = -1.0
        best_epoch = 0
        training_started_at = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_started_at = time.perf_counter()
            if _STOP_REQUESTED:
                with log_path.open("a", encoding="utf-8") as log_handle:
                    log_handle.write("training stopped\n")
                _persist_status(run_dir, "stopped")
                return

            model.train()
            train_loss_total = 0.0
            train_batches = 0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()
                train_loss_total += float(loss.item())
                train_batches += 1
            train_loss = train_loss_total / max(train_batches, 1)

            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            dice_total = 0.0
            iou_total = 0.0
            precision_total = 0.0
            recall_total = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    logits = model(images)
                    loss = criterion(logits, masks)
                    val_loss_total += float(loss.item())
                    val_batches += 1
                    dice, iou, precision, recall = compute_metrics(logits, masks)
                    dice_total += dice
                    iou_total += iou
                    precision_total += precision
                    recall_total += recall
            val_loss = val_loss_total / max(val_batches, 1)
            dice = dice_total / max(val_batches, 1)
            iou = iou_total / max(val_batches, 1)
            precision = precision_total / max(val_batches, 1)
            recall = recall_total / max(val_batches, 1)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                }
            )
            _persist_metric_full(
                run_dir,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                dice=dice,
                iou=iou,
                precision=precision,
                recall=recall,
            )
            if dice >= best_dice:
                best_dice = dice
                best_epoch = epoch
                _persist_best_checkpoint(
                    run_dir,
                    epoch=best_epoch,
                    dice=best_dice,
                    checkpoint_payload={
                        "epoch": best_epoch,
                        "best_dice": float(best_dice),
                        "model_name": model_name,
                        "encoder_name": str(config.get("encoder_name", "")),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                )
            with log_path.open("a", encoding="utf-8") as log_handle:
                epoch_elapsed_sec = time.perf_counter() - epoch_started_at
                total_elapsed_sec = time.perf_counter() - training_started_at
                log_handle.write(
                    "epoch "
                    f"{epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"dice={dice:.4f} elapsed_sec={epoch_elapsed_sec:.4f} "
                    f"total_elapsed_sec={total_elapsed_sec:.4f}\n"
                )

    _persist_status(run_dir, "completed")


def run_external_adapter(run_dir: Path) -> None:
    _write_prepared_dataset_files(run_dir)
    if _can_run_actual_training(run_dir):
        _run_actual_training(run_dir)
    else:
        run_training(run_dir)
    with (run_dir / "train.log").open("a", encoding="utf-8") as handle:
        handle.write("external adapter complete\n")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit("Usage: python -m trainer.external_adapter <run_dir>")
    run_external_adapter(Path(argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
