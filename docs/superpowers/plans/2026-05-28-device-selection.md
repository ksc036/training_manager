# Device Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared PyTorch device selection with `cuda -> mps -> cpu` priority, use it during training and `/test` inference, and record the chosen device in the training log.

**Architecture:** Introduce a small shared helper in `trainer/device.py`, then thread that helper through the real training adapter and the `/test` inference service. Verification centers on unit tests for device resolution and integration tests that confirm the training log captures the chosen device.

**Tech Stack:** Python, PyTorch, FastAPI, Jinja2, Pytest

---

### Task 1: Add Shared Device Helper

**Files:**
- Create: `trainer/device.py`
- Create: `tests/test_device_selection.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from trainer.device import describe_torch_device, resolve_torch_device


def test_resolve_torch_device_prefers_mps_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    device = resolve_torch_device()

    assert device.type == "mps"
    assert describe_torch_device(device) == "mps"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_device_selection.py`
Expected: FAIL with `ModuleNotFoundError` for `trainer.device`

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/device.py
from __future__ import annotations

import torch


def resolve_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_torch_device(device: torch.device) -> str:
    return device.type
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -q tests/test_device_selection.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/device.py tests/test_device_selection.py
git commit -m "feat: add shared torch device helper"
```

### Task 2: Use Shared Device Helper In Training And Test Inference

**Files:**
- Modify: `trainer/external_adapter.py`
- Modify: `app/services/test_inference.py`
- Modify: `tests/test_training_runner.py`
- Modify: `tests/test_test_inference_service.py`

- [ ] **Step 1: Write the failing test**

```python
def test_external_adapter_logs_selected_device(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TRAINING_MANAGER_SYNC_RUNS", "1")
    # seed valid image dataset and start one external-script run here
    # then assert "device=" appears in train.log
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest -q tests/test_training_runner.py -k logs_selected_device`
Expected: FAIL because the log does not yet include the selected device

- [ ] **Step 3: Write minimal implementation**

```python
# trainer/external_adapter.py
from trainer.device import describe_torch_device, resolve_torch_device

device = resolve_torch_device()
device_name = describe_torch_device(device)
model = build_segmentation_model(model_name).to(device)
log_path.write_text(
    f"starting actual torch training with model={model_name}\n"
    f"device={device_name}\n",
    encoding="utf-8",
)
```

```python
# trainer/external_adapter.py
for images, masks in train_loader:
    images = images.to(device)
    masks = masks.to(device)
```

```python
# app/services/test_inference.py
from trainer.device import resolve_torch_device

device = resolve_torch_device()
model = build_segmentation_model(run.model_name).to(device)
tensor = tensor.to(device)
with torch.no_grad():
    logits = model(tensor)
    prediction = (torch.sigmoid(logits) > 0.5).float()[0, 0].detach().cpu().numpy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest -q tests/test_training_runner.py -k logs_selected_device`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add trainer/external_adapter.py app/services/test_inference.py tests/test_training_runner.py tests/test_test_inference_service.py
git commit -m "feat: use shared device selection for training and test inference"
```

### Task 3: Full Verification

**Files:**
- Verify only: `tests/test_device_selection.py`
- Verify only: `tests/test_training_runner.py`
- Verify only: `tests/test_test_inference_service.py`
- Verify only: entire repository

- [ ] **Step 1: Run focused device tests**

Run: `python3 -m pytest -q tests/test_device_selection.py tests/test_training_runner.py -k "device or logs_selected_device"`
Expected: PASS

- [ ] **Step 2: Run focused inference tests**

Run: `python3 -m pytest -q tests/test_test_inference_service.py`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python3 -m pytest -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add trainer/device.py trainer/external_adapter.py app/services/test_inference.py tests/test_device_selection.py tests/test_training_runner.py tests/test_test_inference_service.py docs/superpowers/specs/2026-05-28-device-selection-design.md docs/superpowers/plans/2026-05-28-device-selection.md
git commit -m "feat: add prioritized device selection"
```
