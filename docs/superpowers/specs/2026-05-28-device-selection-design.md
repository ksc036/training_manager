# Device Selection Design

## Summary

Add a shared device-selection helper used by both training and test inference.

Selection priority:

- `cuda`
- `mps`
- `cpu`

The selected device must be applied consistently during real training and `/test` page inference. The training log must clearly record which device/environment was selected so the run detail page shows it through the existing log viewer.

## Goals

- Use the best available PyTorch execution backend automatically.
- Keep device selection logic in one place.
- Record the chosen device in the training log for later inspection.
- Reuse the same logic in `/test` inference so train and test behavior are aligned.

## Non-Goals

- Manual device override from the UI
- Multi-GPU training
- Distributed training
- Per-run device scheduling

## Design

Create a new helper module:

- `trainer/device.py`

It exposes:

- `resolve_torch_device()`
- `describe_torch_device()`

Behavior:

- If `torch.cuda.is_available()` is true, return `torch.device("cuda")`
- Else if `torch.backends.mps.is_available()` is true, return `torch.device("mps")`
- Else return `torch.device("cpu")`

The helper also returns a human-readable description for logging, such as:

- `cuda`
- `mps`
- `cpu`

## Training Integration

Update `trainer/external_adapter.py`:

- resolve device once at the start of real training
- move model to that device
- move input tensors and masks to that device inside train/val loops
- write the selected device to `train.log` near the start of the run

Expected log shape:

- `starting actual torch training with model=unet`
- `device=mps`

or:

- `starting actual torch training with model=unet`
- `device=cuda`

## Test Inference Integration

Update `app/services/test_inference.py`:

- use the same shared helper
- move model and inference tensor to the selected device
- keep final prediction returned on CPU for image rendering

## Error Handling

- If device helper returns `cpu`, training and inference continue normally
- If a selected accelerator fails during `.to(device)` or forward pass, surface the exception as the existing run failure or `/test` page error rather than silently falling back mid-run

## Testing

Add tests for:

- helper priority behavior
- training log contains device info
- real training still writes metrics and checkpoint with device selection enabled
- `/test` inference still renders results with the shared helper in place
