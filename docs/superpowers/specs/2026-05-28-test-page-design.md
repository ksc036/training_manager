# Test Page Design

## Summary

Add a new `/test` page for visual validation of trained segmentation runs.

The page will let the user:

- choose a completed training run by run name
- browse only samples from the dataset version that run trained on
- run inference with that run's `best.ckpt`
- view four side-by-side outputs:
  - original image
  - ground-truth mask
  - predicted mask
  - original-plus-prediction overlay

This page is for qualitative inspection of whether a trained model learned the target masks well.

## Goals

- Make it easy to inspect a trained run without leaving the web app.
- Ensure the selected run and selected sample are consistent with the dataset the run actually trained on.
- Reuse the existing model registry, checkpoint format, and dataset manifest flow.
- Keep the first version simple by doing inference on demand instead of building a cached preview system.

## Non-Goals

- Batch evaluation across an entire dataset version
- Storing prediction previews permanently
- Free-form image upload outside the selected dataset version
- New training or checkpoint formats
- Quantitative benchmark dashboards beyond what already exists in `Runs` and `Compare`

## User Flow

1. User opens `/test`.
2. The page shows completed runs that have a readable `best.ckpt`.
3. User selects one run.
4. The page loads samples only from that run's `dataset_version_id`.
5. User selects one sample.
6. User submits the form.
7. The server:
   - loads the run config and best checkpoint
   - rebuilds the matching model structure from `model_name`
   - loads the sample image and ground-truth mask
   - applies the same inference preprocessing used by training
   - runs prediction
   - upsamples the prediction back to the original image size for display
8. The page renders four visual panels plus lightweight metadata.

## UI Design

Add a new `Test` link to the main navigation.

The `/test` page contains:

- a run selector
- a sample selector
- a `Run Test` button
- a result section rendered only after a successful inference request

The result section shows:

- Original
- Ground Truth
- Prediction
- Overlay

Supplementary text shown above or below the panels:

- run name
- model name
- encoder name
- dataset label
- sample id
- checkpoint path

## Data Sources

### Runs

Source: existing run directories and `RunService`.

Eligibility rules:

- run status should be `completed`
- `checkpoints/best.ckpt` must exist
- `model_name` must be supported by `build_segmentation_model(...)`

### Samples

Source: the dataset manifest referenced by the selected run's `dataset_version_id`.

Only samples from that dataset version are selectable on the test page.

Each sample must provide:

- `sample_id`
- `image_path`
- `mask_path`

## Inference Design

Create a new service module:

- `app/services/test_inference.py`

Responsibilities:

- list testable runs
- resolve the dataset samples for a selected run
- load a run checkpoint
- rebuild the model from `model_name`
- run inference on one sample
- generate display-ready images for the four output panels

### Preprocessing

Use the same basic inference input handling as the current training adapter:

- image converted to `RGB`
- mask converted to grayscale
- image resized to `64x64`
- mask resized to `64x64` only for metric-aligned internal handling if needed
- image normalized to `0..1`
- image tensor converted to `CHW`

The first version intentionally keeps the current training-time resize behavior so the test page reflects what the current model actually saw during training.

### Prediction

The selected model produces logits shaped like `[B, 1, H, W]`.

Inference flow:

- forward pass
- sigmoid
- threshold at `0.5`
- binary prediction mask

### Display Rendering

Display is based on the original image size, not the internal `64x64` inference size.

Steps:

- load original image at original resolution
- resize predicted binary mask back to the original image size
- load the ground-truth mask at original display size
- build a colored overlay from the prediction mask and original image

The page will embed generated preview images using response bytes encoded for HTML rendering.

## Routes And Templates

Add:

- `app/web/routes_test.py`
- `app/web/templates/test.html`

Route behavior:

- `GET /test`
  - render the page
  - show run selector
  - optionally show sample selector if a run is already selected via query param
- `POST /test`
  - validate run and sample selection
  - run inference
  - return page with generated comparison visuals

Update:

- `app/main.py` to register the router
- `app/web/templates/base.html` to add the navigation link

## Error Handling

The page must fail locally and explain the issue without crashing the app.

Handled cases:

- selected run not found
- run exists but has no `best.ckpt`
- unsupported `model_name`
- checkpoint cannot be loaded
- sample is not part of the selected run's dataset version
- image or mask file missing
- image or mask unreadable
- inference exception

User-facing behavior:

- keep the page visible
- show a short error message near the form
- do not render stale prior results for a failed request

## Testing

Add route and service coverage for:

- `/test` page renders
- completed run with checkpoint appears as selectable
- sample list is restricted to the selected run's dataset version
- invalid run selection returns a safe error response
- invalid sample selection returns a safe error response
- inference request returns four rendered outputs
- checkpoint loading uses the run's stored `model_name`

Use small synthetic images and masks so tests stay fast.

## Implementation Notes

- The first version should support the current real checkpoint format created by `torch.save(...)` in the external training path.
- Simulated JSON checkpoints are not a priority for `/test`; they may be excluded from the selectable run list if they cannot be loaded as real model weights.
- The first version should optimize for correctness and clarity, not caching or performance.
