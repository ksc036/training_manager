# Training Manager Design

## Goal

Build a standalone web-based training manager that reads approved segmentation samples from `/Users/ksc/Downloads/samBaseannotaion/annotation_complete`, creates reproducible dataset versions, launches model training, tracks run progress, and compares model performance across runs.

The existing annotation service remains unchanged. The new project owns only training, experiment tracking, and comparison workflows.

## Scope

In scope:

- Scan approved annotation folders from the existing annotation project
- Create dataset versions from selected samples
- Store dataset manifests, file hashes, and metadata
- Launch model training from the web UI
- Track run status and epoch-level metrics
- Compare completed runs by dataset version and model configuration
- Train only with pre-registered model definitions supported by the project

Out of scope for the first version:

- Editing annotations
- Writing back into the annotation service folders
- Distributed training orchestration
- Multi-user permissions
- Cloud artifact storage
- User-defined arbitrary model architecture uploads

## Source Data Contract

The source of truth is:

- `/Users/ksc/Downloads/samBaseannotaion/annotation_complete`

Each approved sample is expected to follow this structure:

```text
annotation_complete/
  <sample_folder>/
    Image/
      <image file>
    mask/
      <mask file>
```

The training manager treats this source as read-only.

## Architecture

The system is a standalone project with its own web UI, SQLite database, and filesystem-managed artifacts.

High-level flow:

1. Scan `annotation_complete`
2. Select approved samples
3. Create dataset version manifest
4. Prepare training workspace from the manifest
5. Launch a training run
6. Persist config, logs, checkpoints, and metrics
7. Compare completed runs in the UI

## Snapshot Strategy

The project uses a hybrid dataset versioning model:

- Keep `annotation_complete` as the original data pool
- Create a dataset version manifest that lists the selected samples
- Record image and mask hashes for reproducibility
- Build a link-based workspace for normal training
- Allow explicit physical copies for user-selected frozen dataset versions

This approach avoids unnecessary data duplication while preserving reproducibility and traceability.

## Storage Model

Use SQLite for metadata and the filesystem for large artifacts.

SQLite stores:

- Dataset versions
- Dataset sample membership
- File hashes and sample metadata
- Training runs
- Epoch-level metrics
- Run status history and summary metrics

Filesystem stores:

- Dataset manifest JSON files
- Link-based or copied workspaces
- Training config files
- Logs
- Checkpoints
- Comparison exports if needed

## Proposed Project Structure

```text
training_manager/
  app/
    web/
    services/
    repositories/
    models/
  trainer/
    scanner/
    datasets/
    runners/
  data/
    datasets/
      manifests/
      workspaces/
      snapshots/
    runs/
      <run_id>/
        config.json
        metrics.csv
        logs/
        checkpoints/
    artifacts/
  db/
    training_manager.sqlite
  docs/
    superpowers/
      specs/
      plans/
```

## Database Design

### `dataset_versions`

Purpose:
Store each frozen dataset version definition.

Fields:

- `id`
- `name`
- `version`
- `created_at`
- `source_root`
- `snapshot_mode`
- `sample_count`
- `manifest_path`
- `notes`
- `tags_json`

### `dataset_samples`

Purpose:
Store sample membership for each dataset version.

Fields:

- `id`
- `dataset_version_id`
- `sample_id`
- `sample_path`
- `image_path`
- `mask_path`
- `image_hash`
- `mask_hash`
- `width`
- `height`

### `training_runs`

Purpose:
Store each training execution.

Fields:

- `id`
- `dataset_version_id`
- `run_name`
- `model_name`
- `encoder_name`
- `status`
- `started_at`
- `ended_at`
- `best_epoch`
- `best_checkpoint_path`
- `run_dir`
- `config_path`
- `log_path`
- `notes`

### `run_metrics`

Purpose:
Store epoch-level metrics for each run.

Fields:

- `id`
- `training_run_id`
- `epoch`
- `train_loss`
- `val_loss`
- `dice`
- `iou`
- `precision`
- `recall`
- `created_at`

### `run_events`

Purpose:
Store status transitions and notable lifecycle events.

Fields:

- `id`
- `training_run_id`
- `event_type`
- `message`
- `created_at`

## Main UI

The first version has four primary screens.

### 1. Datasets

Responsibilities:

- Scan `annotation_complete`
- Show available approved samples
- Show validation issues for malformed samples
- Let the user select samples
- Create a dataset version
- Store a manifest and hash metadata

Key actions:

- `Scan source`
- `Create dataset version`
- `View manifest`
- `Freeze snapshot`

### 2. Train

Responsibilities:

- Choose a dataset version
- Select a pre-registered model and its training configuration
- Start training

Initial configuration fields:

- Dataset version
- Model type
- Encoder or backbone
- Epochs
- Batch size
- Learning rate
- Validation split
- Random seed
- Notes

Key actions:

- `Start training`

### 3. Runs

Responsibilities:

- Show queued, running, completed, failed, and stopped runs
- Show epoch progress and best known metric
- Open logs and checkpoints
- Stop or retry runs

Key columns:

- Run name
- Dataset version
- Model name
- Status
- Current epoch
- Best Dice
- Best IoU
- Start time
- End time

### 4. Compare

Responsibilities:

- Compare multiple completed runs
- Filter by dataset version or model
- Show metric trends and summaries

Key comparisons:

- Same dataset, different models
- Same model, different dataset versions
- Best epoch comparison
- Final metric comparison

## Backend Components

### Dataset Scanner

Responsibilities:

- Walk the approved annotation root
- Identify valid sample folders
- Validate presence of `Image` and `mask`
- Resolve image and mask file paths
- Read width and height
- Compute stable hashes

Outputs:

- Structured sample records
- Validation issue list

### Dataset Registry Service

Responsibilities:

- Create dataset versions from selected sample IDs
- Persist the dataset version row and sample membership rows
- Write manifest JSON
- Build workspace links
- Optionally create a physical snapshot copy

### Training Runner

Responsibilities:

- Create run directories
- Write run config
- Launch the training process in the background
- Attach the run to a dataset version
- Store checkpoint and metrics outputs
- Resolve the selected model from the project's pre-registered model catalog

First-version execution model:

- One training process launched from the web backend per run
- No external queue required
- Status is polled or refreshed from DB and run files

## Model Support Policy

The first version supports only pre-registered models that are implemented and shipped inside the training manager project.

Rules:

- Users choose from a fixed model list exposed by the Train screen
- Each supported model has a stable internal key and a known config shape
- The backend validates that the selected model key belongs to the registered model catalog
- The system does not accept arbitrary Python model uploads or free-form architecture definitions from the web UI

This keeps the first release predictable, easier to test, and easier to compare across runs.

### Run Tracker

Responsibilities:

- Update run status
- Record epoch metrics
- Store best checkpoint information
- Mark runs as completed, failed, or stopped

### Comparison Service

Responsibilities:

- Query comparable runs
- Aggregate summary metrics
- Serve chart-ready metric series

## Run Lifecycle

Run states:

- `queued`
- `running`
- `completed`
- `failed`
- `stopped`

Rules:

- Every run must reference exactly one dataset version
- Every run must write config, metrics, and logs into its own run directory
- Every completed run should record its best checkpoint path
- Failed and stopped runs must remain visible for auditability

## Reproducibility Rules

- The training manager never mutates the source annotation data
- Dataset versions are the only allowed starting point for training
- Each dataset version records sample membership and file hashes
- Each run records the exact training config used
- Comparison views always show which dataset version a run used

## Error Handling

### Invalid Samples

If a sample folder is malformed:

- Exclude it from dataset version creation
- Show the exact reason in the Datasets screen
- Keep scanning the rest of the source tree

### Hash Drift

If source file hashes no longer match a dataset manifest:

- Raise a warning on the dataset version
- Prevent silent assumptions that the source is unchanged

### Training Failure

If training exits unexpectedly:

- Mark the run as `failed`
- Persist the error message
- Preserve the log path and partial artifacts

### Manual Stop

If a user stops a run:

- Mark the run as `stopped`
- Keep completed partial metrics
- Do not classify it as a failure

## Testing Strategy

### Dataset Scanner Tests

- Valid sample detection
- Missing `Image` or `mask` handling
- Hash calculation correctness
- Width and height extraction

### Dataset Registry Tests

- Dataset version creation
- Manifest JSON consistency
- DB rows matching manifest content
- Snapshot mode behavior

### Training Runner Tests

- Run directory creation
- Config file generation
- Status transitions
- Metrics ingestion

### Comparison Tests

- Filtering by dataset version
- Filtering by model
- Best metric ranking
- Epoch-series retrieval

## Recommended Initial Tech Choices

- Python backend
- FastAPI server with server-rendered templates and JSON endpoints
- SQLite
- JSON manifest files
- Filesystem-based checkpoints and logs

The first version should favor simplicity and local operability over framework complexity.

## Non-Goals for the First Delivery

- User accounts
- Concurrent multi-node scheduling
- Automatic hyperparameter sweeps
- Remote storage backends
- Full experiment visualization beyond essential comparison charts

## Success Criteria

The first version is successful if the user can:

1. Scan approved samples from the existing annotation project
2. Create a dataset version with stored manifest and hashes
3. Start a training run from the web UI
4. Observe run progress and final status
5. Compare completed runs by dataset version and model configuration

## Assumptions

- The new project root will be `/Users/ksc/Downloads/training_manager`
- Approved sample folders continue to follow the current `Image` and `mask` layout
- A local training workflow is sufficient for the first release
- SQLite is adequate for the expected scale of the first version
