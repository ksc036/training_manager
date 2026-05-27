# External Trainer Adapter

The training manager now supports an `external-script` backend in addition to the built-in `simulated` backend.

## How It Works

- Set `TRAINING_MANAGER_EXTERNAL_TRAINER_COMMAND`
- The `Train` page will expose a pre-registered model named `External Script Model`
- Starting that model writes `trainer_backend="external-script"` and `backend_config` into `config.json`
- The runner executes the configured command and appends the run directory path as the final argument

## Required Contract

Your external trainer command must accept:

```text
<entry command> <run_dir>
```

Example:

```bash
export TRAINING_MANAGER_EXTERNAL_TRAINER_COMMAND="python -m trainer.worker"
```

When executed, the external trainer is expected to work inside the given `run_dir` and produce the same core artifacts used by the UI:

- `config.json`
- `status.json`
- `metrics.csv`
- `train.log`

If the external trainer also updates SQLite run status and metrics, the `Runs` and `Compare` pages will immediately reflect those results.
