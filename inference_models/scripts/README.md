# Flakiness Check Script

`check_prediction_flakiness.py` checks whether model predictions are stable across repeated runs.

For each **workspace** and **project** entry in your config, the script:

- fetches test images from the Roboflow API for that `workspace` + `project`
- runs each listed `model_id` according to `--scenario` (see below)
- compares outputs against the first pass and reports flaky models
- can save model-load stdout for mismatched runs

## Scenarios (`--scenario`)

| Value | Behavior |
|--------|------------|
| `refetch-artifact` (default) | Before **each** iteration: delete `INFERENCE_HOME` model cache, then load the model again (re-fetch / full reload). |
| `same-model-repeated-inference` | Load the model **once**, then run inference **N** times in the **same process** on the same object. |
| `reload-local-artifact` | **Do not** clear the model cache between iterations; each iteration loads a **new** model instance from the **same** on-disk package (re-download only if the cache was empty). |

Use `--iterations` for how many passes to compare (must be >= 2). For `same-model-repeated-inference`, that is one load plus `iterations - 1` additional inference-only passes.

## Prerequisites

- `ROBOFLOW_API_KEY` available (for example in `.env`)
- A model config JSON file (see `model_config.example.json`)

## Model Config Format

Top-level keys are **Roboflow workspace slugs**. Each workspace maps to an object whose keys are **project slugs** and values are lists of model IDs:

```json
{
  "your-workspace-slug": {
    "my-detection-project": ["yolov8n-640", "rfdetr-base"],
    "my-segmentation-project": ["yolov8n-seg-640"]
  }
}
```

## Example Commands

Run from `inference_models`:

**Default (re-fetch artifact each iteration):**

```bash
uv run python scripts/check_prediction_flakiness.py \
  --models-config scripts/model_config.example.json \
  --num-test-images 25 \
  --iterations 3 \
  --report-json scripts/flakiness_report.json
```

**Same process, same loaded model, repeated inference:**

```bash
uv run python scripts/check_prediction_flakiness.py \
  --models-config scripts/model_config.example.json \
  --scenario same-model-repeated-inference \
  --num-test-images 25 \
  --iterations 5
```

**Reload from local cache each iteration (no cache wipe):**

```bash
uv run python scripts/check_prediction_flakiness.py \
  --models-config scripts/model_config.example.json \
  --scenario reload-local-artifact \
  --num-test-images 25 \
  --iterations 3
```

## Useful Options

- `--streams-output-dir`: where mismatch load logs are written
- `--sample-different-images-limit`: max sample paths stored in diff report
- `--roboflow-images-cache-dir`: location of downloaded image cache (per workspace/project)
- `--no-roboflow-image-cache`: force re-download of images
- `--inference-home`: model package cache (`INFERENCE_HOME`); only cleared in `refetch-artifact`
