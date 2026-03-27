# Flakiness Check Script

`check_prediction_flakiness.py` checks whether model predictions are stable across repeated model reloads.

For each **workspace** and **project** entry in your config, the script:

- fetches test images from the Roboflow API for that `workspace` + `project`
- runs each listed `model_id` for `n` iterations
- clears the model cache before each iteration so the model is loaded fresh
- compares outputs from each iteration against iteration 1
- reports flaky models and saves model-load streams for mismatched runs

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

## Example Command

Run from `inference_models`:

```bash
uv run python scripts/check_prediction_flakiness.py \
  --models-config scripts/model_config.example.json \
  --num-test-images 25 \
  --iterations 3 \
  --report-json scripts/flakiness_report.json
```

## Useful Options

- `--streams-output-dir`: where mismatch load logs are written
- `--sample-different-images-limit`: max sample paths stored in diff report
- `--roboflow-images-cache-dir`: location of downloaded image cache (per workspace/project)
- `--no-roboflow-image-cache`: force re-download of images
