# Detection → ContinueIf → Email Notification Workflow

This example runs the workflow: **image → object detection → continue_if → email_notification**.

- **Detection**: runs a Roboflow object detection model on the input image.
- **ContinueIf**: condition “at least one bounding box” (e.g. `len(predictions) >= 1`).
- **Email**: if the condition passes, the email notification step runs (e.g. “Found detections for this image”).

Workflow definitions live as JSON files in **`workflows/`**. The script loads a workflow by name and always runs it through **ExecutionEngine.init** (then **run**). The problematic workflow raises **ControlFlowDefinitionError** during init, so you can debug that path interactively.

## Workflow files

| File | Description |
|------|-------------|
| **workflow_with_workaround.json** | Email has `message_parameters.predictions` referencing detection so input lineage matches control-flow lineage; compiles and runs. |
| **workflow_problematic.json** | Email has no data input from detection; **ControlFlowDefinitionError** during ExecutionEngine.init. |

## The “problematic” case and workaround

If the email step has **no** input that references the detection step (e.g. only static `subject`, `message`, `receiver_email`), the workflow compiler raises **`ControlFlowDefinitionError`** during init: the control-flow step (ContinueIf) is based on detection’s lineage, but the email step would have empty input lineage.

**Workaround:** In the workflow JSON, give the email step e.g. `"message_parameters": { "predictions": "$steps.detection.predictions" }` so the compiler sees matching lineage.

## Requirements

- Python 3.10+
- Installed `inference` package (from repo root: `pip install -e .`)
- **ROBOFLOW_API_KEY** in the environment (for detection and, if using Roboflow-managed email, for sending).

## Run the workflow

From the repo root:

```bash
export ROBOFLOW_API_KEY="your-api-key"

# Run the workaround workflow (default)
python examples/detection_continue_if_email/run_workflow.py

# Run a specific workflow by name (filename without .json)
python examples/detection_continue_if_email/run_workflow.py --workflow workflow_with_workaround
python examples/detection_continue_if_email/run_workflow.py --workflow workflow_problematic

# Optional: use your own image
python examples/detection_continue_if_email/run_workflow.py --workflow workflow_with_workaround --image path/to/image.jpg
```

Both workflows go through **ExecutionEngine.init**; the problematic one will raise **ControlFlowDefinitionError** there so you can debug interactively.

## Email: dry-run vs actually sending

The workflows expose a **`dry_run`** input (default `true`) wired to the email step’s **`disable_sink`**. When `dry_run` is true, the email step still runs and produces output (e.g. `error_status`, `message`) but **does not send** any email.

- **Default:** `dry_run: true` → output only, no mail sent.
- **To send:** run with `--send-email` (passes `dry_run: false`), or override `dry_run` in your own runtime parameters.

## Email configuration

- Workflow JSON uses **Roboflow Managed API Key** for email by default. Edit **receiver_email** in the JSON if you want to receive the notification.
- If you don’t set a valid receiver or API key, the email step may log an error but the workflow still runs; detection and ContinueIf are unaffected.

## Output

- The script prints workflow outputs (e.g. detection predictions and whether the email step was reached).
- If the condition fails (no detections), the email step is skipped and the output reflects that.
