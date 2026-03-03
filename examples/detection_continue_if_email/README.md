# Detection → ContinueIf → Email Notification Workflow

This example runs the workflow: **image → object detection → continue_if → email_notification**.

- **Detection**: runs a Roboflow object detection model on the input image.
- **ContinueIf**: condition “at least one bounding box” (e.g. `len(predictions) >= 1`).
- **Email**: if the condition passes, the email notification step runs (e.g. “Found detections for this image”).

## The “problematic” case and workaround

If the email step has **no** input that references the detection step (e.g. only static `subject`, `message`, `receiver_email`), the workflow compiler raises **`ControlFlowDefinitionError`**: the control-flow step (ContinueIf) is based on detection’s lineage, but the email step would have empty input lineage, so the compiler rejects it.

**Workaround:** Give the email step at least one input that carries the same lineage as the control flow, e.g.:

```json
"message_parameters": {
  "predictions": "$steps.detection.predictions"
}
```

The example uses this workaround so the workflow compiles and runs. You can still use the message body to describe whether detections were found (e.g. via `{{ $parameters.predictions }}` in the message template).

## Requirements

- Python 3.10+
- Installed `inference` package (from repo root: `pip install -e .`)
- **ROBOFLOW_API_KEY** in the environment (for detection and, if using Roboflow-managed email, for sending).

## Run the workflow

From the repo root:

```bash
export ROBOFLOW_API_KEY="your-api-key"

# Run with default image (small placeholder) or pass a path/URL
python examples/detection_continue_if_email/run_workflow.py

# Optional: use your own image
python examples/detection_continue_if_email/run_workflow.py --image path/to/image.jpg

# Optional: demonstrate the compilation error (no email workaround)
python examples/detection_continue_if_email/run_workflow.py --demonstrate-error
```

## Email configuration

- The script uses **Roboflow Managed API Key** for email by default (no SMTP setup).
- Set **receiver_email** in the workflow or script to an address you control if you want to receive the notification.
- If you don’t set a valid receiver or API key, the email step may log an error but the workflow still runs; detection and ContinueIf are unaffected.

## Output

- The script prints workflow outputs (e.g. detection predictions and whether the email step was reached).
- If the condition fails (no detections), the email step is skipped and the output reflects that.
