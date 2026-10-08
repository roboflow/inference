# Action Recognition Visualization

Connect a camera or WebRTC image to **Action Recognition Model**, then connect
that same image and the model's **window** output to **Action Recognition
Visualization**. Return the visualizer's **image** output to display or record
it. This needs no custom Python and draws directly into the output image.

## Choose the display

- **Compact** (default): labels from the last analyzed window, its bounds and
  the displayed frame position. Useful for small previews.
- **Timeline**: also connect the model's **timeline** output. The block draws
  accumulated intervals clipped to the last analyzed window, one row per class.
  A separate grey track marks frames after that window with no new result.

`max_classes` defaults to 4. Additional classes have an explicit overflow count;
compact mode also limits labels to available width. Long names are shortened to
fit. Text uses the existing OpenCV renderer (non-ASCII characters render as `?`).
The input image is copied by default; dimensions and video metadata are retained.
NumPy and tensor-backed images use the same NumPy drawing path.

For an existing `actions` model step, add:

```json
{
  "type": "roboflow_core/action_recognition_visualization@v1",
  "name": "visualize",
  "image": "$inputs.image",
  "window": "$steps.actions.window",
  "mode": "timeline",
  "timeline": "$steps.actions.timeline"
}
```

Choose `"mode": "compact"` and omit `timeline` for the compact overlay. The
output selector is `$steps.visualize.image`. Both connections must come from
the same model/stream; do not mix a window from one source with another source's
history. The visualizer checks image/window video identity, but the existing
timeline payload carries no source identifier.

## What the overlay means

Labels represent **all actions reported in the last model window**, held until
another analysis. They do not identify the action at the current video frame.
The artificial `1.0` confidence from `latest_predictions` is never displayed.

The model publishes a small `window` dictionary:

| Field | Meaning |
| --- | --- |
| `status` | `collecting`, `ready`, or `error`; held between model calls |
| `classes` | All filtered class names from the latest successful call |
| `start_frame`, `end_frame` | Inclusive first/last sampled source frame of that call |
| `fps` | Declared source FPS, or null when the model had to assume a rate |
| `video_identifier` | Source identity, matched against the displayed image |

The first call may use partial context, so bounds reflect the actual samples,
not a window reconstructed from stride. The `~` prefix marks approximate
frame/FPS times; they are not exact PTS or inference latency. Without declared
FPS, `f360` means source frame 360. The overlay distinguishes collecting context,
a successful window with no reported classes (under the class filter), and
analysis failure. A failure hides labels and bars until another successful
analysis, even on subsequent frames where the older `error_status` output is
empty. The original `error_status` behavior is unchanged.

The timeline is accumulated model history, not the raw segments of one call.
Intervals can overlap and merge: bars are not event identities or counts of
individual movements. Empty successful results show the empty state rather
than old historical bars. Rendering holds no temporal state and does not
extend intervals, buffer replay frames, or run inference.

The model owns window length and sampling; stride controls analysis cadence.
Inference remains synchronous. Holding a label does not establish live-video
throughput or real-model accuracy, and the model still does not run an extra
analysis automatically at the end of a stream.

The [conceptual API](../api/overview.md) defines the ownership of analysis
state and the invariants shared by the model and visualization.
