# Detection visualization (`roboflow_core/detection_visualization@v1`)

One step that draws each detection's box, then its label, on a copy of the
image. It replaces the common pair `bounding_box_visualization@v1` →
`label_visualization@v2` on the same predictions.

- **Parameters:** every `label_visualization@v2` parameter, plus box
  `thickness` (default `2`) and `roundness` (default `0.0`). The palette
  parameters (`color_palette`, `palette_size`, `custom_colors`, `color_axis`)
  color both boxes and labels.
- **Order:** all boxes in row order, then all labels in row order. A box never
  covers a label.
- **Default:** `accurate`. Its pixels equal the box block chained into the
  label block, both accurate.
- **Ownership:** the input image and the predictions are not changed. The
  output is a new image with the input's identity, frame, source, time and
  parent row.
- **`copy_image`** is rejected, as in every V2 annotation block.

## How it runs

```
image ─▶ box implementation ─▶ boxed copy ─▶ label implementation ─▶ output image
          (all rows)                          (all rows)
```

| quality    | boxes                | labels         |
|------------|----------------------|----------------|
| `accurate` | `AccurateBoundingBoxes` | `AccurateLabels` |
| `fast`     | `FastBoundingBoxes`     | `FastLabels`     |

Each step owns its output image. The boxed intermediate is a private copy;
the block does not promise a single image copy.

## Fast quality

The `fast` implementation keeps label texts, row order, palette, text style,
anchors and frame mapping. Its pixels can differ from `accurate`:

- Square boxes are drawn as border bands. Corner joins, borders and pixel
  rounding can rasterize differently from OpenCV. Rounded boxes use the
  accurate renderer.
- Opaque single-line labels are pasted from cached sprites; other labels use
  the reference renderer.

Fast output is not guaranteed to equal accurate output pixel for pixel. It is
not guaranteed to be faster on every input.

## Example

```python
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.engine import CompileOptions, compile_workflow

definition = {
    "version": "2.0",
    "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v3",
            "name": "detector",
            "image": "$inputs.image",
            "model_id": "rfdetr-nano",
        },
        {
            "type": "roboflow_core/detection_visualization@v1",
            "name": "draw",
            "image": "$inputs.image",
            "predictions": "$steps.detector.predictions",
            "text": "Class and Confidence",
            "thickness": 2,
        },
    ],
    "outputs": [
        {"type": "JsonField", "name": "annotated", "selector": "$steps.draw.image"}
    ],
}

# Default: the accurate implementation.
plan = compile_workflow(definition, catalogue=create_catalogue())

# Deployment-wide request for fast rendering.
fast_plan = compile_workflow(
    definition,
    catalogue=create_catalogue(),
    options=CompileOptions(quality="fast"),
)
assert fast_plan.step(("draw",)).implementation.name == "fast"
```

A definition can also ask for a quality itself, which takes precedence over
`CompileOptions.quality`:

```json
"execution": {"quality": "fast", "step_quality": {"$steps.draw": "accurate"}}
```
