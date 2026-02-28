# Workflow Builder Reference

## Workflow JSON Schema

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "<name>"},
    {"type": "WorkflowParameter", "name": "<name>", "default_value": <value>}
  ],
  "steps": [
    {
      "type": "<block_type_identifier>",
      "name": "<unique_step_name>",
      "<property>": "<value_or_selector>"
    }
  ],
  "outputs": [
    {
      "type": "JsonField",
      "name": "<output_name>",
      "selector": "$steps.<step_name>.<output_name>",
      "coordinates_system": "own"
    }
  ]
}
```

### Input Types

| Type | Description | Example |
|------|-------------|---------|
| `WorkflowImage` | Image input (file, URL, or base64) | `{"type": "WorkflowImage", "name": "image"}` |
| `WorkflowParameter` | Runtime parameter | `{"type": "WorkflowParameter", "name": "conf", "default_value": 0.4}` |

### Selector Syntax

| Pattern | Meaning | Example |
|---------|---------|---------|
| `$inputs.<name>` | Reference a workflow input | `$inputs.image` |
| `$steps.<step>.<output>` | Reference a step output | `$steps.detection.predictions` |
| `$steps.<step>.*` | All outputs of a step | `$steps.detection.*` |

### Output Fields

| Property | Required | Description |
|----------|----------|-------------|
| `type` | Yes | Always `"JsonField"` |
| `name` | Yes | Name in the response |
| `selector` | Yes | Selector pointing to step output |
| `coordinates_system` | No | `"own"` for original image coords |

---

## Block Quick Reference

### Models — Roboflow (use with user's trained models or pretrained)

#### Object Detection (`roboflow_core/roboflow_object_detection_model@v2`)
```json
{
  "type": "roboflow_core/roboflow_object_detection_model@v2",
  "name": "detection",
  "images": "$inputs.image",
  "model_id": "yolov8n-640",
  "confidence": 0.4,
  "iou_threshold": 0.3,
  "class_filter": ["dog", "cat"],
  "max_detections": 300
}
```
**Outputs:** `predictions` (object_detection_prediction), `model_id`, `inference_id`

#### Single-Label Classification (`roboflow_core/roboflow_classification_model@v2`)
```json
{
  "type": "roboflow_core/roboflow_classification_model@v2",
  "name": "classification",
  "images": "$inputs.image",
  "model_id": "my-classifier/1",
  "confidence": 0.4
}
```
**Outputs:** `predictions` (classification_prediction), `model_id`, `inference_id`

#### Multi-Label Classification (`roboflow_core/roboflow_multi_label_classification_model@v2`)
```json
{
  "type": "roboflow_core/roboflow_multi_label_classification_model@v2",
  "name": "multi_classification",
  "images": "$inputs.image",
  "model_id": "my-multilabel/1",
  "confidence": 0.4
}
```
**Outputs:** `predictions` (classification_prediction), `model_id`, `inference_id`

#### Instance Segmentation (`roboflow_core/roboflow_instance_segmentation_model@v2`)
```json
{
  "type": "roboflow_core/roboflow_instance_segmentation_model@v2",
  "name": "segmentation",
  "images": "$inputs.image",
  "model_id": "my-segmentation/1",
  "confidence": 0.4,
  "iou_threshold": 0.3
}
```
**Outputs:** `predictions` (instance_segmentation_prediction), `model_id`, `inference_id`

#### Keypoint Detection (`roboflow_core/roboflow_keypoint_detection_model@v2`)
```json
{
  "type": "roboflow_core/roboflow_keypoint_detection_model@v2",
  "name": "keypoints",
  "images": "$inputs.image",
  "model_id": "my-keypoints/1",
  "confidence": 0.4
}
```
**Outputs:** `predictions` (keypoint_detection_prediction), `model_id`, `inference_id`

---

### Models — Foundation / Zero-Shot

#### Anthropic Claude (`roboflow_core/anthropic_claude@v3`)
```json
{
  "type": "roboflow_core/anthropic_claude@v3",
  "name": "claude",
  "images": "$inputs.image",
  "task_type": "visual-question-answering",
  "prompt": "What objects are in this image?",
  "api_key": "$inputs.api_key",
  "model_version": "claude-sonnet-4-5"
}
```
**Task types:** unconstrained, ocr, structured-answering, classification, multi-label-classification, visual-question-answering, caption, detailed-caption, object-detection
**Outputs:** `output` (string | language_model_output), `classes` (list_of_values)

#### Google Gemini (`roboflow_core/google_gemini@v3`)
Same interface as Claude. `model_version` options: gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash, etc.

#### OpenAI (`roboflow_core/openai@v3`)
Same interface as Claude. `model_version` options: gpt-4.1, gpt-4o, gpt-4o-mini, o4-mini, etc.

#### YOLO-World (`roboflow_core/yolo_world_model@v2`)
Zero-shot detection — detects arbitrary classes without training.
```json
{
  "type": "roboflow_core/yolo_world_model@v2",
  "name": "yolo_world",
  "images": "$inputs.image",
  "class_names": ["hardhat", "safety vest", "person"],
  "confidence": 0.005
}
```
**Outputs:** `predictions` (object_detection_prediction)

#### Florence-2 (`roboflow_core/florence_2@v1`)
Multi-task vision model.
```json
{
  "type": "roboflow_core/florence_2@v1",
  "name": "florence",
  "images": "$inputs.image",
  "task_type": "object-detection",
  "classes": ["car", "person"]
}
```

#### CLIP Comparison (`roboflow_core/clip_comparison@v2`)
Compare images against text descriptions.
```json
{
  "type": "roboflow_core/clip_comparison@v2",
  "name": "clip",
  "images": "$inputs.image",
  "texts": "$inputs.descriptions"
}
```
**Outputs:** `similarity` (list_of_values)

#### OCR (`roboflow_core/ocr_model@v2`)
```json
{
  "type": "roboflow_core/ocr_model@v2",
  "name": "ocr",
  "image": "$inputs.image"
}
```
**Outputs:** `result` (string), `parent_id` (string)

---

### Transformations

#### Dynamic Crop (`roboflow_core/dynamic_crop@v1`)
Crops image regions based on detection bounding boxes. **Increases dimensionality** — 1 image with N detections → N cropped images.
```json
{
  "type": "roboflow_core/dynamic_crop@v1",
  "name": "cropping",
  "images": "$inputs.image",
  "predictions": "$steps.detection.predictions"
}
```
**Outputs:** `crops` (image), `predictions` (object_detection_prediction)

#### Detections Filter (`roboflow_core/detections_filter@v1`)
Filter detections by conditions (class, confidence, size, etc.)
```json
{
  "type": "roboflow_core/detections_filter@v1",
  "name": "filter",
  "predictions": "$steps.detection.predictions",
  "operations_parameters": {},
  "filter_definition": {
    "type": "StatementGroup",
    "statements": [
      {
        "type": "BinaryStatement",
        "left_operand": {"type": "DynamicOperand", "operand_name": "class_name"},
        "comparator": {"type": "(String) =="},
        "right_operand": {"type": "StaticOperand", "value": "dog"}
      }
    ]
  }
}
```
**Outputs:** `predictions` (same kind as input)

#### Detections Transformation (`roboflow_core/detections_transformation@v1`)
Transform detection properties (offset boxes, rename classes, filter, sort, etc.)
```json
{
  "type": "roboflow_core/detections_transformation@v1",
  "name": "transform",
  "predictions": "$steps.detection.predictions",
  "operations": [
    {"type": "DetectionsOffset", "offset_x": 10, "offset_y": 10},
    {"type": "DetectionsRename", "class_map": {"old_name": "new_name"}}
  ]
}
```
**Outputs:** `predictions` (same kind as input)

#### Image Slicer (`roboflow_core/image_slicer@v1`)
Slice image into overlapping tiles for SAHI. **Increases dimensionality.**
```json
{
  "type": "roboflow_core/image_slicer@v1",
  "name": "slicer",
  "image": "$inputs.image",
  "slice_width": 640,
  "slice_height": 640,
  "overlap_ratio_width": 0.2,
  "overlap_ratio_height": 0.2
}
```
**Outputs:** `slices` (image)

#### Detections Stitch (`roboflow_core/detections_stitch@v1`)
Re-assemble sliced detections back onto original image. **Decreases dimensionality.**
```json
{
  "type": "roboflow_core/detections_stitch@v1",
  "name": "stitch",
  "reference_image": "$inputs.image",
  "predictions": "$steps.detection.predictions",
  "overlap_filtering_strategy": "nms"
}
```
**Outputs:** `predictions` (object_detection_prediction)

#### Perspective Correction (`roboflow_core/perspective_correction@v1`)
Correct perspective based on keypoints or corners.

---

### Visualizations

All visualization blocks take `image` and `predictions`, output `image`.

| Block | Type Identifier | Key Config |
|-------|----------------|------------|
| Bounding Box | `roboflow_core/bounding_box_visualization@v1` | `thickness`, `roundness` |
| Label | `roboflow_core/label_visualization@v1` | `text` (template), `text_scale`, `text_color` |
| Mask | `roboflow_core/mask_visualization@v1` | `opacity` |
| Halo | `roboflow_core/halo_visualization@v1` | `opacity`, `kernel_size` |
| Corner | `roboflow_core/corner_visualization@v1` | `corner_length`, `thickness` |
| Ellipse | `roboflow_core/ellipse_visualization@v1` | `thickness` |
| Triangle | `roboflow_core/triangle_visualization@v1` | `base`, `height` |
| Dot | `roboflow_core/dot_visualization@v1` | `radius` |
| Trace | `roboflow_core/trace_visualization@v1` | `trace_length` |
| Polygon Zone | `roboflow_core/polygon_zone_visualization@v1` | `zone`, `opacity` |
| Circle | `roboflow_core/circle_visualization@v1` | `thickness` |
| Color | `roboflow_core/color_visualization@v1` | `opacity` |
| Pixelate | `roboflow_core/pixelate_visualization@v1` | `pixel_size` |
| Blur | `roboflow_core/blur_visualization@v1` | `kernel_size` |
| Background Color | `roboflow_core/background_color_visualization@v1` | `color`, `opacity` |
| Classification Label | `roboflow_core/classification_label_visualization@v1` | `text`, `text_scale` |
| Crop | `roboflow_core/crop_visualization@v1` | `position`, `scale_factor` |
| Model Comparison | `roboflow_core/model_comparison_visualization@v1` | `predictions_a`, `predictions_b` |

**Common visualization properties** (inherited by all):
- `image`: input image (image kind)
- `predictions`: detection predictions
- `copy_image`: bool (default true) — whether to copy image before drawing
- `color_palette`: "DEFAULT", "ROBOFLOW", "Matplotlib Viridis", etc.
- `color_axis`: "INDEX", "CLASS", "TRACK"

**Chaining visualizations**: Pass the output `image` of one viz block as the `image` input of the next:
```json
{"type": "roboflow_core/bounding_box_visualization@v1", "name": "bbox", "image": "$inputs.image", "predictions": "$steps.det.predictions"},
{"type": "roboflow_core/label_visualization@v1", "name": "label", "image": "$steps.bbox.image", "predictions": "$steps.det.predictions"}
```

---

### Formatters / Parsers

#### VLM as Classifier (`roboflow_core/vlm_as_classifier@v2`)
Parse VLM output into classification predictions.
```json
{
  "type": "roboflow_core/vlm_as_classifier@v2",
  "name": "parser",
  "image": "$inputs.image",
  "vlm_output": "$steps.claude.output",
  "classes": "$steps.claude.classes"
}
```
**Outputs:** `predictions` (classification_prediction), `error_status` (boolean)

#### VLM as Detector (`roboflow_core/vlm_as_detector@v2`)
Parse VLM output into object detection predictions.
```json
{
  "type": "roboflow_core/vlm_as_detector@v2",
  "name": "parser",
  "image": "$inputs.image",
  "vlm_output": "$steps.claude.output",
  "classes": "$steps.claude.classes",
  "model_type": "anthropic-claude",
  "task_type": "object-detection"
}
```
**Outputs:** `predictions` (object_detection_prediction), `error_status` (boolean)

#### JSON Parser (`roboflow_core/json_parser@v1`)
Parse JSON strings from VLM output.
```json
{
  "type": "roboflow_core/json_parser@v1",
  "name": "json_parser",
  "raw_json": "$steps.vlm.output",
  "expected_fields": ["field1", "field2"]
}
```

#### Expression (`roboflow_core/expression@v1`)
Evaluate conditions and return values (switch/case logic).
```json
{
  "type": "roboflow_core/expression@v1",
  "name": "counter",
  "data": {"predictions": "$steps.detection.predictions"},
  "data_operations": {
    "predictions": [{"type": "SequenceLength"}]
  },
  "switch": {
    "type": "CasesDefinition",
    "cases": [],
    "default": {"type": "DynamicOperand", "operand_name": "predictions"}
  }
}
```
**Outputs:** `output` (wildcard)

#### Property Definition (`roboflow_core/property_definition@v1`)
Extract properties from data using operations.
```json
{
  "type": "roboflow_core/property_definition@v1",
  "name": "class_names",
  "data": "$steps.detection.predictions",
  "operations": [
    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
  ]
}
```
**Outputs:** `output` (wildcard)

#### CSV Formatter (`roboflow_core/csv_formatter@v1`)
Format data as CSV rows.

---

### Flow Control

#### Continue If (`roboflow_core/continue_if@v1`)
Conditionally execute downstream steps.
```json
{
  "type": "roboflow_core/continue_if@v1",
  "name": "check",
  "condition_statement": {
    "type": "StatementGroup",
    "statements": [{
      "type": "BinaryStatement",
      "left_operand": {
        "type": "DynamicOperand",
        "operand_name": "count",
        "operations": [{"type": "SequenceLength"}]
      },
      "comparator": {"type": "(Number) >"},
      "right_operand": {"type": "StaticOperand", "value": 0}
    }]
  },
  "evaluation_parameters": {"count": "$steps.detection.predictions"},
  "next_steps": ["$steps.downstream_step"]
}
```

---

### Sinks

#### Roboflow Dataset Upload (`roboflow_core/roboflow_dataset_upload@v2`)
Upload images and annotations to a Roboflow dataset.

#### Email Notification (`roboflow_core/email_notification@v1`)
Send email alerts.

#### Webhook Sink (`roboflow_core/webhook_sink@v1`)
POST results to a webhook URL.

---

### Classical CV

| Block | Type Identifier | Purpose |
|-------|----------------|---------|
| Image Blur | `roboflow_core/image_blur@v1` | Blur an image |
| Threshold | `roboflow_core/threshold@v1` | Binary thresholding |
| Grayscale | `roboflow_core/convert_grayscale@v1` | Convert to grayscale |
| Contours | `roboflow_core/image_contours@v1` | Find contours |
| Dominant Color | `roboflow_core/dominant_color@v1` | Find dominant color |
| Template Matching | `roboflow_core/template_matching@v1` | Match a template |
| SIFT | `roboflow_core/sift@v1` | SIFT features |
| SIFT Comparison | `roboflow_core/sift_comparison@v2` | Compare SIFT features |
| Distance Measurement | `roboflow_core/distance_measurement@v1` | Measure pixel distances |
| Size Measurement | `roboflow_core/size_measurement@v1` | Measure object sizes |

---

### Analytics (for video/streaming)

| Block | Type Identifier | Purpose |
|-------|----------------|---------|
| Line Counter | `roboflow_core/line_counter@v2` | Count crossings of a line |
| Time in Zone | `roboflow_core/time_in_zone@v3` | Track time in polygon zones |
| Path Deviation | `roboflow_core/path_deviation@v2` | Measure path deviation |
| Velocity | `roboflow_core/velocity@v1` | Measure object velocity |
| Data Aggregator | `roboflow_core/data_aggregator@v1` | Aggregate data over time |

---

### Fusion

| Block | Type Identifier | Purpose |
|-------|----------------|---------|
| Detections Consensus | `roboflow_core/detections_consensus@v1` | Combine detections from multiple models |
| Detections Stitch | `roboflow_core/detections_stitch@v1` | Reassemble sliced detections |

---

## Data Operations Reference

Used in Expression, Property Definition, Data Aggregator, and filter blocks:

| Operation | Description |
|-----------|-------------|
| `SequenceLength` | Count items in a list |
| `DetectionsPropertyExtract` | Extract a property from detections (`class_name`, `confidence`, `x_min`, `y_min`, `width`, `height`, etc.) |
| `ClassificationPropertyExtract` | Extract from classification (`top_class`, `top_class_confidence`, `all_classes`, `all_confidences`) |
| `LookupTable` | Map values through a lookup table |
| `ToNumber` | Convert to number |
| `ToString` | Convert to string |
| `StringSubSequence` | Extract substring |
| `StringToUpperCase` | Convert to uppercase |
| `StringToLowerCase` | Convert to lowercase |
| `NumberRound` | Round a number |

## Condition Comparators

Used in Continue If and Expression blocks:

| Comparator | Description |
|------------|-------------|
| `(Number) ==` | Numeric equality |
| `(Number) !=` | Numeric inequality |
| `(Number) >` | Greater than |
| `(Number) <` | Less than |
| `(Number) >=` | Greater or equal |
| `(Number) <=` | Less or equal |
| `(String) ==` | String equality |
| `(String) !=` | String inequality |
| `in (Sequence)` | Membership test |
| `(String) contains` | Substring test |
| `(String) startsWith` | Prefix test |
| `(String) endsWith` | Suffix test |
| `Exists` | Not null/empty check |

---

## Pretrained Model IDs

These pretrained models are available without training:

| Model ID | Type | Description |
|----------|------|-------------|
| `yolov8n-640` | Object Detection | YOLOv8 Nano (COCO, 80 classes) |
| `yolov8s-640` | Object Detection | YOLOv8 Small (COCO, 80 classes) |
| `yolov8m-640` | Object Detection | YOLOv8 Medium (COCO, 80 classes) |
| `yolov8n-1280` | Object Detection | YOLOv8 Nano high-res |
| `yolov8s-1280` | Object Detection | YOLOv8 Small high-res |
| `yolov11n-640` | Object Detection | YOLOv11 Nano (COCO) |
| `yolov11s-640` | Object Detection | YOLOv11 Small (COCO) |
