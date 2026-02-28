---
name: workflow
description: >
  Build, run, and iterate on Roboflow Workflows — visual AI pipelines that chain
  together models, transformations, and visualizations. Use when the user wants to
  solve a computer vision problem (detection, classification, OCR, VLM analysis,
  counting, tracking, etc.) by generating workflow JSON, running it against images,
  and refining based on visual feedback.
---

# Roboflow Workflow Builder

You help users build Roboflow Workflows — JSON definitions that describe a DAG of
processing blocks for computer vision tasks.

## Setup

Before doing anything, check if the user has a Roboflow API key configured:

```bash
python3 SKILL_DIR/scripts/setup.py status
```

If not configured, ask the user for their API key and workspace name, then run:

```bash
python3 SKILL_DIR/scripts/setup.py set --api-key <KEY> --workspace <NAME>
```

If they also want to use a self-hosted inference server instead of the default
(`https://serverless.roboflow.com`), they can specify:

```bash
python3 SKILL_DIR/scripts/setup.py set --api-key <KEY> --workspace <NAME> --api-url http://localhost:9001
```

## How Workflows Work

A workflow is a JSON document with four top-level keys:

```json
{
  "version": "1.0",
  "inputs": [ ... ],
  "steps": [ ... ],
  "outputs": [ ... ]
}
```

### Inputs

Inputs define what the workflow accepts at runtime:

- `WorkflowImage` — an image input (required for most workflows)
- `WorkflowParameter` — a runtime parameter (string, number, list, etc.)

```json
{"type": "WorkflowImage", "name": "image"},
{"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4}
```

### Steps

Steps are block instances. Each step has:
- `type` — the block type identifier (e.g., `roboflow_core/roboflow_object_detection_model@v2`)
- `name` — a unique name for this step (used in selectors)
- Block-specific properties — configured with static values or **selectors**

### Selectors

Selectors wire data between inputs, steps, and outputs:

- `$inputs.image` — references a workflow input
- `$steps.detection.predictions` — references a step's output
- `$steps.detection.*` — references all outputs of a step

### Outputs

Outputs define what the workflow returns:

```json
{"type": "JsonField", "name": "result", "selector": "$steps.detection.predictions"}
```

### Kinds (Type System)

Each block input/output has a **kind** — a type that determines compatibility:
- `image` — image data
- `object_detection_prediction` — bounding boxes with classes and confidence
- `instance_segmentation_prediction` — masks + bounding boxes
- `classification_prediction` — class probabilities
- `string` — text
- `float_zero_to_one` — confidence values
- `language_model_output` — VLM/LLM text output
- `roboflow_model_id` — model identifier string

Blocks can only connect when output kinds match input kinds.

### Dimensionality

Some blocks increase data dimensionality (1 image → N crops). The dynamic crop
block does this: it takes 1 image and produces N cropped images (one per detection).
Downstream blocks then process each crop individually. This is automatic — just
wire the outputs correctly.

## Building a Workflow — Step by Step

1. **Understand user intent**: What vision problem are they solving?

2. **Identify needed blocks**: Use the search and describe scripts to find the
   right blocks. Start with REFERENCE.md for common blocks.

3. **Construct the JSON**: Build the workflow step by step, wiring selectors correctly.

4. **Validate**: Run the validation script to check for errors before executing.

5. **Run on a test image**: Ask the user for an image (file path or URL), then run
   the workflow and show results.

6. **Iterate**: Show the results (including any visualization images), ask if it
   looks right, and refine.

## Using the Scripts

### Discover blocks

```bash
# List all blocks grouped by category
python3 SKILL_DIR/scripts/list_blocks.py

# Filter by category keyword
python3 SKILL_DIR/scripts/list_blocks.py --category model

# Search by keyword (searches names, descriptions, search_keywords)
python3 SKILL_DIR/scripts/search_blocks.py "object detection"

# Get full details about a specific block
python3 SKILL_DIR/scripts/describe_block.py "roboflow_core/roboflow_object_detection_model@v2"
```

### Validate and run

```bash
# Validate a workflow definition
python3 SKILL_DIR/scripts/validate_workflow.py workflow.json

# Run a workflow against an image
python3 SKILL_DIR/scripts/run_workflow.py workflow.json --image path/to/image.jpg

# Run with a URL image
python3 SKILL_DIR/scripts/run_workflow.py workflow.json --image "https://example.com/image.jpg"

# Run with extra parameters
python3 SKILL_DIR/scripts/run_workflow.py workflow.json --image img.jpg --param confidence=0.5 --param model_id=yolov8n-640
```

### Manage config

```bash
# Check current configuration
python3 SKILL_DIR/scripts/setup.py status

# Set API key and workspace
python3 SKILL_DIR/scripts/setup.py set --api-key <KEY> --workspace <NAME>
```

## Common Patterns

### Pattern 1: Detect + Visualize

Detect objects, draw bounding boxes, show the result.

Blocks: object_detection_model → bounding_box_visualization

### Pattern 2: Detect → Crop → Classify (Two-Stage)

Find objects, crop each one, classify the crop.

Blocks: object_detection_model → dynamic_crop → classification_model

Key: dynamic_crop increases dimensionality. The classification model runs on each
crop. The classification output is nested (per-detection, per-image).

### Pattern 3: VLM Analysis

Send an image to a vision-language model with a prompt.

Blocks: anthropic_claude or google_gemini (task_type: unconstrained or visual-question-answering)

For structured output, add a parser block (vlm_as_classifier, vlm_as_detector, json_parser).

### Pattern 4: VLM as Zero-Shot Detector

Use a VLM to detect arbitrary objects by description.

Blocks: anthropic_claude (task_type: object-detection, classes: [...]) → vlm_as_detector

### Pattern 5: OCR Pipeline

Detect text regions, crop, run OCR.

Blocks: object_detection_model → dynamic_crop → ocr_model

### Pattern 6: SAHI (Small Object Detection)

Slice image into overlapping tiles, detect on each, stitch results back.

Blocks: image_slicer → object_detection_model → detections_stitch → visualization

### Pattern 7: Count Objects

Detect objects and count them, optionally by class.

Blocks: object_detection_model → expression (with SequenceLength on predictions)

## VLM Blocks — Special Notes

VLM blocks (Claude, Gemini, OpenAI, Florence-2) are versatile. Key properties:

- `task_type` controls the prompt template and output format:
  - `unconstrained` — free-form prompt, free-form output
  - `visual-question-answering` — prompt is a question, output is an answer
  - `classification` — provide `classes`, output is a classification
  - `multi-label-classification` — provide `classes`, output can have multiple labels
  - `object-detection` — provide `classes`, output is bounding boxes (parsed by vlm_as_detector)
  - `ocr` — extract text from image
  - `caption` / `detailed-caption` — describe the image
  - `structured-answering` — provide `output_structure` dict, output is JSON

- `api_key` can be `"rf_key:account"` to use a Roboflow-managed key (requires Roboflow API key in workflow inputs)
- Or a direct API key string for the provider

## Important Tips

- **Model IDs**: Roboflow models use `project-slug/version` format (e.g., `"yolov8n-640"` for pretrained, `"my-project/3"` for custom). Use `search_blocks.py` or ask the user for their model ID.

- **Pretrained models**: `yolov8n-640` is a good default for general object detection.

- **Visualization blocks**: Always need both `image` (original image) and `predictions`. The output is an `image` kind that can be returned as a workflow output.

- **Visualization output images**: When a workflow returns visualization images, they come as base64-encoded data in the response. The `run_workflow.py` script saves them to temp files and prints the paths, which you can show the user using the Read tool.

- **Block versions**: Always use the latest version of a block (e.g., `@v2` not `@v1`). Use `describe_block.py` to check available versions.

- **Chaining visualizations**: You can chain multiple visualization blocks. Pass the output image of one visualization as the input image of the next (e.g., bounding_box → label → mask).

- **Coordinates system**: For outputs that are spatial (detections), you can add `"coordinates_system": "own"` to the output to get coordinates relative to the original image.

## Consult the Reference

See REFERENCE.md for a curated cheatsheet of the most commonly used blocks with
their type identifiers, key inputs, outputs, and configuration options.
