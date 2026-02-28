# Plan: Roboflow Workflow Builder Skill for Claude

## Executive Summary

Build a Claude Code skill (`/workflow`) that lets users describe a computer vision
problem in natural language and have Claude generate, run, and iteratively refine
a Roboflow Workflow — all within the conversation.

The skill leverages the inference server's existing HTTP endpoints for block discovery,
workflow validation, and workflow execution, plus the `inference_sdk` Python client
for programmatic execution.

---

## 1. Understanding the System

### 1.1 What is a Workflow?

A Workflow is a JSON document that describes a DAG (directed acyclic graph) of
processing steps. Each step is an instance of a "block" — a pluggable unit that
performs a specific operation (model inference, image transformation, visualization,
data formatting, etc.).

### 1.2 Workflow JSON Structure

```json
{
  "version": "1.0",
  "inputs": [
    {"type": "WorkflowImage", "name": "image"},
    {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.3}
  ],
  "steps": [
    {
      "type": "roboflow_core/roboflow_object_detection_model@v2",
      "name": "detection",
      "image": "$inputs.image",
      "model_id": "yolov8n-640",
      "confidence": "$inputs.confidence"
    },
    {
      "type": "roboflow_core/bounding_box_visualization@v1",
      "name": "bbox_viz",
      "image": "$inputs.image",
      "predictions": "$steps.detection.predictions"
    }
  ],
  "outputs": [
    {"type": "JsonField", "name": "predictions", "selector": "$steps.detection.predictions"},
    {"type": "JsonField", "name": "visualization", "selector": "$steps.bbox_viz.image"}
  ]
}
```

Key concepts:
- **Selectors** (`$inputs.X`, `$steps.X.Y`) wire data between blocks
- **Kinds** (`image`, `object_detection_prediction`, `string`, etc.) are the type system
  that determines which outputs are compatible with which inputs
- **Dimensionality** tracks batch nesting (e.g., cropping 1 image into N crops creates
  dimensionality level 2)

### 1.3 Available Block Categories (~200+ blocks)

| Category | Examples | Use Cases |
|----------|----------|-----------|
| **Models / Roboflow** | object_detection@v2, classification@v2, instance_segmentation@v2, keypoint_detection@v2 | Run user's trained models or pretrained models (yolov8n, etc.) |
| **Models / Foundation** | anthropic_claude@v3, google_gemini@v3, openai@v3, florence2@v1, clip@v1, ocr@v1, yolo_world@v1, segment_anything2@v1 | VLMs, zero-shot detection, OCR, embeddings, SAM |
| **Transformations** | dynamic_crop@v1, detections_filter@v1, detections_transformation@v1, image_slicer@v1, perspective_correction@v1 | Crop, filter, transform detections and images |
| **Visualizations** | bounding_box_visualization@v1, label_visualization@v1, mask_visualization@v1, halo_visualization@v1 (~20 types) | Draw annotations on images |
| **Classical CV** | image_blur@v1, threshold@v1, convert_grayscale@v1, contours@v1, dominant_color@v1, template_matching@v1, sift@v1 | Traditional image processing |
| **Analytics** | line_counter@v2, time_in_zone@v3, velocity@v1, path_deviation@v2 | Video analytics (counting, tracking, zones) |
| **Formatters** | expression@v1, json_parser@v1, csv_formatter@v1, vlm_as_classifier@v2, vlm_as_detector@v2 | Parse/format data, convert VLM output to detections |
| **Flow Control** | continue_if@v1, rate_limiter@v1 | Conditional execution |
| **Sinks** | webhook@v1, email_notification@v1, roboflow/dataset_upload@v2 | Export results |
| **Fusion** | detections_consensus@v1, detections_stitch@v1 | Combine results from multiple models |

### 1.4 Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /workflows/blocks/describe` | POST | Get ALL block schemas, their inputs, outputs, kinds, connections |
| `POST /workflows/validate` | POST | Validate a workflow definition without running it |
| `POST /workflows/run` | POST | Run a workflow with an inline specification |
| `POST /{workspace}/workflows/{id}` | POST | Run a saved workflow by name |
| `POST /workflows/describe_interface` | POST | Get input/output interface of a workflow |
| `GET /workflows/execution_engine/versions` | GET | List execution engine versions |

### 1.5 SDK Usage

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # or https://detect.roboflow.com
    api_key="YOUR_API_KEY"
)

result = client.run_workflow(
    specification=workflow_json,
    images={"image": "path/to/image.jpg"},
    parameters={"confidence": 0.5}
)
# result = [{"predictions": {...}, "visualization": {...}}, ...]
```

---

## 2. Design of the Skill

### 2.1 Skill Directory Structure

```
.claude/skills/workflow-builder/
├── SKILL.md                        # Main skill instructions (loaded by Claude)
├── REFERENCE.md                    # Workflow JSON reference, common patterns, block cheatsheet
├── scripts/
│   ├── list_blocks.py              # Fetch block list with categories from describe endpoint
│   ├── describe_block.py           # Get detailed info about a specific block
│   ├── search_blocks.py            # Search blocks by keyword/capability
│   ├── run_workflow.py             # Run a workflow against an image, return results
│   ├── validate_workflow.py        # Validate a workflow definition
│   └── show_image.py              # Display a base64 image result to the user
└── examples/
    ├── detect_and_visualize.json   # Simple detection + visualization
    ├── detect_crop_classify.json   # Two-stage: detect → crop → classify
    ├── ocr_pipeline.json           # Detection → crop → OCR
    ├── vlm_analysis.json           # VLM image analysis
    └── count_objects.json          # Detection + counting expression
```

### 2.2 SKILL.md — The Core Prompt

The SKILL.md should teach Claude:

1. **What workflows are** and the JSON structure (version, inputs, steps, outputs)
2. **The selector syntax** (`$inputs.X`, `$steps.X.Y`) and how blocks wire together
3. **The iterative workflow** for building workflows:
   - Understand user intent
   - Identify which blocks are needed (use scripts to discover)
   - Construct the JSON
   - Validate it
   - Run on a test image
   - Show results
   - Iterate based on feedback
4. **When to use which tools** — the scripts provide block discovery, validation, and execution
5. **Common patterns** — two-stage detection+classification, VLM analysis, OCR pipeline, etc.
6. **How to handle images** — ask the user for a test image path/URL, or use a default

### 2.3 REFERENCE.md — Technical Reference

Contains:
- Complete workflow JSON schema reference
- All input types (`WorkflowImage`, `WorkflowParameter`, `WorkflowBatchInput`)
- Output format (`JsonField` with `selector` and optional `coordinates_system`)
- Selector syntax rules
- Kinds type system overview (the most common kinds and what they mean)
- Dimensionality rules (when crops increase nesting depth)
- Top ~30 most useful blocks with their type identifiers, key inputs, and outputs
  (curated, not the full 200+ — Claude can look up details via scripts)
- Common workflow patterns as templates

### 2.4 Scripts

#### `list_blocks.py`
- Calls `POST /workflows/blocks/describe`
- Returns a summarized, categorized list: block name, type identifier, short description
- Supports filtering by category keyword
- Output is compact enough for Claude's context window

#### `describe_block.py <block_type_identifier>`
- Calls `POST /workflows/blocks/describe` and filters to the specific block
- Returns: full schema (all properties with descriptions, types, defaults),
  outputs (names and kinds), compatible input/output connections
- This is what Claude uses to understand exactly how to configure a block

#### `search_blocks.py <keyword>`
- Searches block names, descriptions, and search_keywords for matches
- Returns matching blocks with type identifiers and short descriptions
- Useful when Claude needs to find the right block for a capability

#### `run_workflow.py --spec <json_file_or_inline> --image <path_or_url> [--params key=value...]`
- Uses `inference_sdk.InferenceHTTPClient.run_workflow()` or direct HTTP POST
- Returns the workflow outputs
- For image outputs (visualizations), saves to a temp file and prints the path
  so Claude can show it to the user
- Handles errors gracefully and returns actionable error messages

#### `validate_workflow.py --spec <json_file_or_inline>`
- Calls `POST /workflows/validate`
- Returns success or detailed error messages
- Used as a quick check before running

#### `show_image.py <base64_or_path>`
- Converts base64 image data to a file and prints the path
- Claude can then use the Read tool to display it to the user

### 2.5 Example Workflows

Pre-built JSON templates for common patterns that Claude can reference and adapt:

1. **detect_and_visualize.json** — Object detection + bounding box + label visualization
2. **detect_crop_classify.json** — Two-stage: detect objects → crop → classify each crop
3. **ocr_pipeline.json** — Detect text regions → crop → OCR
4. **vlm_analysis.json** — Send image to VLM with custom prompt
5. **count_objects.json** — Detect objects and count by class using expressions

---

## 3. User Experience Flow

### 3.1 Happy Path

```
User: I want to detect dogs in my images, classify their breed, and get a
      visualization with breed labels on the bounding boxes.

Claude: I'll build a workflow for that! This is a classic two-stage pipeline:
        1. Object detection (find dogs)
        2. Crop each detection
        3. Classify each crop (breed identification)
        4. Replace detection labels with breed names
        5. Visualize with bounding boxes + labels

        [Claude uses scripts/search_blocks.py and scripts/describe_block.py
         to look up exact block schemas]

        Here's the workflow I've generated:
        [Shows JSON]

        Do you have a test image I can run this on? You can give me a file path
        or URL.

User: Here's one: https://media.roboflow.com/inference/dog.jpeg

Claude: [Runs scripts/validate_workflow.py, then scripts/run_workflow.py]

        Here are the results:
        - Detected 2 dogs with breeds: Golden Retriever (0.92), Labrador (0.87)
        - [Shows visualization image]

        Does this look right? Would you like to adjust anything?

User: Can you also add a confidence threshold parameter so I can tune it?

Claude: [Modifies the workflow JSON to add a WorkflowParameter for confidence
         and wires it to the detection block. Re-runs.]

        Done! Now the workflow accepts a `confidence` parameter (default 0.4).
        Here's the updated result with the same image...
```

### 3.2 Discovery Path

```
User: What kinds of things can I do with workflows?

Claude: [Uses scripts/list_blocks.py to show categories]

        Workflows can do:
        - Object detection, classification, segmentation (Roboflow models)
        - Zero-shot detection (YOLO-World, Florence-2)
        - VLM analysis (Claude, GPT-4V, Gemini) for open-ended questions
        - OCR (text reading)
        - Image processing (blur, threshold, crop, resize)
        - Visualizations (bounding boxes, masks, labels, heatmaps)
        - Analytics (counting, zone tracking, velocity)
        - And more...

        What problem are you trying to solve?
```

### 3.3 Iteration Path

After running a workflow, the user sees results and can ask Claude to:
- Adjust parameters (confidence, NMS threshold, class filters)
- Add new steps (e.g., add visualization, add a filter, add OCR)
- Fix issues (wrong model, incorrect wiring)
- Save the workflow (output the final JSON)

---

## 4. Implementation Plan

### Phase 1: Foundation (MVP)

**Goal:** A working skill that can build and run simple workflows.

1. **Create SKILL.md** with:
   - Workflow JSON format instructions
   - Selector syntax reference
   - Step-by-step process for building workflows
   - Instructions for using each script

2. **Create REFERENCE.md** with:
   - Top 30 most useful blocks (curated cheatsheet)
   - Common workflow patterns
   - Kind type reference
   - Input/output format reference

3. **Create scripts/**:
   - `list_blocks.py` — calls describe endpoint, returns categorized summary
   - `describe_block.py` — returns full schema for one block
   - `search_blocks.py` — keyword search across blocks
   - `run_workflow.py` — runs workflow via SDK, handles image outputs
   - `validate_workflow.py` — validates workflow definition

4. **Create examples/** with 3-5 template workflows

5. **Test** the skill end-to-end with several use cases:
   - Simple detection
   - Two-stage detection + classification
   - VLM analysis
   - OCR pipeline

### Phase 2: Enhanced Block Discovery

**Goal:** Make Claude better at finding and configuring the right blocks.

6. **Build a lightweight block index** — a pre-generated summary of all blocks
   organized by capability (what problem they solve) rather than just category.
   This could be a JSON or markdown file that's small enough to include in REFERENCE.md.

7. **Add connection-aware search** — the search script could also return which
   blocks can connect to a given block's outputs, helping Claude chain steps.

8. **Improve error recovery** — when validation fails, parse the error message
   and suggest fixes.

### Phase 3: User Context Integration

**Goal:** Integrate with the user's Roboflow workspace.

9. **Add `list_models.py`** — calls Roboflow API to list the user's available
   models (project/version format), so Claude can suggest using their trained models
   instead of generic pretrained ones.

10. **Add `list_workflows.py`** — loads saved workflows from the user's workspace,
    so Claude can modify existing workflows.

11. **Add `save_workflow.py`** — saves a workflow to the user's workspace via the API.

### Phase 4: Advanced Features

**Goal:** Support video, complex pipelines, and SDK code generation.

12. **Video pipeline generation** — teach Claude to generate Python scripts using
    the SDK to run workflows on video streams.

13. **WebRTC live preview** — integrate with the WebRTC endpoint for live webcam
    workflow testing.

14. **Complex pipeline patterns** — conditional execution, multi-model fusion,
    analytics blocks with zones.

---

## 5. Technical Considerations

### 5.1 Block Discovery Strategy

The full `/workflows/blocks/describe` response is very large (~200+ blocks with full schemas).
We should NOT include this in the skill's static context. Instead:

- **REFERENCE.md** contains a curated cheatsheet of the ~30 most commonly used blocks
  with their type identifiers, key inputs/outputs, and one-line descriptions.
- **Scripts** provide on-demand detailed lookup for any block.
- Claude first identifies the general capability needed, then uses the search/describe
  scripts to get exact configuration details.

### 5.2 Inference Server Connectivity

The scripts need to know where the inference server is. Options:
- Environment variable: `ROBOFLOW_API_URL` (default: `https://detect.roboflow.com`)
- Environment variable: `ROBOFLOW_API_KEY`
- The scripts should work with both local servers and the hosted Roboflow API

### 5.3 Image Handling

- Workflow outputs that are images come back as base64-encoded strings
- The `run_workflow.py` script should decode these and save to temp files
- Claude can then use the Read tool to show the image to the user
- Input images can be URLs, local file paths, or base64

### 5.4 Context Window Management

The skill should be designed to be lean on context:
- SKILL.md: ~2-3 pages of core instructions
- REFERENCE.md: ~3-4 pages of curated reference (block cheatsheet + patterns)
- Scripts handle the heavy lifting of block discovery outside of context
- Example workflows are small JSON files (<30 lines each)

### 5.5 Error Handling

Common failure modes and how the skill should handle them:
- **Kind mismatch**: Script returns which kinds are expected vs. provided
- **Missing model**: Suggest available models or pretrained alternatives
- **Invalid selector**: Script validates selector format and step name existence
- **API key missing**: Prompt user for their Roboflow API key
- **Server unreachable**: Check if local server is running, suggest hosted API

---

## 6. What Needs to Be Built in This Repository

The skill itself lives in `.claude/skills/workflow-builder/` and is purely a set of
markdown files and Python scripts. It does NOT modify the inference server code.

However, some enhancements to the inference server could make the skill work better:

### 6.1 Useful New Endpoints (Optional, Not Required for MVP)

1. **`GET /workflows/blocks/summary`** — A lightweight endpoint that returns just
   block names, type identifiers, categories, and short descriptions (without full
   schemas). Much smaller than the full describe endpoint. This would be ideal for
   the skill's block discovery.

2. **`POST /workflows/blocks/describe/{block_type}`** — Describe a single block by
   type identifier. Currently you have to fetch ALL blocks and filter client-side.

These are nice-to-haves. The MVP skill can work by fetching the full describe
response and filtering in the Python scripts.

### 6.2 Files to Create

```
.claude/skills/workflow-builder/
├── SKILL.md
├── REFERENCE.md
├── scripts/
│   ├── list_blocks.py
│   ├── describe_block.py
│   ├── search_blocks.py
│   ├── run_workflow.py
│   ├── validate_workflow.py
│   └── show_image.py
└── examples/
    ├── detect_and_visualize.json
    ├── detect_crop_classify.json
    ├── ocr_pipeline.json
    ├── vlm_analysis.json
    └── count_objects.json
```

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Block describe response too large for context | Scripts filter and summarize; REFERENCE.md has curated top-30 |
| Claude generates invalid selectors | Validate script catches errors before running; REFERENCE.md has selector syntax rules |
| Model not available on server | Scripts check model availability; suggest pretrained alternatives |
| Dimensionality mismatches (crop → classify wiring) | REFERENCE.md documents the crop dimensionality pattern; describe script shows dimensionality info |
| API key not configured | Scripts check for env vars and prompt user |
| Inference server not running | Scripts detect connection failures and suggest alternatives (hosted API) |

---

## 8. Success Criteria

The skill is successful when a user can:

1. Describe a vision problem in plain English
2. Have Claude generate a working workflow JSON
3. Run the workflow on a test image and see results
4. Iterate on the workflow based on visual feedback
5. Get the final workflow JSON to deploy

All within a single Claude conversation, without needing to understand the
workflow JSON format or block system themselves.
