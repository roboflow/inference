# Segment Anything 3 (SAM 3)

[Segment Anything 3 (SAM 3)](https://ai.meta.com/sam3) is a unified foundation model for promptable segmentation in images and videos. It builds upon SAM 2 by introducing the ability to exhaustively segment all instances of an open-vocabulary concept specified by a short text phrase or exemplars.

SAM 3 can detect, segment, and track objects using:
- **Text prompts** (e.g., "a person", "red car")
- **Visual prompts** (boxes, points)

## How to Use SAM 3 with Inference

You can use SAM 3 via the Inference Python SDK or the HTTP API.

### Prerequisites

To use SAM 3, you will need a Roboflow API key. [Sign up for a free Roboflow account](https://app.roboflow.com) to retrieve your key.

### Python SDK

You can run SAM 3 locally using the `inference` package.

#### 1. Install the package

```bash
uv pip install inference-gpu[sam3]
```

#### 2. Run Inference

Here is an example of how to use SAM 3 with a text prompt to segment objects.

```python
import os
from inference.models.sam3 import SegmentAnything3
from inference.core.entities.requests.sam3 import Sam3Prompt

# Set your API key
os.environ["API_KEY"] = "<YOUR_ROBOFLOW_API_KEY>"

# Initialize the model
# The model will automatically download weights if not present
model = SegmentAnything3(model_id="sam3/sam3_final")

# Define your image (can be a path, URL, or numpy array)
image_path = "path/to/your/image.jpg"

# Define prompts
# SAM 3 supports both text and visual prompts
prompts = [
    Sam3Prompt(type="text", text="person"),
    Sam3Prompt(type="text", text="car")
]

# Run inference
response = model.segment_image(
    image=image_path,
    prompts=prompts,
    output_prob_thresh=0.5,
    format="polygon" # or "rle", "json"
)

# Process results
for prompt_result in response.prompt_results:
    print(f"Prompt: {prompt_result.echo.text}")
    for prediction in prompt_result.predictions:
        print(f"  Confidence: {prediction.confidence}")
        print(f"  Mask: {prediction.masks}")
```

### Interactive Segmentation (SAM 2 Style)

SAM 3 also supports interactive segmentation using points and boxes, maintaining compatibility with the SAM 2 interface. This is handled by the `Sam3ForInteractiveImageSegmentation` class.

This mode is ideal for "human-in-the-loop" workflows where you want to refine masks using clicks or bounding boxes.

```python
from inference.models.sam3 import Sam3ForInteractiveImageSegmentation

# Initialize the interactive model
model = Sam3ForInteractiveImageSegmentation(model_id="sam3/sam3_final")

# Embed the image (calculates image features)
embedding, img_shape, image_id = model.embed_image(image="path/to/image.jpg")

# Segment with a point prompt
# points are (x, y), label 1 is positive (include), 0 is negative (exclude)
masks, scores, logits = model.segment_image(
    image_id=image_id,
    prompts={
        "points": [{"x": 500, "y": 400, "positive": True}]
    }
)

# The result 'masks' contains the segmentation masks for the prompt
```

### HTTP API

You can run SAM 3 via the Inference HTTP API. This is useful if you are running the Inference Server in Docker.

SAM 3 exposes two main modes via API:
1. **Promptable Visual Segmentation (PVS)**: Similar to SAM 2, using points and boxes.
2. **Promptable Concept Segmentation (PCS)**: Using text prompts or mixed text/visual prompts.

#### 1. Start the Server

```bash
docker run -it --rm -p 9001:9001 --gpus=all roboflow/inference-server:latest
```

#### 2. Concept Segmentation (Text Prompts)

This is the most common usage for SAM 3, allowing you to segment objects by text description.

```bash
curl -X POST 'http://localhost:9001/sam3/concept_segment?api_key=<YOUR_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {
      "type": "url",
      "value": "https://media.roboflow.com/inference/sample.jpg"
    },
    "prompts": [
        { "type": "text", "text": "cat" },
        { "type": "text", "text": "dog" }
    ]
  }'
```

#### 3. Visual Segmentation (Points/Boxes)

For interactive segmentation similar to SAM 2, you can use the visual segmentation endpoints.

**Step 1: Embed the Image** (Optional but recommended for speed)

```bash
curl -X POST 'http://localhost:9001/sam3/embed_image?api_key=<YOUR_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "image": {
      "type": "url",
      "value": "https://media.roboflow.com/inference/sample.jpg"
    }
  }'
# Returns an "image_id"
```

**Step 2: Segment with Points**

```bash
curl -X POST 'http://localhost:9001/sam3/visual_segment?api_key=<YOUR_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "image_id": "<IMAGE_ID_FROM_STEP_1>",
    "prompts": [
      { "points": [ { "x": 100, "y": 100, "positive": true } ] }
    ]
  }'
```

## Workflow Integration

SAM 3 is fully integrated into [Inference Workflows](https://inference.roboflow.com/workflows/core_steps/). You can use the **SAM 3** block to add zero-shot instance segmentation to your pipeline.

The Workflow block allows you to:
- Use **Text Prompts** to segment objects by class name.
- Use **Box Prompts** from other detection models (like YOLO) to generate precise masks for detected objects.

### Example: Text Prompting in Workflows

1. Add a **SAM 3** block to your workflow.
2. Connect an image input.
3. In the `class_names` field, enter the classes you want to segment (e.g., `["person", "vehicle"]`).
4. The block will output instance segmentation predictions compatible with other workflow steps.

## Capabilities & Features

- **Open Vocabulary Segmentation**: Unlike SAM 2 which requires visual prompts, SAM 3 can find objects based on text descriptions.
- **High Performance**: Achieves state-of-the-art performance on open-vocabulary benchmarks.
- **Unified Architecture**: Handles both detection and segmentation in a single model.

For more technical details, refer to the [official SAM 3 paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).
