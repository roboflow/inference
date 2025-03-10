# Qwen2.5 VL

[Qwen2.5 VL](https://qwenlm.github.io/blog/qwen2.5-vl/) is a large multimodal model developed by the Qwen Team.

You can use Qwen2.5 VL to:

1. Ask questions about images (Visual Question Answering)
2. Recognize objects and landmarks worldwide with high accuracy
3. Precisely locate objects using bounding boxes or points (Object Grounding)
4. Extract and understand text from images with enhanced multi-language OCR
5. Parse and analyze documents using the unique QwenVL HTML format
6. Understand videos (including hour-long videos) and locate specific events
7. Generate structured JSON outputs for coordinates and attributes
8. Act as a visual agent for computer and phone interfaces

## Installation

To install inference with the extra dependencies necessary to run Qwen2.5 VL, run:

```bash
pip install inference[transformers]
```

or

```bash
pip install inference-gpu[transformers]
```

## How to Use Qwen2.5 VL (Visual Question Answering)

Create a new Python file called `app.py` and add the following code:

```python
import inference
from inference.models.qwen25vl.qwen25vl import Qwen25VL

# Initialize the model
model = Qwen25VL("qwen2.5-vl-7b", api_key="YOUR ROBOFLOW API KEY")

# Load an image
from PIL import Image
image = Image.open("image.jpg")  # Change to your image path

# Define your prompt
prompt = "What objects are in this image?"

# Make a prediction
result = model.predict(image, prompt)
print(result)
```

In this code, we:
1. Load the Qwen2.5 VL model
2. Load an image
3. Define a prompt to ask about the image
4. Run the model and print the results

To use Qwen2.5 VL with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com).

## How to Use Qwen2.5 VL (Object Detection)

Create a new Python file called `app.py` and add the following code:

```python
import inference
from inference.models.qwen25vl.qwen25vl import Qwen25VL
from PIL import Image
import json

# Initialize the model
model = Qwen25VL("qwen2.5-vl-7b", api_key="YOUR ROBOFLOW API KEY")

# Load an image
image = Image.open("image.jpg")  # Change to your image path

# Define your detection prompt
prompt = "Detect all objects in this image and return their locations as JSON."

# Make a prediction
result = model.predict(image, prompt)[0]
print(result)

# If you want to visualize the results, you can use supervision
import supervision as sv
import numpy as np

# Parse the JSON result
detections_data = json.loads(result)

# Create a Detections object
xyxy_list = []
class_name_list = []

for detection in detections_data:
    if "bbox_2d" in detection:
        xyxy_list.append(detection["bbox_2d"])
        class_name_list.append(detection["label"])

xyxy = np.array(xyxy_list)
class_name = np.array(class_name_list)

detections = sv.Detections(
    xyxy=xyxy,
    class_id=None,
    data={'class_name': class_name}
)

# Visualize
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections)
sv.plot_image(annotated_image)
```

This code will:
1. Load the Qwen2.5 VL model
2. Ask it to detect objects in an image
3. Parse the results (which come in JSON format)
4. Visualize the detections with bounding boxes

## Advanced Usage with System Prompts

You can customize the model's behavior by providing a system prompt:

```python
# Define system prompt and query
prompt = "Identify all landmarks in this image"
system_prompt = "You are an expert in world landmarks recognition"

# Combine them
combined_prompt = f"{prompt}<system_prompt>{system_prompt}"

# Make a prediction
result = model.predict(image, combined_prompt)
```

## Model Variants

Qwen2.5 VL is available in multiple sizes:
- Qwen2.5-VL-3B: Smaller model suitable for edge devices
- Qwen2.5-VL-7B: Medium-sized model with good performance
- Qwen2.5-VL-72B: Large model with state-of-the-art capabilities

You can also use the LoRA versions of these models:

```python
from inference.models.qwen25vl.qwen25vl import LoRAQwen25VL

# Initialize a LoRA model
lora_model = LoRAQwen25VL("qwen2.5-vl-7b-lora", api_key="YOUR ROBOFLOW API KEY")
```

## Learn More

For more details about Qwen2.5 VL's capabilities, including video understanding and visual agent abilities, visit the [official Qwen2.5 VL page](https://qwenlm.github.io/blog/qwen2.5-vl/).