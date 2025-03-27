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

To use Qwen2.5 VL with the Inference SDK, install:

```bash
pip install inference-sdk
```

## How to Use Qwen2.5 VL (Visual Question Answering)

Create a new Python file called `app.py` and add the following code:

```python
from inference_sdk import InferenceHTTPClient

def run_qwen25_inference():
    # Create a client pointing to your inference server
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",  # You can also use a remote server if needed
        api_key="YOUR_API_KEY"            # Optional if your model requires an API key
    )
    
    # Invoke the model with an image and a prompt
    result = client.run_workflow(
        workspace_name="YOUR_WORKSPACE_NAME",  # Replace with your workspace name
        workflow_id="image-text/93",           # The model or workflow id
        images={
            "image": "https://media.roboflow.com/dog.jpeg"  # Can be a URL or local path
        },
        parameters={
            "prompt": "Tell me something about this dog!"
        }
    )
    
    print(result)

if __name__ == "__main__":
    run_qwen25_inference()
```

In this code, we:
1. Create an Inference HTTP client that connects to your inference server
2. Specify an image (either by URL or local path)
3. Define a prompt to ask about the image
4. Run the model and print the results

To use Qwen2.5 VL with Inference, you will need an API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com).

## How to Use Qwen2.5 VL (Object Detection)

Create a new Python file called `object_detection.py` and add the following code:

```python
from inference_sdk import InferenceHTTPClient
import json
import supervision as sv
import numpy as np
from PIL import Image

def run_qwen25_object_detection():
    # Create a client pointing to your inference server
    client = InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key="YOUR_API_KEY"
    )
    
    # Path to your local image
    image_path = "path/to/your/image.jpg"
    
    # Invoke the model with an image and a detection prompt
    result = client.run_workflow(
        workspace_name="YOUR_WORKSPACE_NAME",
        workflow_id="image-text/93",
        images={
            "image": image_path
        },
        parameters={
            "prompt": "Detect all objects in this image and return their locations as JSON."
        }
    )
    
    # Parse the JSON result
    detections_data = json.loads(result[0])
    
    # Load the image for visualization
    image = Image.open(image_path)
    
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

if __name__ == "__main__":
    run_qwen25_object_detection()
```

This code will:
1. Connect to your inference server
2. Ask Qwen2.5 VL to detect objects in an image
3. Parse the results (which come in JSON format)
4. Visualize the detections with bounding boxes

## Advanced Usage with System Prompts

You can customize the model's behavior by providing a system prompt:

```python
from inference_sdk import InferenceHTTPClient

# Create a client pointing to your inference server
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_API_KEY"
)

# Invoke the model with a system prompt
result = client.run_workflow(
    workspace_name="YOUR_WORKSPACE_NAME",
    workflow_id="image-text/93",
    images={
        "image": "path/to/image.jpg"
    },
    parameters={
        "prompt": "Identify all landmarks in this image<system_prompt>You are an expert in world landmarks recognition"
    }
)

print(result)
```

The system prompt is appended to the user prompt with the `<system_prompt>` delimiter.

## Model Variants

Qwen2.5 VL is currently only available as Qwen2.5-VL-7B.

The workflow ID may vary depending on which model variant you're using. Contact your administrator or refer to your deployment documentation for the correct workflow ID.

## Learn More

For more details about Qwen2.5 VL's capabilities, including video understanding and visual agent abilities, visit the [official Qwen2.5 VL page](https://qwenlm.github.io/blog/qwen2.5-vl/).