[YOLO26](https://blog.roboflow.com/yolo26/), released in January 2026, is a multi-task computer vision model family designed for edge deployment with faster CPU inference, improved small-object detection, and end-to-end predictions without Non-Maximum Suppression (NMS).

YOLOv26 is optimized for real-time performance on edge devices, featuring a simplified architecture that removes the Distribution Focal Loss (DFL) module for broader device compatibility and faster inference. The model family includes five size variants (Nano, Small, Medium, Large, and Extra Large) to support different performance and deployment requirements.

Key improvements over previous YOLO versions include:

- **Faster CPU Inference**: Up to 43% faster CPU inference compared to YOLOv11-N

- **End-to-End Predictions**: Eliminates NMS post-processing for reduced latency

- **Enhanced Small-Object Detection**: Utilizes ProgLoss and STAL loss functions for improved accuracy

- **Broader Device Support**: Supports multiple export formats (TFLite, CoreML, OpenVINO, TensorRT, ONNX)

- **Improved Training**: Introduces MuSGD optimizer for stable training and faster convergence

## Supported Model Types

You can deploy the following YOLOv26 model types with Inference:

- Object Detection

- Instance Segmentation

- Keypoint Detection (Pose Estimation)

## Model Overview

- [What is YOLOv26? An Introduction](https://blog.roboflow.com/yolo26/)
- [How to Train and Deploy a YOLOv26 Object Detection Model](https://blog.roboflow.com/yolo26-in-roboflow/)

## Usage Example

You can use YOLOv26 with the following code:

```python
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

# Load image
response = requests.get("https://media.roboflow.com/dog.jpeg")

if response.status_code == 200:
    image_data = BytesIO(response.content)
    image = Image.open(image_data)

# Object Detection with your YOLO26 model
model = get_model("<project-id>/<version-id>")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)
```

## Supported Inputs

Click a link below to see instructions on how to run a YOLOv26 model on different inputs:

- [Image](../quickstart/run_model_on_image.md)
- [Video, Webcam, or RTSP Stream](../quickstart/run_model_on_rtsp_webcam.md)

## License

See our [Licensing Guide](https://roboflow.com/licensing) for more information about how your use of YOLOv26 is licensed when using Inference to deploy your model.
