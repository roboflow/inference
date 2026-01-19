# Quick Overview

`inference-models` is a library that makes running computer vision models simple and efficient across different hardware environments. It provides a unified interface - the same code runs on a laptop CPU during prototyping and on production GPUs or Jetson devices without modification.

## Why inference-models?

**The problem:** Different AI frameworks (PyTorch, ONNX, TensorRT) require different code and setup. Managing dependencies across environments is a headache, and the most efficient backends like TensorRT provide the most complexity - from CUDA version compatibility to engine building and optimization.

**The solution:** `inference-models` provides a single interface across all backends. The library automatically detects available hardware (CPU, NVIDIA GPU, Jetson) and selects the optimal backend (TensorRT, ONNX, PyTorch). Code written for CPU prototyping works unchanged on production GPUs, leveraging TensorRT's performance when available.

!!! tip "TensorRT Engine Management"
    Need pre-compiled TensorRT engines for maximum performance? Roboflow platform provides tools for TensorRT compilation and optimization. [Contact us](https://roboflow.com/contact) to learn more.

## Basic Usage

Different model categories have different interfaces tailored to their specific tasks. Object detection models use the standard `model(image)` call and return bounding boxes, while vision-language models use `model.prompt(image, text)` for multimodal interactions. Below you can see usage examples for models from different categories, demonstrating how the API adapts to each task type while maintaining consistency in core operations like loading and batching.


### Object Detection

=== "numpy"

    ```python hl_lines="9"
    import cv2
    import supervision as sv
    from inference_models import AutoModel

    # Load model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("rfdetr-base")

    # Load image
    image = cv2.imread("path/to/image.jpg")

    # Run inference
    predictions = model(image)

    # Visualize results
    annotator = sv.BoxAnnotator()
    annotated = annotator.annotate(image, predictions[0].to_supervision())
    cv2.imwrite("output.jpg", annotated)
    ```

=== "torch.Tensor"

    ```python hl_lines="10"
    import cv2
    import supervision as sv
    from inference_models import AutoModel
    from torchvision.io import read_image

    # Load model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("rfdetr-base")

    # Load image
    image = read_image("path/to/image.jpg")

    # Run inference
    predictions = model(image)

    # Visualize results
    annotator = sv.BoxAnnotator()
    annotated = annotator.annotate(image, predictions[0].to_supervision())
    cv2.imwrite("output.jpg", annotated)
    ```

=== "Batched"

    ```python hl_lines="9-12"
    import cv2
    import supervision as sv
    from inference_models import AutoModel

    # Load model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("rfdetr-base")

    # Load images - can be list of numpy arrays, list of 3D tensors, or 4D tensor
    images = [
        cv2.imread("path/to/image1.jpg"),
        cv2.imread("path/to/image2.jpg"),
    ]

    # Run batched inference
    predictions = model(images)

    # Process results for each image
    annotator = sv.BoxAnnotator()
    for i, (image, prediction) in enumerate(zip(images, predictions)):
        annotated = annotator.annotate(image, prediction.to_supervision())
        cv2.imwrite(f"output_{i}.jpg", annotated)
    ```

### Image Classification

=== "numpy"

    ```python hl_lines="8"
    import cv2
    from inference_models import AutoModel

    # Load classification model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("resnet18")

    # Load image
    image = cv2.imread("path/to/image.jpg")

    # Run inference
    prediction = model(image)

    # Get top prediction
    top_class = model.class_names[prediction.class_id[0]]
    confidence = prediction.confidence[0][top_class]
    print(f"Predicted class: {top_class} (confidence: {confidence})")
    ```

=== "torch.Tensor"

    ```python hl_lines="8"
    from inference_models import AutoModel
    from torchvision.io import read_image

    # Load classification model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("resnet18")

    # Load image
    image = read_image("path/to/image.jpg")

    # Run inference
    prediction = model(image)

    # Get top prediction
    top_class = model.class_names[prediction.class_id[0]]
    confidence = prediction.confidence[0][top_class]
    print(f"Predicted class: {top_class} (confidence: {confidence})")
    ```

=== "Batched"

    ```python hl_lines="8-11"
    import cv2
    from inference_models import AutoModel

    # Load classification model (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("resnet18")

    # Load images - can be list of numpy arrays, list of 3D tensors, or 4D tensor
    images = [
        cv2.imread("path/to/image1.jpg"),
        cv2.imread("path/to/image2.jpg"),
    ]

    # Run batched inference
    prediction = model(images)

    # Process results for each image
    for i in range(len(images)):
        top_class = model.class_names[prediction.class_id[i]]
        confidence = prediction.confidence[i][top_class]
        print(f"Image {i}: {top_class} (confidence: {confidence})")
    ```

### Vision-Language Models

=== "numpy"

    ```python hl_lines="8"
    import cv2
    from inference_models import AutoModel

    # Load VLM (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("florence2-base")

    # Load image
    image = cv2.imread("path/to/image.jpg")

    # Run with prompt
    result = model.prompt(image, "Describe the image contents")

    print(result[0])
    ```

=== "torch.Tensor"

    ```python hl_lines="8"
    from inference_models import AutoModel
    from torchvision.io import read_image

    # Load VLM (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("florence2-base")

    # Load image
    image = read_image("path/to/image.jpg")

    # Run with prompt
    result = model.prompt(image, "Describe the image contents")

    print(result[0])
    ```

=== "Batched"

    ```python hl_lines="8-11"
    import cv2
    from inference_models import AutoModel

    # Load VLM (optionally specify device="cuda:0" or leave default)
    model = AutoModel.from_pretrained("florence2-base")

    # Load images - can be list of numpy arrays, list of 3D tensors, or 4D tensor
    images = [
        cv2.imread("path/to/image1.jpg"),
        cv2.imread("path/to/image2.jpg"),
    ]

    # Run batched inference with prompt
    result = model.prompt(images, "Describe the image contents")

    print(result[0])  # First image description
    print(result[1])  # Second image description
    ```

## Backend Selection

Many models are available in multiple backend variants (PyTorch, ONNX, TensorRT). By default, `inference-models` uses **automatic backend negotiation**: it detects your installed dependencies and hardware, then selects the fastest compatible backend (priority: TensorRT > PyTorch > Hugging Face > ONNX). This means the same code automatically uses TensorRT on production GPUs or falls back to CPU backends in development - no code changes needed.

=== "Auto-Negotiation (Default)"

    Automatically selects the best available backend for your environment:

    ```python
    from inference_models import AutoModel

    # Automatic backend selection - uses best available option
    model = AutoModel.from_pretrained("rfdetr-base")
    # On GPU with TensorRT installed: uses TensorRT
    # On GPU without TensorRT: uses PyTorch or ONNX
    # On CPU: uses PyTorch CPU or ONNX CPU
    ```

=== "Force PyTorch"

    Explicitly specify PyTorch backend:

    ```python hl_lines="4"
    from inference_models import AutoModel

    # Force PyTorch backend
    model = AutoModel.from_pretrained("rfdetr-base", backend="torch")
    ```

=== "Force ONNX"

    Explicitly specify ONNX Runtime backend:

    ```python hl_lines="4"
    from inference_models import AutoModel

    # Force ONNX Runtime backend
    model = AutoModel.from_pretrained("rfdetr-base", backend="onnx")
    ```

=== "Force TensorRT"

    Explicitly specify TensorRT backend:

    ```python hl_lines="4"
    from inference_models import AutoModel

    # Force TensorRT backend
    model = AutoModel.from_pretrained("rfdetr-base", backend="trt")
    ```

## Next Steps

- **[Installation Guide](installation.md)** - Detailed installation options for all backends
- **[Understand Core Concepts](../how-to/understand-core-concepts.md)** - Deep dive into design philosophy
- **[Supported Models](../models/index.md)** - Browse all available models

