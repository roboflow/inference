# Inference Benchmarks

This page contains performance benchmarks for various models running with Inference on different hardware platforms.

## NVIDIA L4 GPU

### Object Detection

Performance benchmarks for object detection models, comparing standard ONNX runtime with TensorRT-optimized adapters:

| Model Type | Size | Inference/sec (ONNX) | Inference/sec (TRT) | Improvement Multiplier |
|------------|------|---------------------|---------------------|----------------------|
| rfdetr-nano | 384x384 | 103.5 | 299.5 | 2.9 |
| rfdetr-small | 512x512 | 57.4 | 253.4 | 4.4 |
| rfdetr-medium | 576x576 | 62.8 | 201.8 | 3.2 |
| rfdetr-large | 704x704 | 36.9 | 160.5 | 4.3 |
| rfdetr-xlarge | 700x700 | 18.1 | 96.1 | 5.3 |
| rfdetr-2xlarge | 880x880 | 17.4 | 74.1 | 4.3 |

### Segmentation

Performance benchmarks for instance segmentation models:

| Model Type | Size | Inference/sec (ONNX) | Inference/sec (TRT) | Improvement Multiplier |
|------------|------|---------------------|---------------------|----------------------|
| rfdetr-seg-nano | 312x312 | 51.9 | 105.3 | 2.0 |
| rfdetr-seg-small | 384x384 | 57.5 | 126.7 | 2.2 |
| rfdetr-seg-medium | 432x432 | 39.7 | 99.7 | 2.5 |
| rfdetr-seg-large | 504x504 | 32.8 | 93.2 | 2.8 |
| rfdetr-seg-xlarge | 624x624 | 17 | 68.8 | 4.0 |
| rfdetr-seg-2xlarge | 768x768 | 10.7 | 59 | 5.5 |

### Classification

Performance benchmarks for classification models:

| Model Type | Size | Inference/sec (ONNX) | Inference/sec (TRT) | Improvement Multiplier |
|------------|------|---------------------|---------------------|----------------------|
| ResNet50 | 224x224 | 358.6 | 600.8 | 1.7 |
| ViT | 224x224 | 238 | 306.5 | 1.3 |

## Jetson Orin NX

### Object Detection

Performance benchmarks for object detection models on Jetson Orin NX:

| Model Type | Size | Inference/sec (ONNX) | Inference/sec (TRT) | Improvement Multiplier |
|------------|------|---------------------|---------------------|----------------------|
| rfdetr-nano | 384x384 | 21.2 | 78.5 | 3.7 |
| rfdetr-small | 512x512 | 13.9 | 52.5 | 3.8 |
| rfdetr-medium | 576x576 | 11 | 44 | 4.0 |
| yolov8n-640 | 640x640 | 35.6 | 89 | 2.5 |
| yolov8s-640 | 640x640 | 26.3 | 69.5 | 2.6 |
| yolov8m-640 | 640x640 | 13.5 | 44.5 | 3.3 |
| yolov8l-640 | 640x640 | 9 | 32.5 | 3.6 |
| yolov8x-640 | 640x640 | 6.4 | 22 | 3.4 |

## Benchmark Methodology

All benchmarks were conducted using the [Roboflow Inference CLI](../inference_helpers/inference_cli.md). All were used single image (batch size=1, `-bs 1`) and on 500 iterations (`-bi 500`)

- **Inference/sec (ONNX)** - Standard ONNX runtime, measured with:
  ```bash
  inference benchmark python-package-speed -m [model]
  ```

- **Inference/sec (TRT)** - TensorRT-optimized adapters (supported in `inference 1.0` and later), measured with:
  ```bash
  USE_INFERENCE_MODELS=TRUE inference benchmark python-package-speed -m [model]
  ```