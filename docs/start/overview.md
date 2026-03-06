# Inference

Inference is an open-source computer vision deployment hub by [Roboflow](https://roboflow.com).
It handles model serving, video stream management, pre/post-processing, and
GPU/CPU optimization so you can focus on building your application.

## Architecture

Read more about the Inference architecture [here](../../understand/architecture). Quick links:

<p style="text-align: center; font-size: 1.2em;"><a href="../../inference_helpers/inference_sdk">inference_sdk</a> · <a href="../../inference_helpers/inference_cli">inference_cli</a> · <a href="../../quickstart/docker">Inference server</a> · <a href="../../using_inference/about">inference</a></p>

![Inference Architecture](../../images/img-inference-diagram-v1.png)

## Features

- **[Model Serving](../quickstart/run_a_model)** - Object detection, classification, segmentation,
  keypoint detection, OCR, VQA, gaze detection, and more. Supports
  [pre-trained](../quickstart/aliases), [fine-tuned](https://roboflow.com/train),
  and [foundation](../foundation/about) models.
- **[Workflows](../workflows/about)** - Chain models, logic, integrations, and custom
  Python code into declarative computation graphs with 100+ built-in blocks.
- **[Video Streaming](../workflows/video_processing/overview)** - Efficient
  `InferencePipeline` for consuming camera feeds, RTSP streams, and video files
  with automatic frame management and state tracking.
- **[Speed](../understand/features#speed)** - Automatic parallelization, hardware
  acceleration, dynamic batching, and optional TensorRT quantization.
- **[Extensibility](../understand/features#extensibility)** - Open source (Apache 2.0).
  Add custom models, Workflow blocks, and backends.

## Deploy Anywhere

| | [Serverless](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) | [Dedicated](https://docs.roboflow.com/deploy/dedicated-deployments) | [Self-Hosted](../install/) |
|---|---|---|---|
| Fine-Tuned & Pre-Trained Models | ✅ | ✅ | ✅ |
| Workflows | ✅ | ✅ | ✅ |
| Foundation Models | | ✅ | ✅ |
| Video Streaming | | ✅ | ✅ |
| Dynamic Python Blocks | | ✅ | ✅ |
| Runs Offline | | | ✅ |
| Billing | Per-Call | Hourly | Free + [metered](https://roboflow.com/pricing) |

**Serverless** - Scales to zero, pay per inference, supports cloud-hosted VLMs. No video streaming or heavy foundation models.

**Dedicated** - Single-tenant VMs with optional GPU. Video streaming, custom Python, heavy models (SAM 2, Florence-2, PaliGemma). Billed hourly.

**Self-Hosted** - Run on your own hardware. Full feature set, works offline. [Install guide →](../install/)

**Bring Your Own Cloud** - Self-host on [AWS, Azure, or GCP](../install/cloud/) for enterprise compliance.

## Quick Start

??? note "Installation"

    === "CPU"
        ```bash
        pip install inference
        ```

    === "Nvidia GPU"
        ```bash
        pip install inference-gpu
        ```

Or use Docker (recommended for production):

```bash
pip install inference-cli
inference server start
```

```python
from inference import get_model

model = get_model(model_id="yolov8n-640")
results = model.infer("https://media.roboflow.com/inference/people-walking.jpg")
```

!!! info

    For a more detailed example, see the tutorial on [running a model](../quickstart/run_a_model).

## Related Products

- **[Roboflow App](https://app.roboflow.com/)** - Upload data, annotate images, train and deploy models. [Get your API key](https://docs.roboflow.com/api-reference/authentication).
- **[Universe](https://universe.roboflow.com/)** - Browse and use community datasets and models. Pass any Universe `model_id` directly to Inference. [Learn more →](../quickstart/explore_models.md)
- **[Supervision](https://supervision.roboflow.com/latest/)** - Post-process results: plot bounding boxes, track objects, slice images for small object detection. Works with Inference, Ultralytics, Hugging Face, and more.
- **[Workflows](../workflows/about.md)** - Chain blocks together to build CV pipelines without writing code. [Browse blocks →](/workflows/blocks/index.md)

## Open Source

Core functionality is open source under Apache 2.0.
[View on GitHub](https://github.com/roboflow/inference) ·
[Contribute](https://github.com/roboflow/inference/blob/main/CONTRIBUTING.md)

Models are subject to their underlying architecture licenses
([details](https://github.com/roboflow/inference/tree/main/inference/models)).
Cloud-connected features require a [Roboflow API key](https://roboflow.com/pricing).
