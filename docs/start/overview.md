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

* **Serverless** - Pay-per-Inference, scales to zero. Doesn't support large [foundation models](../foundation/about).
* **Dedicated** - Single-tenant VMs with optional GPU. Supports larger foundation models (SAM 2, Florence-2, PaliGemma). Billed hourly.
* **Self-Hosted** - Run on your own hardware. [Install guide →](../install/)
* **Bring Your Own Cloud** - Self-host on [AWS, Azure, or GCP](../install/cloud/) for enterprise compliance.

## Quick Start

Install the [inference-sdk](../inference_helpers/inference_sdk.md):

```bash
pip install inference-sdk
```

To self-host, start a local [Inference server](../install/) with the [inference-cli](../inference_helpers/inference_cli.md):

```bash
pip install inference-cli && inference server start --port 9001
```

Then run inference:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    # api_url="https://serverless.roboflow.com", # for Roboflow hosted inference
    api_url="http://localhost:9001",  # for self-hosted inference
    api_key="YOUR_API_KEY", # For private/fine-tuned models
)

image_url="https://media.roboflow.com/inference/people-walking.jpg"
results = client.infer(image_url, model_id="rfdetr-small")
```

For more information, see [Run a model](../quickstart/run_a_model.md).

## Related Products

<div class="product-cards">
  <div class="product-card">
    <div class="product-card__header">
      <img class="product-card__icon" src="https://universe.roboflow.com/favicon.ico" alt="Roboflow">
      <a class="product-card__title" href="https://app.roboflow.com/">Roboflow App</a>
    </div>
    <p class="product-card__desc">Upload data, annotate images, train and deploy models. <a href="https://docs.roboflow.com/api-reference/authentication">Get your API key</a>.</p>
  </div>
  <div class="product-card">
    <div class="product-card__header">
      <img class="product-card__icon" src="../../images/universe-icon.svg" alt="Universe">
      <a class="product-card__title" href="https://universe.roboflow.com/">Universe</a>
    </div>
    <p class="product-card__desc">Browse and use community datasets and models. Pass any Universe <code>model_id</code> directly to Inference.</p>
  </div>
  <div class="product-card">
    <div class="product-card__header">
      <img class="product-card__icon" src="https://supervision.roboflow.com/assets/supervision-lenny.png" alt="Supervision">
      <a class="product-card__title" href="https://supervision.roboflow.com/latest/">Supervision</a>
    </div>
    <p class="product-card__desc">Post-process results: decode predictions, plot bounding boxes, track objects, slice images for small object detection.</p>
  </div>
</div>

## Open Source

Core functionality is open source under Apache 2.0.
[View on GitHub](https://github.com/roboflow/inference) ·
[Contribute](https://github.com/roboflow/inference/blob/main/CONTRIBUTING.md)

Models are subject to their underlying architecture licenses
([details](https://github.com/roboflow/inference/tree/main/inference/models)).
Cloud-connected features require a [Roboflow API key](https://roboflow.com/pricing).
