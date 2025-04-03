# Overview

Inference is the centralized hub that handles the core tasks of building
computer vision applications. It implements the recurring heavy-lifting required for
efficiently processing and managing video streams, deploying and monitoring
models and pipelines, optimizing CPU and GPU resources, and managing dependencies.

## Core Principles

There are three three core principles of Inference:

1. **Easy to Use** - You shouldn't need a PhD to use computer vision. Getting a project up and running should take hours, not weeks. Developers should spend their time on the things that are unique to their project, not reinventing the wheel. We try to maximize your speed of iteration and experimentation.
2. **Production Grade** - Inference should be stable, thoroughly tested, scalable, secure, and fully-featured. It should be as fast as possible without degrading the developer experience.
3. **Extensible** - Developers should never be constrained by what's included in the box. Adding additional models and functionality should be easy and seamless.

## Use at Any Scale

Inference is developed by [Roboflow](https://roboflow.com), a computer vision
platform used by over a million developers and over half of the Fortune 100.
Over the years, it has powered billions of inferences for mission critical
applications like monitoring essential infrastructure, enforcing safety
controls in warehouses, ensuring quality of life-saving products, and
powering instant replay at some of the world's premier sporting events.

## Open Source

The core functionality of Inference is open source
[on GitHub](https://github.com/roboflow/inference)
under the Apache 2.0 license. You may freely fork, extend, or
[contribute](https://github.com/roboflow/inference/blob/main/CONTRIBUTING.md) to its functionality.

Models are subject to licensing which respects the underlying architecture.
These licenses are listed in
[`inference/models`](https://github.com/roboflow/inference/tree/main/inference/models).
Paid Roboflow accounts include a commercial license for some models
(see [roboflow.com/licensing](https://roboflow.com/licensing) for details).

Cloud connected functionality (like our model and Workflows registries, dataset management,
model monitoring, device management, and managed infrastructure) requires a Roboflow account
and API key & is metered based on usage according to Roboflow's
[platform pricing tiers](https://roboflow.com/pricing).

|                         | Open Access | With API Key |
|-------------------------|-------------|--------------|
| [Pre-Trained Models](https://inference.roboflow.com/quickstart/aliases/#supported-pre-trained-models) | ✅ | ✅
| [Foundation Models](https://inference.roboflow.com/foundation/about/) | ✅ | ✅
| [Video Stream Management](https://inference.roboflow.com/workflows/video_processing/overview/) | ✅ | ✅
| [Dynamic Python Blocks](https://inference.roboflow.com/workflows/custom_python_code_blocks/) | ✅ | ✅
| [Public Workflows](https://inference.roboflow.com/workflows/about/) | ✅ | ✅
| [Private Workflows](https://docs.roboflow.com/workflows/create-a-workflow) |  | ✅
| [Fine-Tuned Models](https://roboflow.com/train) |  | ✅
| [Universe Models](https://roboflow.com/universe) |  | ✅
| [Active Learning](https://inference.roboflow.com/workflows/blocks/roboflow_dataset_upload/) |  | ✅
| [Serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) |  | ✅
| [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) |  | ✅
| [Commercial Model Licensing](https://roboflow.com/licensing) |  | Paid
| [Device Management](https://docs.roboflow.com/roboflow-enterprise) |  | Enterprise
| [Model Monitoring](https://docs.roboflow.com/deploy/model-monitoring) |  | Enterprise

## Managed Compute

If you don't want to manage your own infrastructure for self-hosting, Roboflow offers a hosted Inference Server via [one-click Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) (CPU and GPU machines) billed hourly, or simple models and Workflows via our [serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) billed per API-call.

Roboflow offers a [generous free-tier](https://roboflow.com/pricing) to get started.

## Local Installation

Self-hosting for production or local development is easy. We recommend using Docker
to manage your installation because machine learning dependencies are often
fragile and finicky. On most systems, the easiest way to get started is to use our CLI
to choose the right docker image with the `inference server start` command:

```bash
pip install inference-cli
inference server start
```

For detailed instructions on various systems, see our [installation guide](../install/index.md)
then see [next steps](./next.md) to connect to your new Inference Server.