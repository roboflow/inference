# Installation

You can install `inference` in a [Python>=3.9,<3.13](https://www.python.org/) environment.

!!! example "Installation Command"

    === "CPU"
        ```bash
        pip install inference
        ```

    === "Nvidia GPU"
        ```bash
        pip install inference-gpu
        ```

# Quickstart

With the following code snippet, we can load a model and then we used that model's `infer(...)` method to run an image through it.

```python
# import a utility function for loading Roboflow models
from inference import get_model

# define the image url to use for inference
image = "https://media.roboflow.com/inference/people-walking.jpg"

# load a pre-trained yolov8n model
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)
```

!!! note
	
	For a more detailed example, please refer to the tutorial on [running a model](../quickstart/run_a_model).

# Choosing a Deployment Method

There are three primary ways to deploy Inference:

* [Serverless Hosted API](#serverless-hosted-api) - for smaller image models.
* [Dedicated Deployment](#dedicated-deployments) - for bigger models and streaming video.
* [Self Hosted](#self-hosting) - on your own edge device or server.

Each has pros and cons and which one you should choose depends on your particular
use-case and organizational constraints.

|                         | Serverless | Dedicated | Self-Hosted |
|-------------------------|------------|-----------|-------------|
| Workflows               | ✅         | ✅         | ✅          |
| Basic Logic Blocks      | ✅         | ✅         | ✅          |
| Pre-Trained Models      | ✅         | ✅         | ✅          |
| Fine-Tuned Models       | ✅         | ✅         | ✅          |
| Universe Models         | ✅         | ✅         | ✅          |
| Active Learning         | ✅         | ✅         | ✅          |
| Model Monitoring        | ✅         | ✅         | ✅          |
| Foundation Models       |            | ✅         | ✅          |
| Video Stream Management |            | ✅         | ✅          |
| Dynamic Python Blocks   |            | ✅         | ✅          |
| Device Management       |            | ✅         | ✅          |
| Access Local Devices    |            |            | ✅          |
| Can Run Offline         |            |            | ✅          |
| Billing                 | Per-Call   | Hourly     | [See Below](#self-hosting) |

## Cloud Hosting

By far the easiest way to get started is with Roboflow's managed services. You can
jump straight to building without having to setup any infrastructure. It's often
the front-door to using Inference even for those who know they will eventually want
to self host.

There are two cloud hosted offerings with different targeted use-cases, capabilities,
and pricing models.

### Serverless Hosted API

The [Serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) supports running Workflows on
pre-trained & fine-tuned models, chaining models, basic logic, visualizations, and
external integrations.

It supports cloud-hosted VLMs like ChatGPT and Anthropic Claude, but does not support
running heavy models like Florence-2 or SAM 2. It also does not support streaming
video.

The Serverless API scales down to zero when you're not using it (and up to infinity
under load) with quick (a couple of seconds) cold-start time. You pay per model
inference with no minimums. Roboflow's free tier credits may be used.

### Dedicated Deployments

[Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) are single-tenant virtual machines that
are allocated for your exclusive use. They can optionally be configured with a GPU
and used in development mode (where you may be evicted if capacity is needed for a
higher priority task & are limited to 3-hour sessions) or production mode (guaranteed
capacity and no session time limit).

On a Dedicated Deployment, you can stream video, run custom Python code, access
heavy foundation models like SAM 2, Florence-2, and Paligemma (including your fine-tunes
of those models), and install additional dependencies. They are much higher performance
machines than the instances backing the Serverless Hosted API.

Scale-up time is on the order of a minute or two.

!!! info "Dedicated Deployments Availability"
    Dedicated Deployments are only available to Roboflow Workspaces with an active
    subscription (and are not available on the free trial). They are billed hourly.

## Self Hosting

[Running at the edge](/install/index.md) is a core priority and focus area of Inference. For many use-cases
latency matters, bandwidth is limited, interfacing with local devices is key, and
resiliency to Internet outages is mandatory.

Running locally on a development machine, an AI computer, or an edge device is as simple
as starting a Docker container.

!!! info "Self-Hosted Pricing"
    Basic usage of self-hosted Inference Servers is completely free.
    
    Workflows and Models that require
    [a Roboflow API Key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)
    to access Roboflow Cloud powered features (for example: the private model repository)
    are [metered and consume credits](https://roboflow.com/pricing)
    (which cost money after a generous free tier is used up) based on the number of images
    or the hours of video processed.

Detailed [installation instructions and device-specific performance tips are here](/install/index.md).

## Bring Your Own Cloud

Sometimes enterprise compliance policies regarding sensitive data requires running
workloads on-premises. This is supported via
[self-hosting on your own cloud](../install/cloud/index.md). Billing is the same
as for self-hosting on an edge device.

<br />

# Next Steps

Once you've decided on a deployment method and have a server running,
[interfacing with it is easy](../start/next.md).
