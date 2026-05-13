# About Inference Server

The Inference Server is a standalone microservice that wraps the [Inference](../start/overview.md) Python package and exposes it over an HTTP API.

## Deployment Options

You can self-host Inference Server, or use our hosted APIs:

- [Serverless API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) — hosted by Roboflow, scales to zero, pay per inference.
- [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) — hosted by Roboflow, single-tenant VMs with optional GPU.
- [Self-Hosted](../install/index.md) — run on your own edge hardware (Raspberry Pi, NVIDIA GPU, NVIDIA Jetson...) with Docker
- [Deploy in Your Own Cloud](../install/cloud.md) — run on your own cloud infrastructure (AWS, GCP, Azure) with Docker

You can interact with an Inference Server using [Inference SDK](../inference_helpers/inference_sdk.md).

## Running with Dockers

Before you begin, ensure that you have [Docker installed](https://www.docker.com/get-started) on your machine. The easiest way to start the Inference Server is with the [Inference CLI](../inference_helpers/inference_cli.md):

```bash
pip install inference-cli && inference server start
```

This pulls the appropriate Docker image for your machine (with pre-installed dependencies) and starts the Inference Server on port 9001.
Check server status:

```bash
inference server status
```

## Manually Set Up a Docker Container

`inference server start` runs `docker run` under the hood with recommended security settings, caching, and platform-specific options.

If you want to manually start the inference server container, refer to **Manually Starting the Container** section in your platform's install guide:

- [Linux](../install/linux/#manually-starting-the-container)
- [Windows](../install/windows/#manually-starting-the-container)
- [Mac](../install/mac/#manually-starting-the-container)
- [Jetson](../install/jetson/#manually-starting-the-container)
- [Raspberry Pi](../install/raspberry-pi/#manually-starting-the-container)
