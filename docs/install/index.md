# Install Inference

Inference runs as a Docker container. This ensures all dependencies, model weights, and application caching can be managed in one place.

You can make HTTP requests to the Docker container to run models and Workflows.

Inference is designed to run both on the edge and in the cloud.

## Run on the Edge

Inference runs on many edge and personal computing devices. Choose your device below to find the installation guide you need:

- [Windows](install/windows.md)
- [macOS](install/mac.md)
- [NVIDIA Jetson](install/jetson.md)
- [Raspberry Pi](install/jetson.md)

## Run in the Cloud

You can run Inference on servers in the cloud. Choose your cloud below to find the installation guide you need:

- [Amazon Web Services](install/cloud/aws.md)
- [Microsoft Azure](install/cloud/azure.md)
- [Google Cloud Platform](install/cloud/gcp.md)

## Run with Roboflow

You can run Workflows developed with Inference in the Roboflow Cloud. You can use:

- [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments), cloud CPUs or GPUs dedicated to your Workflows.
- [Serverless API](https://docs.roboflow.com/deploy/serverless), which auto-scales with your workloads.

Running on another device? [Learn more about the architectures on which Inference is designed to run](/install/other/).