---
description: Command-line guide for starting the server, running predictions, benchmarking, and deploying Roboflow Inference.
---

# Inference CLI

<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />
</div>

Roboflow Inference CLI is a command-line interface for the `inference` ecosystem, providing an easy way to:

- run and manage [`inference` server](./cli_commands/server.md) locally
- process data with [Workflows](../workflows/about.md)
- [benchmark](./cli_commands/benchmark.md) `inference` performance 
- make [predictions](./cli_commands/infer.md) from your models
- deploy `inference` server in [cloud](./cli_commands/cloud.md)

### Installation

```bash
pip install inference-cli
```

Note that if you have installed the `inference` Python package, the CLI extensions are already included.

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU (including Jetson)

For Jetson specific inference server images, check out the <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a> package, or pull the images directly following instructions in the official [Roboflow Inference documentation](../quickstart/docker.md#step-1-pull-from-docker-hub).
