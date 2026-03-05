## Setup

Before you begin, ensure that you have [Docker installed](https://www.docker.com/get-started) on your machine. 

Docker provides a containerized environment, allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system.

## Set up a Docker Inference Server

Another easy way to run the Roboflow Inference Server with Docker is via the command line.

[Install the CLI](../inference_helpers/inference_cli.md) and use it to start the Inference Server:

```bash
pip install inference-cli && inference server start
```

This will pull the appropriate Docker image for your machine and start the Inference Server on port 9001. You can then send requests to the server to get predictions from your model, as described in [Quickstart Guide](../quickstart/run_model_on_image.md).

Once you have your inference server running, you can check its status with the following command:

```bash
inference server status
```

## Manually Set Up a Docker Container

`inference server start` will do `docker run` commands under the hood with recommended security settings, caching, and platform-specific options.

If you want to manually start the inference server container, refer to **Manually Starting the Container** section in your platform's install guide:

- [Linux](../install/linux/#manually-starting-the-container)
- [Windows](../install/windows/#manually-starting-the-container)
- [Mac](../install/mac/#manually-starting-the-container)
- [Jetson](../install/jetson/#manually-starting-the-container)
- [Raspberry Pi](../install/raspberry-pi/#manually-starting-the-container)
