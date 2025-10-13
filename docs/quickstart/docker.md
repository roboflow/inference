## Setup

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from <a href="https://www.docker.com/get-started" target="_blank">Docker's official website</a>.

## Set up a Docker Inference Server via `inference server start``

Another easy way to run the Roboflow Inference Server with Docker is via the command line.

First, [Install the CLI](../models.md#-cli).

Running the Inference Server is as simple as running the following command:

```bash
inference server start
```

This will pull the appropriate Docker image for your machine and start the Inference Server on port 9001. You can then send requests to the server to get predictions from your model, as described in [Quickstart Guide](../quickstart/run_model_on_image.md).

Once you have your inference server running, you can check its status with the following command:

```bash
inference server status
```

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson or TensorRT Runtime inference server images, pull the images directly following the [instructions below](#step-1-pull-from-docker-hub).

## Manually Set Up a Docker Container

### Step #1: Pull from Docker Hub

If you don't wish to build the Docker image locally or prefer to use the official releases, you can directly pull the
pre-built images from the Docker Hub. These images are maintained by the Roboflow team and are optimized for various
hardware configurations.

!!! example "docker pull"

    === "x86 CPU"

        Official Roboflow Inference Server Docker Image for x86 CPU Targets.

        ```
        docker pull roboflow/roboflow-inference-server-cpu
        ```

    === "arm64 CPU"

        Official Roboflow Inference Server Docker Image for ARM CPU Targets.

        ```
        docker pull roboflow/roboflow-inference-server-cpu
        ```

    === "GPU"

        Official Roboflow Inference Server Docker Image for Nvidia GPU Targets.

        ```
        docker pull roboflow/roboflow-inference-server-gpu
        ```

    === "Jetson 4.5.x" (Deprecated)

        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 4.5.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-4.5.0
        ```

    === "Jetson 4.6.x" (Deprecated)

        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 4.6.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-4.6.1
        ```

    === "Jetson 5.x"

        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 5.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-5.1.1
        ```

    === "Jetson 6.x"

        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 6.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-6.0.0
        ```

### Step #2: Run the Docker Container

Once you have a Docker image (either built locally or pulled from Docker Hub), you can run the Roboflow Inference
Server in a container.

!!! example "docker run"

    === "x86 CPU"
        ```
        docker run -it --net=host \
        roboflow/roboflow-inference-server-cpu:latest
        ```

    === "arm64 CPU"
        ```
        docker run -p 9001:9001 \
        roboflow/roboflow-inference-server-cpu:latest
        ```

    === "GPU"
        ```
        docker run -it --network=host --gpus=all \
        roboflow/roboflow-inference-server-gpu:latest
        ```

    === "Jetson 4.5.x"
        ```
        docker run --privileged --net=host --runtime=nvidia \
        roboflow/roboflow-inference-server-jetson-4.5.0:latest
        ```

    === "Jetson 4.6.x"
        ```
        docker run --privileged --net=host --runtime=nvidia \
        roboflow/roboflow-inference-server-jetson-4.6.1:latest
        ```

    === "Jetson 5.x"
        ```
        docker run --privileged --net=host --runtime=nvidia \
        roboflow/roboflow-inference-server-jetson-5.1.1:latest
        ```

    **_Note:_** The Jetson images come with TensorRT dependencies. To use TensorRT acceleration with your model, pass an additional environment variable at runtime `-e ONNXRUNTIME_EXECUTION_PROVIDERS=TensorrtExecutionProvider`. This can improve inference speed, however, this also incurs a costly startup expense when the model is loaded.
    **_Note:_** On Windows and macOS, you may need to use `-p 9001:9001` instead of `--net=host` to expose the port to the host machine.

You may add the flag `-e ROBOFLOW_API_KEY=<YOUR API KEY>` to your `docker run` command so that you do not need to provide a Roboflow API key in your requests. Substitute `<YOUR API KEY>` with your Roboflow API key. Learn how to retrieve your <a href="https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key" target="_blank">Roboflow API key here</a>.

You may add the flag `-v $(pwd)/cache:/tmp/cache` to create a cache folder on your home device so that you do not need to redownload or recompile model artifacts upon inference container reboot. You can also (preferably) store artificats in a <a href="https://docs.docker.com/storage/volumes/" target="_blank">docker volume</a> named `inference-cache` by adding the flag `-v inference-cache:/tmp/cache`.

### Advanced: Build a Docker Container from Scratch

To build a Docker image locally, first clone the Inference Server repository.

```bash
git clone https://github.com/roboflow/inference
```

Choose a Dockerfile from the following options, depending on the hardware you want to run Inference Server on.

!!! example "docker build"

    === "x86 CPU"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.cpu \
        -t roboflow/roboflow-inference-server-cpu .
        ```

    === "arm64 CPU"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.cpu \
        -t roboflow/roboflow-inference-server-cpu .
        ```

    === "GPU"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.gpu \
        -t roboflow/roboflow-inference-server-gpu .
        ```

    === "Jetson 4.5.x"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.jetson \
        -t roboflow/roboflow-inference-server-jetson-4.5.0 .
        ```

    === "Jetson 4.6.x"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.jetson \
        -t roboflow/roboflow-inference-server-jetson-4.6.1 .
        ```

    === "Jetson 5.x"
        ```
        docker build \
        -f docker/dockerfiles/Dockerfile.onnx.jetson.5.1.1 \
        -t roboflow/roboflow-inference-server-jetson-5.1.1 .
        ```
