## Setup

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment, 
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If 
you haven't installed Docker yet, you can get it from [Docker's official website](https://www.docker.com/get-started).

## Pull

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

    === "GPU + TensorRT"
        Official Roboflow Inference Server Docker Image for Nvidia GPU with TensorRT Runtime Targets.
    
        ```
        docker pull roboflow/roboflow-inference-server-trt
        ```

    === "Jetson 4.5.x"
        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 4.5.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-4.5.0
        ```

    === "Jetson 4.6.x"
        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 4.6.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-4.6.1
        ```

    === "Jetson 5.x"
        Official Roboflow Inference Server Docker Image for Nvidia Jetson JetPack 5.x Targets.

        ```
        docker pull roboflow/roboflow-inference-server-jetson-5.1.1
        ```

## Run

Once you have a Docker image (either built locally or pulled from Docker Hub), you can run the Roboflow Inference 
Server in a container. 

!!! example "docker run"

    === "x86 CPU"
        ```
        docker run --net=host \
        roboflow/roboflow-inference-server-cpu:latest
        ```

    === "arm64 CPU"
        ```
        docker run -p 9001:9001 \
        roboflow/roboflow-inference-server-cpu:latest
        ```

    === "GPU"
        ```
        docker run --network=host --gpus=all \
        roboflow/roboflow-inference-server-gpu:latest
        ```

    === "GPU + TensorRT"
        ```
        docker run --network=host --gpus=all \
        roboflow/roboflow-inference-server-trt:latest
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

You may add the flag `-e API_KEY=<YOUR API KEY>` to your `docker run` command so that you do not need to provide a Roboflow API key in your requests. Substitute `<YOUR API KEY>` with your Roboflow API key. Learn how to retrieve your [Roboflow API key here](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

You may add the flag `-v $(pwd)/cache:/cache` to create a cache folder on your home device so that you do not need to redownload or recompile model artifacts upon inference container reboot. You can also (preferably) store artificats in a [docker volume](https://docs.docker.com/storage/volumes/) named `inference-cache` by adding the flag `-v inference-cache:/cache`.

## Build

To build a Docker image locally, first clone the Inference Server repository.

```bash
git clone https://github.com/roboflow/inference
```

Choose a Dockerfile from the following options, depending on the hardware you want to run Inference Server on.

!!! example "docker build"

    === "x86 CPU"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.cpu \
        -t roboflow/roboflow-inference-server-cpu .
        ```
    
    === "arm64 CPU"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.cpu \
        -t roboflow/roboflow-inference-server-cpu .
        ```
    
    === "GPU"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.gpu \
        -t roboflow/roboflow-inference-server-gpu .
        ```

    === "GPU + TensorRT"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.trt \
        roboflow/roboflow-inference-server-trt .
        ```

    === "Jetson 4.5.x"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.jetson \
        -t roboflow/roboflow-inference-server-jetson-4.5.0 .
        ```

    === "Jetson 4.6.x"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.jetson \
        -t roboflow/roboflow-inference-server-jetson-4.6.1 .
        ```

    === "Jetson 5.x"
        ```
        docker build \
        -f dockerfiles/Dockerfile.onnx.jetson.5.1.1 \
        -t roboflow/roboflow-inference-server-jetson-5.1.1 .
        ```
