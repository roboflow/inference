# Install on NVIDIA Jetson

## Overview

Jetson is NVIDIA’s line of compact, power-efficient modules designed to run AI and deep learning workloads at the edge. They combine a GPU, CPU, and neural accelerators on a single board, making them ideal for robotics, drones, smart cameras, and other embedded applications where you need real-time computer vision or inference without a cloud connection. For more details, see NVIDIA’s official Jetson overview:  
https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/

## Prerequisites

**Disk Space**: Allocate at least 10 GB free for the Roboflow Jetson image (8.14 GB). 

**JetPack Version**: Must be running a supported JetPack (4.5, 4.6, 5.x, or 6.x).

**Recommended Hardware**: For best performance while running Inference, we recommend an NVIDIA Orin NX 16 GB or above. 

 **Docker & NVIDIA Container Toolkit**  
  - Requires Docker + NVIDIA runtime so containers can access the GPU.  
  - Instead of detailing installation here, follow these instructions:  
    - **Docker install:**  
      https://docs.docker.com/engine/install/ubuntu/  
    - **NVIDIA Container Toolkit:**  
      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

We have specialized containers built with support for hardware acceleration on JetPack L4T.
To automatically detect your JetPack version and use the right container with good default settings
run:

```bash
pip install inference-cli
inference server start
```

## Manually Starting the Container

If you want more control of the container settings you can also start it
manually. Jetson devices with NVIDIA Jetpack are pre-configured with NVIDIA Container
Runtime and will be hardware accelerated out of the box:

=== "Jetpack 6.2"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        roboflow/roboflow-inference-server-jetson-6.2.0:latest
    ```

=== "Jetpack 6.0"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        roboflow/roboflow-inference-server-jetson-6.0.0:latest
    ```

=== "Jetpack 5"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        roboflow/roboflow-inference-server-jetson-5.1.1:latest
    ```

=== "Jetpack 4.6"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.
    
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-4.6.1:latest
    ```

=== "Jetpack 4.5"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.

    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-4.5.0:latest
    ```

## TensorRT
You can optionally enable [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's
model optimization runtime that will greatly increase your models' speed at the expense
of a heavy compilation and optimization step (sometimes 15+ minutes) the first time you
load each model.

Enable TensorRT by adding `TensorrtExecutionProvider` to the `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.

=== "Jetpack 6.2"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-6.2.0:latest
    ```

=== "Jetpack 6.0"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-6.0.0:latest
    ```

=== "Jetpack 5"
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-5.1.1:latest
    ```

=== "Jetpack 4.6"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.
    
    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-4.6.1:latest
    ```

=== "Jetpack 4.5"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.

    ```bash
    sudo docker run -d \
        --name inference-server \
        --runtime nvidia \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-jetson-4.5.0:latest
    ```

## Docker Compose

If you are using Docker Compose for your application, the equivalent yaml is:

=== "Jetpack 6.2"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-jetson-6.2.0:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        runtime: nvidia

        # Optionally: uncomment the following lines to enable TensorRT:
        # environment:
        #   ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "Jetpack 6.0"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-jetson-6.0.0:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        runtime: nvidia

        # Optionally: uncomment the following lines to enable TensorRT:
        # environment:
        #   ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "Jetpack 5"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-jetson-5.1.1:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        runtime: nvidia

        # Optionally: uncomment the following lines to enable TensorRT:
        # environment:
        #   ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "Jetpack 4.6"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.
    
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-jetson-4.6.1:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        runtime: nvidia

        # Optionally: uncomment the following lines to enable TensorRT:
        # environment:
        #   ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "Jetpack 4.5"
    !!! Warning
        Jetpack 4 is deprecated and will not receive future updates.
        Please migrate to Jetpack 6.

    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-jetson-4.5.0:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        runtime: nvidia

        # Optionally: uncomment the following lines to enable TensorRT:
        # environment:
        #   ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,CPUExecutionProvider]"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

--8<-- "install/using-your-new-server.md"

--8<-- "docs/install/enterprise-considerations.md"