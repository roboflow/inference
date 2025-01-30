# Install on NVIDIA Jetson

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

=== "Jetpack 6"
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

=== "Jetpack 6"
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

=== "Jetpack 6"
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