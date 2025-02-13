# Install on Linux

The easiest way to start the correct container optimized for your machine
and with good default settings (like a cache volume and a secure, non-privileged
execution mode) is to use the CLI to choose and start the container using the
`inference server start` command.
(Note: you will need to [install docker](https://docs.docker.com/engine/install/) first):

```bash
pip install inference-cli
inference server start
```

## Manually Starting the Container

If you want more control of the container settings you can also start it
manually.

=== "CPU"
    The core CPU Docker image includes support for OpenVINO acceleration on
    x64 CPUs via onnxruntime. Heavy models like SAM2 may run too
    slowly (dozens of seconds per image) to be practical (and you should
    look into getting a CUDA-capable GPU if you want to use them).

    The primary use-cases for CPU inference are processing still images
    (eg for NSFW classification of uploads or document verification) or infrequent
    sampling of frames on a video (eg for occupancy tracking of a parking lot).

    To get started with CPU inference, use the `roboflow/roboflow-inference-server-cpu:latest`
    container.

    ```bash
    sudo docker run -d \
        --name inference-server \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        roboflow/roboflow-inference-server-cpu:latest
    ```

=== "GPU"
    The GPU container adds support for hardware acceleration on cards that support CUDA
    via NVIDIA-Docker. First follow the
    [NVIDIA Container Toolkit isntallation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    then add `--gpus all` to the `docker run` command:

    ```bash
    sudo docker run -d \
        --name inference-server \
        --gpus all \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        roboflow/roboflow-inference-server-gpu:latest
    ```

=== "TensorRT"
    With the GPU container you can optionally enable
    [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's model optimization
    runtime that will greatly increase your models' speed at the expense of a heavy
    compilation and optimization step (sometimes 15+ minutes) the first time you
    load each model.

    You can enable TensorRT by adding `TensorrtExecutionProvider` to the `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.

    ```bash
    sudo docker run -d \
        --name inference-server \
        --gpus all \
        --read-only \
        -p 9001:9001 \
        --volume ~/.inference/cache:/tmp:rw \
        --security-opt="no-new-privileges" \
        --cap-drop="ALL" \
        --cap-add="NET_BIND_SERVICE" \
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]" \
        roboflow/roboflow-inference-server-gpu:latest
    ```

## Docker Compose

If you are using Docker Compose for your application, the equivalent yaml is:

=== "CPU"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-cpu:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "GPU"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-gpu:latest
    
        read_only: true
        ports:
          - "9001:9001"

        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all
                  capabilities: [gpu]
    
        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

=== "TensorRT"
    ```yaml
    version: "3.9"
    
    services:
      inference-server:
        container_name: inference-server
        image: roboflow/roboflow-inference-server-gpu:latest
    
        read_only: true
        ports:
          - "9001:9001"
        
        volumes:
          - "${HOME}/.inference/cache:/tmp:rw"
    
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: all
                  capabilities: [gpu]

        environment:
          ONNXRUNTIME_EXECUTION_PROVIDERS: "[TensorrtExecutionProvider,CUDAExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]"

        security_opt:
          - no-new-privileges
        cap_drop:
          - ALL
        cap_add:
          - NET_BIND_SERVICE
    ```

--8<-- "install/using-your-new-server.md"

--8<-- "docs/install/enterprise-considerations.md"