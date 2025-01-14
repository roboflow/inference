# Install on Windows

First, you'll need to [install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/).
Then, use the CLI to start the container.

=== "CPU"
    ```bash
    pip install inference-cli
    inference server start
    ```

=== "GPU"
    To access the GPU, you'll need to ensure you've installed the up to date NVIDIA drivers and the
    latest version of WSL2 and that the WSL 2 backend is configured in Docker.
    [Follow the setup instructions from Docker](https://docs.docker.com/desktop/features/gpu/).

    Then, use the CLI to start the container:

    ```bash
    pip install inference-cli
    inference server start
    ```

!!! Note
    If the `pip install` command fails, you may need to
    [install Python](https://www.python.org/downloads/release/python-3128/#:~:text=Windows%20installer%20(64%2Dbit))
    first. Once you have Python version 3.12, 3.11, 3.10, or 3.9 on your machine, retry the command.

## Manually Starting the Container

If you want more control of the container settings you can also start it
manually.

=== "CPU"
    The core CPU Docker image includes support for OpenVINO acceleration on
    x64 CPUs via onnxruntime. Heavy models like SAM2 and CogVLM may run too
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
    via NVIDIA-Docker. Ensure you have
    [setup Docker to access the GPU](https://docs.docker.com/desktop/features/gpu/)
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
    load it.

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

## Enterprise Considerations

[A Helm Chart](https://github.com/roboflow/inference/tree/main/inference/enterprise/helm-chart)
is available for enterprise cloud deployments. Enterprise networking solutions to support
deployment in OT networks are also available upon request.

Roboflow also offers customized support and installation packages and
[a pre-configured edge-device](https://roboflow.com/hardware)
suitable for rapid prototyping. [Contact our sales team](https://roboflow.com/sales)
if you're part of a large organization and interested in learning more.