# Install on Windows

You can install Inference on Windows in two ways:

1. With our dedicated Windows Installer (for x86)
2. With Docker

## Windows Installer (x86)

You can install and run Roboflow Inference on your Windows machine using a native desktop application.

To get started, download the Windows installer from the [latest release of Inference on Github](https://github.com/roboflow/inference/releases).

Once you have downloaded the installer, open the installer software and follow the on-screen instructions.

When the install is finished it will offer to launch the Inference server after the setup completes.

To stop the inference server, close the terminal window the application opens.

To start your Inference server, open Roboflow Inference from your Start Menu.

## Using Docker

First, you'll need to [install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/).

Then, use the Roboflow Inference CLI to start the container.

=== "CPU"
    ```cmd
    pip install inference-cli
    inference server start
    ```

=== "GPU"
    To access the GPU, you'll need to ensure you've installed the up to date NVIDIA drivers and the
    latest version of WSL2 and that the WSL 2 backend is configured in Docker.
    [Follow the setup instructions from Docker](https://docs.docker.com/desktop/features/gpu/).

    Then, use the CLI to start the container:

    ```cmd
    pip install inference-cli
    inference server start
    ```

!!! Note
    If the `pip install` command fails, you may need to
    [install Python](https://www.python.org/downloads/release/python-3128/#:~:text=Windows%20installer%20(64%2Dbit))
    first. Once you have Python version 3.12, 3.11, 3.10, or 3.9 on your machine, retry the command.

## Manually Start the Container

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

    ```cmd
    docker run -d ^
        --name inference-server ^
        --read-only ^
        -p 9001:9001 ^
        --volume "%USERPROFILE%\.inference\cache:/tmp:rw" ^
        --security-opt="no-new-privileges" ^
        --cap-drop="ALL" ^
        --cap-add="NET_BIND_SERVICE" ^
        roboflow/roboflow-inference-server-cpu:latest
    ```

=== "GPU"
    The GPU container adds support for hardware acceleration on cards that support CUDA
    via NVIDIA-Docker. Ensure you have
    [setup Docker to access the GPU](https://docs.docker.com/desktop/features/gpu/)
    then add `--gpus all` to the `docker run` command:

    ```cmd
    docker run -d ^
        --name inference-server ^
        --gpus all ^
        --read-only ^
        -p 9001:9001 ^
        --volume "%USERPROFILE%\.inference\cache:/tmp:rw" ^
        --security-opt="no-new-privileges" ^
        --cap-drop="ALL" ^
        --cap-add="NET_BIND_SERVICE" ^
        roboflow/roboflow-inference-server-gpu:latest
    ```

=== "TensorRT"
    With the GPU container you can optionally enable
    [TensorRT](https://developer.nvidia.com/tensorrt), NVIDIA's model optimization
    runtime that will greatly increase your models' speed at the expense of a heavy
    compilation and optimization step (sometimes 15+ minutes) the first time you
    load each model.

    You can enable TensorRT by adding `TensorrtExecutionProvider` to the `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.

    ```cmd
    docker run -d ^
        --name inference-server ^
        --gpus all ^
        --read-only ^
        -p 9001:9001 ^
        --volume "%USERPROFILE%\.inference\cache:/tmp:rw" ^
        --security-opt="no-new-privileges" ^
        --cap-drop="ALL" ^
        --cap-add="NET_BIND_SERVICE" ^
        -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]" ^
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
          - "${USERPROFILE}/.inference/cache:/tmp:rw"
    
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
          - "${USERPROFILE}/.inference/cache:/tmp:rw"
    
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
          - "${USERPROFILE}/.inference/cache:/tmp:rw"
    
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