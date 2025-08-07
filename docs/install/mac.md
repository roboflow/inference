# Install on MacOS

You can install Inference on macOS in two ways:

1. With our dedicated macOS Installer (for Apple Silicon)
2. With Docker

## macOS Installer (Apple Silicon)

You can install and run Roboflow Inference on your macOS machine using a native desktop application.

You must use an Apple Silicon machine to use this installation method.

To get started, download the macOS `.dmg` file from the [latest release of Inference on Github](https://github.com/roboflow/inference/releases).

Once you have downloaded the `.dmg` file, copy the Roboflow Inference application to the Application Folder.

![](https://docs.roboflow.com/~gitbook/image?url=https%3A%2F%2F2667452268-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FMR3m936tBXGm5QsAcPwe%252Fuploads%252Fnx7uxN0cM7IJRm1nR7Vn%252FScreenshot%25202025-05-29%2520at%252010.37.33.png%3Falt%3Dmedia%26token%3Dd59b8be9-4a45-439e-8142-f6118892562b&width=768&dpr=3&quality=100&sign=dee977be&sv=2)

To start your server, go your Application Folder and double click the Roboflow Inference application. Your Inference server will then start.

## Using Docker
=== "CPU"
    First, you'll need to
    [install Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/).
    Then, use the CLI to start the container.

    ```bash
    pip install inference-cli
    inference server start
    ```

    ## Manually Start the Container

    If you want more control of the container settings you can also start it
    manually:

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
    Apple does not yet support
    [passing the Metal Performance Shader (MPS) device to Docker](https://github.com/pytorch/pytorch/issues/81224)
    so hardware acceleration is not possible inside a container on Mac.

    !!! Tip
        It's easiest to [get started with the CPU Docker](#cpu) and switch to running
        outside of Docker with MPS acceleration later if you need more speed.

    We recommend using
    [`pyenv`](https://github.com/pyenv/pyenv) and
    [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv)
    to manage your Python environments on Mac (especially because, in 2025, [homebrew](https://brew.sh) is
    defaulting to Python 3.13 which is not yet compatible with several of the machine learning dependencies
    that Inference uses).

    Once you have installed and setup `pyenv` and `pyenv-virtualenv` (be sure to follow the full instructions
    for setting up your shell), create and activate an `inference` virtual environment with Python 3.12:

    ```bash
    pyenv install 3.12
    pyenv virtualenv 3.12 inference
    pyenv activate inference
    ```

    To install and run the server outside of Docker, clone the repo, install the dependencies,
    copy `cpu_http.py` into the top level of the repo, and start the server with
    [`uvicorn`](https://www.uvicorn.org/):

    ```bash
    git clone https://github.com/roboflow/inference.git
    cd inference
    pip install .
    cp docker/config/cpu_http.py .
    uvicorn cpu_http:app --port 9001 --host 0.0.0.0
    ```

    Your server is now running at [`localhost:9001`](http://localhost:9001) with MPS acceleration.

--8<-- "install/using-your-new-server.md"