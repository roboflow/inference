# Install on MacOS

## OSX Native App (Apple Silicon)

You can now run Roboflow Inference Server on your Apple Silicon Mac using our native desktop app!

Simply download the latest DMS disk image from the latest release on Github.
➡️ **[View Latest Release and Download Installers on Github](https://github.com/roboflow/inference/releases)**

### OSX Installation Steps
 - [Download the Roboflow Inference DMG](https://github.com/roboflow/inference/releases) disk image
 - Mount hte disk image by double clicking it
 - Drag the Roboflow Inference App to the Application Folder
 - Go to your Application Folder and double click the Roboflow Inference App to start the server

## Using Docker
=== "CPU"
    First, you'll need to
    [install Docker Desktop](https://docs.docker.com/desktop/setup/install/mac-install/).
    Then, use the CLI to start the container.

    ```bash
    pip install inference-cli
    inference server start
    ```

    ## Manually Starting the Container

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
        It's easiest to [get started with the CPU Docker](#using-docker) and switch to running
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