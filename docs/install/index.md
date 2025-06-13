# 🚀 Native Desktop Apps

You can now run Roboflow Inference Server on your Windows or macOS machine with our native desktop applications! This is the quickest and most effortless way to get up and running.

Simply download the latest installer for your operating system.  You can find these attached to our **latest release on GitHub**.

➡️ **[View Latest Release and Download Installers on Github](https://github.com/roboflow/inference/releases)**

### Windows (x86)
 - [Download the latest installer](https://github.com/roboflow/inference/releases) and run it to install Roboflow Inference
 - When the install is finished it will offer to launch the Inference server after the setup completes
 - To stop the inference server simply close the terminal window it opens
 - To start it again later, you can find Roboflow Inference in your Start Menu

### MacOS (Apple Silicon)
 - [Download the Roboflow Inference DMG](https://github.com/roboflow/inference/releases) disk image
 - Mount hte disk image by double clicking it
 - Drag the Roboflow Inference App to the Application Folder
 - Go to your Application Folder and double click the Roboflow Inference App to start the server




---

# Local Installation using Docker

Inference is built to be run at the edge. It loads and executes model
weights and does computation locally. It can run fully offline (once
model weights are downloaded) but it's often useful to maintain a
network connection for interfacing with outside systems (like PLCs on
the local network, or remote systems for storing data and sending
notifications).

## Run via Docker

The preferred way to use Inference is via Docker
(see [Why Docker](/understand/architecture.md#why-docker)).

[Install Docker](https://docs.docker.com/engine/install/) (and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU acceleration if you have a CUDA-enabled GPU). Then run:

```bash
pip install inference-cli
inference server start
```

The `inference server start` command attempts to automatically choose
and configure the optimal container to optimize performance on your machine.
See [Using Your New Server](#using-your-new-server) for next steps.

!!! Tip
    Special installation notes and performance tips by device are also available.
    Browse the navigation on the left for detailed install guides.

## Dev Mode

The `--dev` parameter to `inference server start` starts in development mode.
This spins up a companion Jupyter notebook server with a quickstart guide on
[`localhost:9002`](http://localhost:9002). Dive in there for a whirlwind tour
of your new Inference Server's functionality!

```bash
inference server start --dev
```

--8<-- "install/using-your-new-server.md"