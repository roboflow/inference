# Local Installation

Inference is built to be run at the edge. It loads and executes model
weights and does computation locally. It can run fully offline (once
model weights are downloaded) but it's often useful to maintain a
network connection for interfacing with outside systems (like PLCs on
the local network, or remote systems for storing data and sending
notifications).

## Run via Docker

The preferred way to use Inference is via Docker 
(see [Why Docker](/understand/architecture/#why-docker)).

[Install Docker](https://docs.docker.com/engine/install/) (and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU acceleration if you have a CUDA-enabled GPU). Then run

```bash
pip install inference-cli
inference server start --dev
```

The `inference server start` command attempts to automatically choose 
and configure the optimal container to optimize performance on your machine.
Special installation notes and performance tips by device are also available.

## Dev Mode

The `--dev` parameter to `inference server start` starts in development mode.
This spins up a companion Jupyter notebook server with a quickstart guide on 
[`localhost:9002`](http://localhost:9002). Dive in there for a whirlwind tour
of your new Inference Server's functionality!

## Connecting to the Server

Now you're ready to connect your camera streams and
[start building & deploying Workflows in the UI](https://app.roboflow.com/workflows)
or [interacting with your new server](https://inference.roboflow.com/workflows/create_and_run/)
via its API.