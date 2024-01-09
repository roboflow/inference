<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />
</div>

## Roboflow Inference CLI

Roboflow Inference CLI offers a lightweight interface for running the Roboflow inference server locally or the Roboflow Hosted API.

To create custom inference server Docker images, go to the parent package, [Roboflow Inference](https://pypi.org/project/inference/).

[Roboflow](https://roboflow.com) has everything you need to deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

## Examples

### inference server start

Starts a local inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from [Docker's official website](https://www.docker.com/get-started).

The CLI will automatically detect the device you are running on and pull the appropriate Docker image.

```bash
inference server start --port 9001 [-e {optional_path_to_file_with_env_variables}]
```

Parameter `--env-file` (or `-e`) is the optional path for .env file that will be loaded into inference server
in case that values of internal parameters needs to be adjusted. Any value passed explicitly as command parameter
is considered as more important and will shadow the value defined in `.env` file under the same target variable name.

#### Development Mode

Use the `--dev` flag to start the Inference Server in development mode. Development mode enables the Inference Server's built in notebook environment for easy testing and development.

### inference server status

Checks the status of the local inference server.

```bash
inference server status
```

### inference server stop

Stops the inference server.

```bash
inference server stop
```

### inference infer

Runs inference on a single image. It takes a path to an image, a Roboflow project name, model version, and API key, and will return a JSON object with the model's predictions. You can also specify a host to run inference on our hosted inference server.

#### Local image

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key
```

#### Hosted image

```bash
inference infer https://[YOUR_HOSTED_IMAGE_URL] --project-id my-project --model-version 1 --api-key my-api-key
```

#### Hosted API inference

```bash
inference infer ./image.jpg --project-id my-project --model-version 1 --api-key my-api-key --host https://detect.roboflow.com
```

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson specific inference server images, check out the [Roboflow Inference](https://pypi.org/project/inference/) package, or pull the images directly following instructions in the official [Roboflow Inference documentation](https://inference.roboflow.com/quickstart/docker/#pull-from-docker-hub).
