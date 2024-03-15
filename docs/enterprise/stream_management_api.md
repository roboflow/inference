# Stream Management

> [!IMPORTANT] 
> We require a Roboflow Enterprise License to use this in production. See inference/enterpise/LICENSE.txt for details.


## Overview

This feature is designed to cater to users requiring the execution of inference to generate predictions using
Roboflow object-detection models, particularly when dealing with online video streams.
It enhances the functionalities of the familiar `inference.Stream()` and `InferencePipeline()` interfaces, as found in
the open-source version of the library, by introducing a sophisticated management layer. The inclusion of additional
capabilities empowers users to remotely manage the state of inference pipelines through the HTTP management interface
integrated into this package.

This functionality proves beneficial in various scenarios, **including but not limited to**:

- Performing inference across multiple online video streams simultaneously.
- Executing inference on multiple devices that necessitate coordination.
- Establishing a monitoring layer to oversee video processing based on the `inference` package.

## Design

![Stream Management - design](https://storage.googleapis.com/com-roboflow-marketing/inference/stream_management_api_design.jpg)

## Example use-case

Joe aims to monitor objects within the footage captured by a fleet of IP cameras installed in his factory. After
successfully training an object-detection model on the Roboflow platform, he is now prepared for deployment. With four
cameras in his factory, Joe opts for a model that is sufficiently compact, allowing for over 30 inferences per second
on his Jetson devices. Considering this computational budget per device, Joe determines that he requires two Jetson
devices to efficiently process footage from all cameras, anticipating an inference throughput of approximately
15 frames per second for each video source.

To streamline the deployment, Joe chooses to deploy Stream Management containers to all available Jetson devices within
his local network. This setup enables him to communicate with each Jetson device via HTTP, facilitating the
orchestration of processing tasks. Joe develops a web app through which he can send commands to the devices and retrieve
metrics regarding the statuses of the video streams.

Finally, Joe implements a UDP server capable of receiving predictions, leveraging the `supervision` package to
effectively track objects in the footage. This comprehensive approach allows Joe to manage and monitor the
object-detection process seamlessly across his fleet of Jetson devices.

## How to run?

### In docker - using `docker compose`

The most prevalent use-cases are conveniently encapsulated with Docker Compose configurations, ensuring readiness for
immediate use. Nevertheless, in specific instances where custom configuration adjustments are required within Docker
containers, such as passing camera devices, alternative options may prove more suitable.

#### CPU-based devices

```bash
repository_root$ docker compose -f ./docker/dockerfiles/stream-management-api.compose-cpu.yaml up
```

#### GPU-based devices

```bash
repository_root$ docker compose -f ./docker/dockerfiles/stream-management-api.compose-gpu.yaml up
```

#### Jetson devices (`JetPack 5.1.1`)

```bash
repository_root$ docker-compose -f ./docker/dockerfiles/stream-management-api.compose-jetson.5.1.1.yaml up
```

**Disclaimer:** At Jetson devices, some operations (like container bootstrap or initialisation of model) takes more time
than for other ones. In particular - docker compose definition in current form do not define active awaiting
TCP socket port to be opened by Stream Manager - which means that initial requests to HTTP API may be responded with
HTTP 503.

### In docker - running API and stream manager containers separately

#### Run

##### CPU-based devices

```bash
docker run -d --name stream_manager --network host roboflow/roboflow-inference-stream-manager-cpu:latest
docker run -d --name stream_management_api --network host roboflow/roboflow-inference-stream-management-api:latest
```

##### GPU-based devices

```bash
docker run -d --name stream_manager --network host --runtime nvidia roboflow/roboflow-inference-stream-manager-gpu:latest
docker run -d --name stream_management_api --network host roboflow/roboflow-inference-stream-management-api:latest
```

##### Jetson devices (`JetPack 5.1.1`)

```bash
docker run -d --name stream_manager --network host --runtime nvidia roboflow/roboflow-inference-stream-manager-jetson-5.1.1:latest
docker run -d --name stream_management_api --network host roboflow/roboflow-inference-stream-management-api:latest
```

#### Configuration parameters

##### Stream Management API

- `STREAM_MANAGER_HOST` - hostname for stream manager container (alter with container name if `--network host` not used
  or used against remote machine)
- `STREAM_MANAGER_PORT` - port to communicate with stream manager (must match with stream manager container)

##### Stream Manager

- `PORT` - port at which server will be running
- one can mount volume under container's `/tmp/cache` to enable permanent storage of models - for faster inference
  pipelines initialisation
- at the level of this container the connectivity to camera must be enabled - so if device passing to docker must
  happen - it should happen at this stage

#### Build (Optional)

##### Stream Management API

```bash
docker build -t roboflow/roboflow-inference-stream-management-api:dev -f docker/dockerfiles/Dockerfile.stream_management_api .
```

##### Stream Manager

```bash
docker build -t roboflow/roboflow-inference-stream-manager-{device}:dev -f docker/dockerfiles/Dockerfile.onnx.{device}.stream_manager .
```

### Bare-metal deployment

In some cases, it would be required to deploy the application at host level. This is possible, although
client must resolve the environment in a way that is presented in Stream Manager and Stream Management API dockerfiles
appropriate for specific platform. Once this is done the following command should be run:

```bash
repository_root$ python -m inference.enterprise.stream_management.manager.app  # runs manager
```

```bash
repository_root$ python -m inference.enterprise.stream_management.api.app  # runs management API
```

## How to integrate?

After running `roboflow-inference-stream-management-api` container, HTTP API will be available under
`http://127.0.0.1:8080` (given that default configuration is used).

One can call `wget http://127.0.0.1:8080/openapi.json` to get OpenApi specification of API that can be rendered
<a href="https://editor.swagger.io/" target="_blank">here</a>

Example Python client is provided below:

```python
import requests
from typing import Optional

URL = "http://127.0.0.1:8080"

def list_pipelines() -> dict:
    response = requests.get(f"{URL}/list_pipelines")
    return response.json()


def get_pipeline_status(pipeline_id: str) -> dict:
    response = requests.get(f"{URL}/status/{pipeline_id}")
    return response.json()


def pause_pipeline(pipeline_id: str) -> dict:
    response = requests.post(f"{URL}/pause/{pipeline_id}")
    return response.json()


def resume_pipeline(pipeline_id: str) -> dict:
    response = requests.post(f"{URL}/resume/{pipeline_id}")
    return response.json()

def terminate_pipeline(pipeline_id: str) -> dict:
    response = requests.post(f"{URL}/terminate/{pipeline_id}")
    return response.json()

def initialise_pipeline(
    video_reference: str,
    model_id: str,
    api_key: str,
    sink_host: str,
    sink_port: int,
    max_fps: Optional[int] = None,
) -> dict:
    response = requests.post(
        f"{URL}/initialise",
        json={
            "type": "init",
            "sink_configuration": {
                "type": "udp_sink",
                "host": sink_host,
                "port": sink_port,
            },
            "video_reference": video_reference,
            "model_id": model_id,
            "api_key": api_key,
            "max_fps": max_fps,

        },
    )
    return response.json()
```

### Important notes

- Please remember that `initialise_pipeline()` must be filled with `video_reference` and `sink_configuration`
  in such a way, that any resource (video file / camera device) or URI (stream reference, sink reference) **must be
  reachable from Stream Manager environment!** For instance - in some cases inside docker containers `localhost` will
  be bound into **container localhost** not the localhost of the machine hosting container.

## Developer notes

The pivotal element of the implementation is the Stream Manager component, operating as an application in
single-threaded, TCP-server mode. It systematically processes requests received from a TCP socket,
taking on the responsibility of spawning and overseeing processes that run the `InferencePipelineManager`.
Communication between the `InferencePipelineManager` processes and the main process of the Stream Manager occurs
through multiprocessing queues. These queues facilitate the exchange of input commands and the retrieval of results.

Requests directed to the Stream Manager are sequentially handled in blocking mode,
ensuring that each request must conclude before the initiation of the next one.

### Communication protocol - requests

Stream Manager accepts the following binary protocol in communication. Each communication payload contains:

```
[HEADER: 4B, big-endian, not signed - int value with message size][MESSAGE: utf-8 serialised json of size dictated by header]
```

Message must be a valid JSON after decoding and represent valid command.

#### `list_pipelines` command

```json
{
  "type": "list_pipelines"
}
```

#### `init` command

```json
{
  "type": "init",
  "model_id": "some/1",
  "video_reference": "rtsp://192.168.0.1:554",
  "sink_configuration": {
    "type": "udp_sink",
    "host": "192.168.0.3",
    "port": 9999
  },
  "api_key": "YOUR-API-KEY",
  "max_fps": 16,
  "model_configuration": {
    "type": "object-detection",
    "class_agnostic_nms": true,
    "confidence": 0.5,
    "iou_threshold": 0.4,
    "max_candidates": 300,
    "max_detections": 3000
  },
  "video_source_properties": {
    "frame_width": 1920,
    "frame_height": 1080,
    "fps": 30
  }
}
```
{% include 'model_id.md' %}

#### `terminate` command

```json
{
  "type": "terminate",
  "pipeline_id": "my_pipeline"
}
```

#### `pause` command

```json
{
  "type": "mute",
  "pipeline_id": "my_pipeline"
}
```

#### `resume` command

```json
{
  "type": "resume",
  "pipeline_id": "my_pipeline"
}
```

#### `status` command

```json
{
  "type": "status",
  "pipeline_id": "my_pipeline"
}
```

### Communication protocol - responses

Stream Manager, for each request that can be processed (without timeout or source disconnection), will return the
result in a format:

```
[HEADER: 4B, big-endian, not signed - int value with result size][RESULT: utf-8 serialised json of size dictated by header]
```

Structure of result:

- `request_id` - field with random string representing request id assigned by Stream Manager - to ease debugging
- `pipeline_id` - if command from request can be associated to specific pipeline - its ID will be denoted in response
- `response` - payload of operation response

Each `response` has the `status` key with two values possible: `success` or `failure` to denote operation status.
Each failed response contain `error_type` key to dispatch error handling and optional fields `error_class` and
`error_message` representing inner details of error.

Content of successful responses depends on type of operation.

## Future work

- securing API connection layer (to enable safe remote control)
- securing TCP socket of Stream Manager
