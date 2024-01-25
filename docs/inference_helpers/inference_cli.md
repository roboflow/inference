<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />
</div>

## Roboflow Inference CLI

Roboflow Inference CLI offers a lightweight interface for running the Roboflow inference server locally or the Roboflow Hosted API.

To create custom inference server Docker images, go to the parent package, <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a>.

<a href="https://roboflow.com" target="_blank">Roboflow</a> has everything you need to deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

### Installation

```bash
pip install roboflow-cli
```

## Examples

### inference server start

Starts a local inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from <a href="https://www.docker.com/get-started" target="_blank">Docker's official website</a>.

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

Runs inference on:

- single image
- directory wih images 
- video file

It takes input path / url and model version to produce predictions (and optionally make visualisation using 
`supervision`). You can also specify a host to run inference on our hosted inference server.

!!! note
    
    If you decided to use hosted inference server - make sure command `inference server start` was used first 

!!! tip
    
    Use `inference infer --help` to display description of parameters

!!! tip
    
    Roboflow API key can be provided via `ROBOFLOW_API_KEY` environment variable

#### Local image

This command is going to make a prediction from local image using selected model and print the prediction on 
the console.

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY}
```

To display visualised prediction use `-D` option. To save prediction and visualisation in a local directory,
use `-o {path_to_your_directory}` option. Those options work also in other modes.

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY} -D -o {path_to_your_output_directory}
```

#### Hosted image

```bash
inference infer -i https://[YOUR_HOSTED_IMAGE_URL] -m {your_project}/{version} --api-key {YOUR_API_KEY}
```

#### Hosted API inference

```bash
inference infer -i ./image.jpg -m {your_project}/{version} --api-key {YOUR_API_KEY} -h https://detect.roboflow.com
```

#### Local directory

```bash
inference infer -i {your_directory_with_images} -m {your_project}/{version} -o {path_to_your_output_directory} --api-key {YOUR_API_KEY}
```

#### Video file

```bash
inference infer -i {path_to_your_video_file} -m {your_project}/{version} -o {path_to_your_output_directory} --api-key {YOUR_API_KEY}
```

#### Configuration of visualisation
Option `-c` can be provided with a path to `*.yml` file configuring `supervision` visualisation.
There are few pre-defined configs:
- `bounding_boxes` - with `BoundingBoxAnnotator` and `LabelAnnotator` annotators
- `bounding_boxes_tracing` - with `ByteTracker` and annotators (`BoundingBoxAnnotator`, `LabelAnnotator`)
- `masks` - with `MaskAnnotator` and `LabelAnnotator` annotators
- `polygons` - with `PolygonAnnotator` and `LabelAnnotator` annotators

Custom configuration can be created following the schema:
```yaml
annotators:
  - type: "bounding_box"
    params:
      thickness: 2
  - type: "label"
    params:
      text_scale: 0.5
      text_thickness: 2
      text_padding: 5
  - type: "trace"
    params:
      trace_length: 60
      thickness: 2
tracking:
  track_thresh: 0.25
  track_buffer: 30
  match_thresh: 0.8
  frame_rate: 30
```
`annotators` field is a list of dictionaries with two keys: `type` and `param`. `type` points to 
name of annotator class:
```python
from supervision import *
ANNOTATOR_TYPE2CLASS = {
    "bounding_box": BoundingBoxAnnotator,
    "mask": MaskAnnotator,
    "polygon": PolygonAnnotator,
    "color": ColorAnnotator,
    "halo": HaloAnnotator,
    "ellipse": EllipseAnnotator,
    "box_corner": BoxCornerAnnotator,
    "circle": CircleAnnotator,
    "dot": DotAnnotator,
    "label": LabelAnnotator,
    "blur": BlurAnnotator,
    "trace": TraceAnnotator,
    "heat_map": HeatMapAnnotator,
    "pixelate": PixelateAnnotator,
    "triangle": TriangleAnnotator,
}
```
`param` is a dictionary of annotator constructor parameters (check them in 
[`supervision`](https://github.com/roboflow/supervision) docs - you would only be able
to use primitive values, classes and enums that are defined in constructors may not be possible
to resolve from yaml config).

`tracking` is an optional key that holds a dictionary with constructor parameters for
`ByteTrack`.

#### Configuration of model
`-mc` parameter can be provided with path to `*.yml` file that specifies 
model configuration (like confidence threshold or IoU threshold). If given,
configuration will be used to initialise `InferenceConfiguration` object
from `inference_sdk` library. See [sdk docs](./inference_sdk.md) to discover
which options can be configured via `*.yml` file - configuration keys must match
with names of fields in `InferenceConfiguration` object.

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson specific inference server images, check out the <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a> package, or pull the images directly following instructions in the official [Roboflow Inference documentation](/quickstart/docker/#pull-from-docker-hub).
