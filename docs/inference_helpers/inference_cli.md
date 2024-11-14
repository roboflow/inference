<div align="center">
    <img
    width="100%"
    src="https://github.com/roboflow/inference/assets/6319317/9230d986-183d-4ab0-922b-4b497f16d937"
    />
</div>

## Roboflow Inference CLI

Roboflow Inference CLI offers a lightweight interface for running the Roboflow inference server locally or the Roboflow Hosted API.

To create custom Inference server Docker images, go to the parent package, <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a>.

<a href="https://roboflow.com" target="_blank">Roboflow</a> has everything you need to deploy a computer vision model to a range of devices and environments. Inference supports object detection, classification, and instance segmentation models, and running foundation models (CLIP and SAM).

### Installation

```bash
pip install roboflow-cli
```

## Examples

### inference server start

Starts a local Inference server. It optionally takes a port number (default is 9001) and will only start the docker container if there is not already a container running on that port.

If you would rather run your server on a virtual machine in Google cloud or Amazon cloud, skip to the section titled "Deploy Inference on Cloud" below.

Before you begin, ensure that you have Docker installed on your machine. Docker provides a containerized environment,
allowing the Roboflow Inference Server to run in a consistent and isolated manner, regardless of the host system. If
you haven't installed Docker yet, you can get it from <a href="https://www.docker.com/get-started" target="_blank">Docker's official website</a>.

The CLI will automatically detect the device you are running on and pull the appropriate Docker image.

```bash
inference server start --port 9001 [-e {optional_path_to_file_with_env_variables}]
```

Parameter `--env-file` (or `-e`) is the optional path for .env file that will be loaded into your Inference server
in case that values of internal parameters needs to be adjusted. Any value passed explicitly as command parameter
is considered as more important and will shadow the value defined in `.env` file under the same target variable name.

#### Development Mode

Use the `--dev` flag to start the Inference Server in development mode. Development mode enables the Inference Server's built in notebook environment for easy testing and development.

#### Tunnel

Use the `--tunnel` flag to start the Inference Server with a tunnel to expose inference to external requests on a TLS-enabled endpoint.

The random generated address will be on server start output:

```
Tunnel to local inference running on https://somethingrandom-ip-192-168-0-1.roboflow.run
```

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

## Deploy Inference on a Cloud VM

You can deploy Roboflow Inference containers to virtual machines in the cloud. These VMs are configured to run CPU or GPU-based Inference servers under the hood, so you don't have to deal with OS/GPU drivers/docker installations, etc! The Inference cli currently supports deploying the Roboflow Inference container images into a virtual machine running on Google (GCP) or Amazon cloud (AWS).

The Roboflow Inference CLI assumes the corresponding cloud CLI is configured for the project you want to deploy the virtual machine into. Read instructions for setting up [Google/GCP - gcloud cli](https://cloud.google.com/sdk/docs/install) or the [Amazon/AWS aws cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

Roboflow Inference cloud deploy is powered by the popular [Skypilot project](https://github.com/skypilot-org/skypilot).

### Important: Cloud Deploy Installation

Before using cloud deploy, optional dependencies must be installed:

```bash
# Install dependencies required for cloud deploy
pip install inference[cloud-deploy]
```


### Cloud Deploy Examples

We illustrate Inference cloud deploy with some examples, below.

*Deploy GPU or CPU inference to AWS or GCP*

```bash
# Deploy the roboflow Inference GPU container into a GPU-enabled VM in AWS

inference cloud deploy --provider aws --compute-type gpu
```

```bash
# Deploy the roboflow Inference CPU container into a CPU-only VM in GCP

inference cloud deploy --provider gcp --compute-type cpu

```

Note the "cluster name" printed after the deployment completes. This handle is used in many subsequent commands.
The deploy command also prints helpful debug and cost information about your VM.

Deploying Inference into a cloud VM will also print out an endpoint of the form "http://1.2.3.4:9001"; you can now run inferences against this endpoint.

Note that the port 9001 is automatically opened - check with your security admin if this is acceptable for your cloud/project.

### View status of deployments

```bash
inference cloud status
```

### Stop and start deployments

```bash
# Stop the VM, you only pay for disk storage while the VM is stopped
inference cloud stop <deployment_handle>

```

### Restart deployments

```bash
inference cloud start <deployment_handle>
```

### Undeploy (delete) the cloud deployment

```bash
inference cloud undeploy <deployment_handle>
```

### SSH into the cloud deployment

You can SSH into your cloud deployment with the following command:
```bash
ssh <deployment_handle>
```

The required SSH key is automatically added to your .ssh/config, you don't need to configure this manually.

### Cloud Deploy Customization

Roboflow Inference cloud deploy will create VMs based on internally tested templates.

For advanced usecases and to customize the template, you can use your [sky yaml](https://skypilot.readthedocs.io/en/latest/reference/yaml-spec.html) template on the command-line, like so:

```bash
inference cloud deploy --custom /path/to/sky-template.yaml
```

If you want you can download the standard template stored in the roboflow cli and the modify it for your needs, this command will do that.

```bash
# This command will print out the standard gcp/cpu sky template.
inference cloud deploy --dry-run --provider gcp --compute-type cpu
```

Then you can deploy a custom template based off your changes.

As an aside, you can also use the [sky cli](https://skypilot.readthedocs.io/en/latest/reference/cli.html) to control your deployment(s) and access some more advanced functionality.

Roboflow Inference deploy currently supports AWS and GCP, please open an issue on the [Inference GitHub repository](https://github.com/roboflow/inference/issues) if you would like to see other cloud providers supported.


### inference infer

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
- `bounding_boxes` - with `BoxAnnotator` and `LabelAnnotator` annotators
- `bounding_boxes_tracing` - with `ByteTracker` and annotators (`BoxAnnotator`, `LabelAnnotator`)
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
  track_activation_threshold: 0.25
  lost_track_buffer: 30
  minimum_matching_threshold: 0.8
  frame_rate: 30
```
`annotators` field is a list of dictionaries with two keys: `type` and `param`. `type` points to 
name of annotator class:
```python
from supervision import *
ANNOTATOR_TYPE2CLASS = {
    "bounding_box": BoxAnnotator,
    "box": BoxAnnotator,
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

### inference benchmark

!!! note

    The command is introduced in `inference_cli>=0.9.10`

`inference benchmark` is a set of command suited to run benchmarks of `inference`. There are two types of benchmark 
available `inference benchmark api-speed` - to test `inference` HTTP server and `inference benchmark python-package-speed`
to verify the performance of `inference` Python package.

!!! tip
    
    Use `inference benchmark api-speed --help` / `inference benchmark python-package-speed --help` to
    display all options of benchmark commands.

!!! tip
    
    Roboflow API key can be provided via `ROBOFLOW_API_KEY` environment variable

#### Running benchmark of Python package 

Basic benchmark can be run using the following command: 

```bash
inference benchmark python-package-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```
Command runs specified number of inferences using pointed model and saves statistics (including benchmark 
parameter, throughput, latency, errors and platform details) in pointed directory.

#### Running benchmark of `inference server`

!!! note

    Before running API benchmark - make sure the server is up and running:
    ```bash
    inference server start
    ```
Basic benchmark can be run using the following command: 

```bash
inference benchmark api-speed \
  -m {your_model_id} \
  -d {pre-configured dataset name or path to directory with images} \
  -o {output_directory}  
```
Command runs specified number of inferences using pointed model and saves statistics (including benchmark 
parameter, throughput, latency, errors and platform details) in pointed directory.

This benchmark has more configuration options to support different ways HTTP API profiling. In default mode,
single client will be spawned, and it will send one request after another sequentially. This may be suboptimal
in specific cases, so one may specify number of concurrent clients using `-c {number_of_clients}` option.
Each client will send next request once previous is handled. This option will also not cover all scenarios
of tests. For instance one may want to send `x` requests each second (which is closer to the scenario of
production environment where multiple clients are sending requests concurrently). In this scenario, `--rps {value}` 
option can be used (and `-c` will be ignored). Value provided in `--rps` option specifies how many requests 
are to be spawned **each second** without waiting for previous requests to be handled. In I/O intensive benchmark 
scenarios - we suggest running command from multiple separate processes and possibly multiple hosts.

## Supported Devices

Roboflow Inference CLI currently supports the following device targets:

- x86 CPU
- ARM64 CPU
- NVIDIA GPU

For Jetson specific inference server images, check out the <a href="https://pypi.org/project/inference/" target="_blank">Roboflow Inference</a> package, or pull the images directly following instructions in the official [Roboflow Inference documentation](/quickstart/docker/#pull-from-docker-hub).
