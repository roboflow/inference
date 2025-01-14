<div align="center">
  <p>
    <a align="center" href="" target="https://inference.roboflow.com/">
      <img
        width="100%"
        src="https://github.com/roboflow/inference/blob/main/banner.png?raw=true"
      >
    </a>
  </p>

  <br>

[notebooks](https://github.com/roboflow/notebooks) | [supervision](https://github.com/roboflow/supervision) | [autodistill](https://github.com/autodistill/autodistill) | [maestro](https://github.com/roboflow/multimodal-maestro)

  <br>

[![version](https://badge.fury.io/py/inference.svg)](https://badge.fury.io/py/inference)
[![downloads](https://img.shields.io/pypi/dm/inference)](https://pypistats.org/packages/inference)
[![docker pulls](https://img.shields.io/docker/pulls/roboflow/roboflow-inference-server-cpu)](https://hub.docker.com/u/roboflow)
[![license](https://img.shields.io/pypi/l/inference)](https://github.com/roboflow/inference/blob/main/LICENSE.core)
[![discord](https://img.shields.io/discord/1159501506232451173?logo=discord&label=discord&labelColor=fff&color=5865f2&link=https%3A%2F%2Fdiscord.gg%2FGbfgXGJ8Bk)](https://discord.gg/GbfgXGJ8Bk)

<!-- [![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/workflows) -->

</div>

## Make Any Camera an AI Camera

Inference turns any computer or edge device into a command center for your computer vision projects.

* 🛠️ Self-host [your own fine-tuned models](https://inference.roboflow.com/quickstart/explore_models/)
* 🧠 Access the latest and greatest foundation models (like [Florence-2](https://blog.roboflow.com/florence-2/), [CLIP](https://blog.roboflow.com/openai-clip/), and [SAM2](https://blog.roboflow.com/what-is-segment-anything-2/))
* 🤝 Use Workflows to track, count, time, measure, and visualize
* 👁️ Combine ML with traditional CV methods (like OCR, Barcode Reading, QR, and template matching)
* 📈 Monitor, record, and analyze predictions
* 🎥 Manage cameras and video streams
* 📬 Send notifications when events happen
* 🛜 Connect with external systems and APIs
* 🔗 [Extend](https://inference.roboflow.com/workflows/create_workflow_block/) with your own code and models
* 🚀 Deploy production systems at scale

See [Example Workflows](https://inference.roboflow.com/workflows/gallery/) for common use-cases like detecting small objects with SAHI, multi-model consensus, active learning, reading license plates, blurring faces, background removal, and more.

[Time In Zone Workflow Example](https://github.com/user-attachments/assets/743233d9-3460-442d-83f8-20e29e76b346)

## 🔥 quickstart

[Install Docker](https://docs.docker.com/engine/install/) (and
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
for GPU acceleration if you have a CUDA-enabled GPU). Then run

```
pip install inference-cli && inference server start --dev
```

This will pull the proper image for your machine and start it in development mode.

In development mode, a Jupyter notebook server with a quickstart guide runs on 
[`localhost:9002`](http://localhost:9002). Dive in there for a whirlwind tour
of your new Inference Server's functionality!

Now you're ready to connect your camera streams and
[start building & deploying Workflows in the UI](https://app.roboflow.com/workflows)
or [interacting with your new server](https://inference.roboflow.com/workflows/create_and_run/)
via its API.

## 🛠️ build with Workflows

A key component of Inference is [Workflows](https://roboflow.com/workflows), composable blocks of common functionality that give models a common interface to make chaining and experimentation easy.

![License Plate OCR Workflow Visualization](https://github.com/user-attachments/assets/178046a2-011e-489d-bfc2-41dcfefe44a4)

With Workflows, you can:
* Detect, classify, and segment objects in images using state-of-the-art models.
* Use Large Multimodal Models (LMMs) to make determinations at any stage in a workflow.
* Seamlessly swap out models for a given task.
* Chain models together.
* Track, count, time, measure, and visualize objects.
* Add business logic and extend functionality to work with your external systems.

Workflows allow you to extend simple model predictions to build computer vision micro-services that fit into a larger application or fully self-contained visual agents that run on a video stream.

[Learn more](https://roboflow.com/workflows), read [the Workflows docs](https://inference.roboflow.com/workflows/about/), or [start building](https://app.roboflow.com/workflows).
<br /><br />
<p align="left">
<a href="https://youtu.be/VCbcC5OEGRU" title="Live Video Analytics | Build a Traffic Monitoring Application with Workflows"><img src="https://img.youtube.com/vi/VCbcC5OEGRU/0.jpg" alt="Workflows Tutorial" width="300px" align="left" /></a>
<a href="https://youtu.be/VCbcC5OEGRU" title="Live Video Analytics | Build a Traffic Monitoring Application with Workflows"><strong>Tutorial: Build a Traffic Monitoring Application with Workflows</strong></a>
<div><strong>Created: 22 Oct 2024</strong></div>
<br/>Learn how to build and deploy Workflows for common use-cases like detecting vehicles, filtering detections, visualizing results, and calculating dwell time on a live video stream.</p>
<br /><br />

## 📟 connecting via api
  
Once you've installed Inference, your machine is a fully-featured CV center.
You can use its API to run models and workflows on images and video streams.
By default, the server is running locally on
[`localhost:9001`](http://localhost:9001).

To interface with your server via Python, use our SDK.
`pip install inference-sdk` then run
[an example model comparison Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSHhIODdZR0FGUWhaVmtOVWNEeVUiLCJ3b3Jrc3BhY2VJZCI6IlhySm9BRVFCQkFPc2ozMmpYZ0lPIiwidXNlcklkIjoiNXcyMFZ6UU9iVFhqSmhUanE2a2FkOXVicm0zMyIsImlhdCI6MTczNTIzNDA4Mn0.AA78pZnlivFs5pBPVX9cMigFAOIIMZk0dA4gxEF5tj4)
like this:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

result = client.run_workflow(
    workspace_name="roboflow-docs",
    workflow_id="model-comparison",
    images={
        "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
    },
    parameters={
        "model1": "yolov8n-640",
        "model2": "yolov11n-640"
    }
)

print(result)
```

In other languages, use the server's REST API;
you can access the API docs for your server at
[`/docs` (OpenAPI format)](http://localhost:9001/docs) or
[`/redoc` (Redoc Format)](http://localhost:9001/redoc).

Check out [the inference_sdk docs](https://inference.roboflow.com/inference_helpers/inference_sdk/)
to see what else you can do with your new server.

## 🎥 connect to video streams

The inference server is a video processing beast. You can set it up to run
Workflows on RTSP streams, webcam devices, and more. It will handle hardware
acceleration, multiprocessing, video decoding and GPU batching to get the
most out of your hardware.

[This example workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiNHMzSDAzcmtyU0JiSDhFMjEzZUUiLCJ3b3Jrc3BhY2VJZCI6IlhySm9BRVFCQkFPc2ozMmpYZ0lPIiwidXNlcklkIjoiNXcyMFZ6UU9iVFhqSmhUanE2a2FkOXVicm0zMyIsImlhdCI6MTczNTIzOTk3NX0.TYdmD5AS8tbpz8AxEr5xW-05LlegK61kq-5_OReIrwc?showGraph=true&hideToolbar=false)
will watch a stream for frames that
[CLIP thinks](https://blog.roboflow.com/openai-clip/) match an
inputted text prompt.
```python
from inference_sdk import InferenceHTTPClient
import atexit
import time

max_fps = 4

client = InferenceHTTPClient(
    api_url="http://localhost:9001", # use local inference server
    # api_key="<YOUR API KEY>" # optional to access your private data and models
)

# Start a stream on an rtsp stream
result = client.start_inference_pipeline_with_workflow(
    video_reference=["rtsp://user:password@192.168.0.100:554/"],
    workspace_name="roboflow-docs",
    workflow_id="clip-frames",
    max_fps=max_fps,
    workflows_parameters={
        "prompt": "blurry", # change to look for something else
        "threshold": 0.16
    }
)

pipeline_id = result["context"]["pipeline_id"]

# Terminate the pipeline when the script exits
atexit.register(lambda: client.terminate_inference_pipeline(pipeline_id))

while True:
  result = client.consume_inference_pipeline_result(pipeline_id=pipeline_id)

  if not result["outputs"] or not result["outputs"][0]:
    # still initializing
    continue

  output = result["outputs"][0]
  is_match = output.get("is_match")
  similarity = round(output.get("similarity")*100, 1)
  print(f"Matches prompt? {is_match} (similarity: {similarity}%)")

  time.sleep(1/max_fps)
```

Pipeline outputs can be consumed via API for downstream processing or the
Workflow can be configured to call external services with Notification blocks
(like [Email](https://inference.roboflow.com/workflows/blocks/email_notification/)
or [Twilio](https://inference.roboflow.com/workflows/blocks/twilio_sms_notification/))
or the [Webhook block](https://inference.roboflow.com/workflows/blocks/webhook_sink/).
For more info on video pipeline management, see the
[Video Processing overview](https://inference.roboflow.com/workflows/video_processing/overview/).

If you have a Roboflow account & have linked an API key, you can also remotely
[monitor and manage your running streams](https://app.roboflow.com/devices)
via the Roboflow UI.

## 🔑 connect to the cloud

Without an API Key, you can access a wide range of pre-trained and foundational models and run public Workflows.

Pass an optional [Roboflow API Key](https://app.roboflow.com/settings/api) to the `inference_sdk` or API to access additional features enhanced by Roboflow's Cloud
platform. When running with an API Key, usage is metered according to
Roboflow's [pricing tiers](https://roboflow.com/pricing).

|                         | Open Access | With API Key |
|-------------------------|-------------|--------------|
| [Pre-Trained Models](https://inference.roboflow.com/quickstart/aliases/#supported-pre-trained-models) | ✅ | ✅
| [Foundation Models](https://inference.roboflow.com/foundation/about/) | ✅ | ✅
| [Video Stream Management](https://inference.roboflow.com/workflows/video_processing/overview/) | ✅ | ✅
| [Dynamic Python Blocks](https://inference.roboflow.com/workflows/custom_python_code_blocks/) | ✅ | ✅
| [Public Workflows](https://inference.roboflow.com/workflows/about/) | ✅ | ✅
| [Private Workflows](https://docs.roboflow.com/workflows/create-a-workflow) |  | ✅
| [Fine-Tuned Models](https://roboflow.com/train) |  | ✅
| [Universe Models](https://roboflow.com/universe) |  | ✅
| [Active Learning](https://inference.roboflow.com/workflows/blocks/roboflow_dataset_upload/) |  | ✅
| [Serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) |  | ✅
| [Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) |  | ✅
| [Commercial Model Licensing](https://roboflow.com/licensing) |  | Paid
| [Device Management](https://docs.roboflow.com/roboflow-enterprise) |  | Enterprise
| [Model Monitoring](https://docs.roboflow.com/deploy/model-monitoring) |  | Enterprise

## 🌩️ hosted compute

If you don't want to manage your own infrastructure for self-hosting, Roboflow offers a hosted Inference Server via [one-click Dedicated Deployments](https://docs.roboflow.com/deploy/dedicated-deployments) (CPU and GPU machines) billed hourly, or simple models and Workflows via our [serverless Hosted API](https://docs.roboflow.com/deploy/hosted-api) billed per API-call.

We offer a [generous free-tier](https://roboflow.com/pricing) to get started.

## 🖥️ run on-prem or self-hosted

Inference is designed to run on a wide range of hardware from beefy cloud servers to tiny edge devices. This lets you easily develop against your local machine or our cloud infrastructure and then seamlessly switch to another device for production deployment.

`inference server start` attempts to automatically choose the optimal container to optimize performance on your machine, special installation notes and performance tips by device are listed below.

<details>
<summary><b>CPU</b></summary>
<br />
The core docker image includes support for OpenVINO acceleration on x64 CPUs via onnxruntime. Heavy models like SAM2 and CogVLM may run too slowly (dozens of seconds per image) to be practical. The primary use-cases for CPU inference are processing still images (eg for NSFW classification of uploads or document verification) or infrequent sampling of frames on a video (eg for occupancy tracking of a parking lot).
<br /><br />
You may also want to consider using our <a href="https://docs.roboflow.com/deploy/hosted-api">serverless Hosted API</a> for light or spiky load.
<br /><br />
To start the container manually, run

```
sudo docker run -p 9001:9001 -v ~/.inference/cache:/tmp/cache roboflow/roboflow-inference-server-cpu:latest
```

To install the python package natively, install via PyPi

```
pip install inference
```
<br />
</details>
<details>
<summary><b>Mac / Apple Silicon (MPS)</b></summary>
<br />
Apple does not yet support <a href="https://github.com/pytorch/pytorch/issues/81224">passing the Metal Performance Shader device to Docker</a> so hardware acceleration is not possible inside the container.
<br /><br />
We recommend starting with the CPU Docker via <code>inference server start</code> but, if you need more speed, the <code>inference</code> Python package supports hardware acceleration via the <a href="https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html">onnxruntime CoreMLExecutionProvider</a> and the <a href="https://pytorch.org/docs/stable/notes/mps.html">PyTorch <code>mps</code> device backend</a>. By using these, inference gets a big boost when running outside of Docker on Apple Silicon.
<br /><br />
To install outside of Docker, clone the repo then install the dependencies in a new virtual environment:

```
git clone https://github.com/roboflow/inference.git
cd inference
python3 -m venv inf
source inf/bin/activate
pip install .
cp docker/config/cpu_http.py .
```

Then start the server by running `uvicorn` with the `cpu_http` module in your virtual environment:
```
# source inf/bin/activate
uvicorn cpu_http:app --port 9001 --host 0.0.0.0
```

Your server is now running at [`localhost:9001`](http://localhost:9001) with MPS acceleration.

To run natively in python, `pip install inference` will automatically pull in the CoreMLExecutionProvider on Mac.
<br />
</details>
<details>
<summary><b>NVIDIA GPU (Linux)</b></summary>
<br />
By default, <code>inference server start</code> should run the right container automatically.
<br /><br />
To start the server manually, use the <code>roboflow/roboflow-inference-server-gpu:latest</code> docker container with NVIDIA Container Runtime:

```
sudo docker run --gpus all --net=host -v ~/.inference/cache:/tmp/cache roboflow/roboflow-inference-server-gpu:latest
```

Or `pip install inference-gpu` to run the python package natively.

You can enable TensorRT by adding `TensorrtExecutionProvider` to the `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.
<br /><br />
<b>⚠️ Note:</b> TensorRT is not enabled by default due to long (15+ minute) compilation times each time a new model is initialized. We cache the TensorRT engine in `/tmp/cache`, which is a Docker volume mounted from `~/.inference/cache` by default.

```
export ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider]"
```
<br />
</details>
<details>
<summary><b>NVIDIA GPU (Windows/WSL)</b></summary>
<br />
To get GPU acceleration on Windows, you need WSL2 with NVIDIA Container Toolkit. <a href="https://docs.nvidia.com/cuda/wsl-user-guide/index.html">Follow the guide here</a> then use the instructions for <code>NVIDIA GPU (Linux)</code> above.
<br />
</details>
<details>
<summary><b>NVIDIA Jetson / JetPack</b></summary>
<br />
We have specialized containers built with support for hardware acceleration on JetPack 4, 5, and 6. <code>inference server start</code> will automatically detect your JetPack version and use the right container.
<br /><br />
To start the server manually, use the container for your JetPack version with the nvidia runtime. For example, on JetPack 6:

```
sudo docker run --runtime=nvidia --net=host -v ~/.inference/cache:/tmp/cache roboflow/roboflow-inference-server-jetson-6.0.0:latest
```

You can enable TensorRT by adding `TensorrtExecutionProvider` to the `ONNXRUNTIME_EXECUTION_PROVIDERS` environment variable.
<br /><br />
<b>⚠️ Note:</b> TensorRT is not enabled by default due to long (15+ minute) compilation times each time a new model is initialized. We cache the TensorRT engine in `/tmp/cache`, which is a Docker volume mounted from `~/.inference/cache` by default.

```
sudo docker run \
  --runtime=nvidia \
  --net=host \
  -e ONNXRUNTIME_EXECUTION_PROVIDERS="[TensorrtExecutionProvider,CUDAExecutionProvider,OpenVINOExecutionProvider,CoreMLExecutionProvider,CPUExecutionProvider]" \
  -v ~/.inference/cache:/tmp/cache \
  roboflow/roboflow-inference-server-jetson-6.0.0:latest
```
<br />
</details>
<details>
<summary><b>Raspberry Pi</b></summary>
<br />
The CPU container works on Raspberry Pi 4 Model B and Raspberry Pi 5 so long as you are using <a href="https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit">the 64-bit version of the operating system</a>. Simply run <code>inference server start</code> and you'll be all set.
<br /><br />
Expect about 1fps on Pi 4 and 4fps on Pi 5 for a "Roboflow 3.0 Fast" object detection model (equivalent to a "nano" sized YOLO model).
<br />
</details>
<details>
<summary><b>Other GPUs</b></summary>
<br />
We do not currently support hardware acceleration on other GPUs besides those listed here but ONNX Runtime has <a href="https://onnxruntime.ai/docs/execution-providers/">additional execution providers</a> for AMD/ROCm, Arm NN, Rockchip, and others. If you install one of these runtimes, you can enable it via the <code>ONNXRUNTIME_EXECUTION_PROVIDERS</code> environment variable.
<br /><br />
For example:

```
export ONNXRUNTIME_EXECUTION_PROVIDERS="[ROCMExecutionProvider,OpenVINOExecutionProvider,CPUExecutionProvider]"
```

This is untested and performance improvements are not guaranteed.
<br />
</details>
<details>
<summary><b>Other Edge Devices</b></summary>
<br />
Roboflow has <a href="https://docs.roboflow.com/deploy/supported-deployment-devices">SDKs for running object detection natively</a> on other deployment targets like <a href="https://docs.roboflow.com/deploy/sdks/web-browser">Tensorflow.js in a web browser</a>, <a href="https://docs.roboflow.com/deploy/sdks/mobile-ios-on-device">Native Swift on iOS</a> via CoreML, and <a href="https://docs.roboflow.com/deploy/sdks/lens-studio">Snap Lens Studio</a>.
<br /><br />
Connect to an Inference Server via its API for additional functionality beyond object detection (like running Workflows).
<br />
</details>

### ⭐️ New: Enterprise Hardware

For manufacturing and logistics use-cases Roboflow now offers [the Flowbox](https://roboflow.com/industries/manufacturing/box), a ruggedized CV center pre-configured with Inference and optimized for running in secure networks. It has integrated support for machine vision cameras like Basler and Lucid over GigE, supports interfacing with PLCs and HMIs via OPC or MQTT, enables enterprise device management through a DMZ, and comes with the support of our team of computer vision experts to ensure your project is a success.

## 📚 documentation

Visit our [documentation](https://inference.roboflow.com) to explore comprehensive guides, detailed API references, and a wide array of tutorials designed to help you harness the full potential of the Inference package.

## © license

The core of Inference is licensed under Apache 2.0.

Models are subject to licensing which respects the underlying architecture. These licenses are listed in [`inference/models`](/inference/models). Paid Roboflow accounts include a commercial license for some models (see [roboflow.com/licensing](https://roboflow.com/licensing) for details).

Cloud connected functionality (like our model and Workflows registries, dataset management, model monitoring, device management, and managed infrastructure) requires a Roboflow account and API key & is metered based on usage.

Enterprise functionality is source-available in [`inference/enterprise`](/inference/enterprise/) under an [enterprise license](/inference/enterprise/LICENSE.txt) and usage in production requires an active Enterprise contract in good standing.

See the "Self Hosting and Edge Deployment" section of the [Roboflow Licensing](https://roboflow.com/licensing) documentation for more information on how Roboflow Inference is licensed.

## 🏆 contribution

We would love your input to improve Roboflow Inference! Please see our [contributing guide](https://github.com/roboflow/inference/blob/master/CONTRIBUTING.md) to get started. Thank you to all of our contributors! 🙏


<br>

<div align="center">
  <div align="center">
      <a href="https://youtube.com/roboflow">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/youtube.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634652"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/roboflow-app.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949746649"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://www.linkedin.com/company/roboflow-ai/">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/linkedin.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633691"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://docs.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/knowledge.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949634511"
            width="3%"
          />
      </a>
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://disuss.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/forum.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633584"
            width="3%"
          />
      <img src="https://raw.githubusercontent.com/ultralytics/assets/main/social/logo-transparent.png" width="3%"/>
      <a href="https://blog.roboflow.com">
          <img
            src="https://media.roboflow.com/notebooks/template/icons/purple/blog.png?ik-sdk-version=javascript-1.4.3&updatedAt=1672949633605"
            width="3%"
          />
      </a>
      </a>
  </div>
</div>
