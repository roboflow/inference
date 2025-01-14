# Inference vs Alternatives

With its wide aperature of functionality, Inference's features overlap
with many other pieces of software. This guide aims to help readers
understand when they should (and should not) choose Inference over
other tools.

## Inference Servers

### NVIDIA Triton Inference Server

Triton is a powerhouse tool for machine learning experts to deploy ML
models at scale. Its primary focus is on extremely optimized pipelines
that run efficiently on NVIDIA hardware. It can be tough to use, trading
off simplicity and a quick development cycle for raw speed and is
geared towards expert users. It can chain models together, but doing
so is a rigid and manual process.

By contrast, Inference tries to be as fast as possible while remaining
developer friendly. It invests heavily in tooling to make it quick and
easy to iterate on your project, and provides APIs for remotely updating
your configuration. Workflows is more flexible and feature rich than
Triton's model ensembles. Additionally, Inference leans heavily into computer
vision specific features like visualization and stateful video
functionality and integrations like notifications and data sinks.

**Choose Triton if:** you're a machine learning expert that values speed on
NVIDIA GPUs above all else.

### Lightning LitServe

LitServe is a lightweight and customizable inference server focused on serving
models with minimal overhead. It is fairly minimalistic but flexible and
self-contained.

Like Triton, LitServe is task-agnostic, meaning it is designed to balance the
needs of vision models with NLP, audio, and tabular models. This means it's not
as feature-rich for computer vision applications (for example, it doesn't have
any built-in features for streaming video). It is also highly focused on model
serving without an abstraction layer like Workflows for model chaining and
integrations with other tools.

**Choose LitServe if:** you are working on general-purpose machine learning
tasks and want a lightweight, unopinionated, code-centric solution that's faster
and comes with more batteries included than rolling your own server with
FastAPI or Flask.

### Tensorflow Serving

If you're deeply engrained in the Tensorflow ecosystem and want to deploy a
variety of Tensorflow models in different modalities like NLP, recommender
systems, and audio in addition to CV models, Tensorflow Serving may be a
good choice.

It can be complex to setup and maintain and lacks features many users would
consider table stakes (like pre- and post-processing which in many cases will
need to be custom coded). Like several of the other servers listed here, it
lacks depth in vision-specific functionality.

**Choose Tensorflow Serving if:** the Tensorflow ecosystem is very important
to you and you're willing to put in the legwork to take advantage of its
advanced feature set.

### TorchServe

The PyTorch ecosystem's equivalent of Tensorflow Serving is TorchServe. It's
optimized for serving PyTorch models across several domains including vision,
NLP, tabular data, and audio.

Like Tensorflow Serving, it is designed for large-scale cloud deployments and
can require custom configuration for things like pre- and post-processing and
deploying multiple models. Because of its wide mandate it lacks many
vision-specific features (like video streaming).

**Chose TorchServe if:** you're looking for a way to scale and customize the
deployment of your PyTorch models and don't need vision-specific functionality.

### FastAPI or Flask

In the olden days, most people rolled their own servers to expose their ML
models to client applications. In fact, Roboflow Inference's HTTP interface
and REST API are built on FastAPI.

In this day and age, it's certainly still possible to start from scratch,
but you'll be reinventing the wheel and will run into a lot of footguns
others have already solved along the way. It's usually better and faster
to use one of the existing ML-focused servers.

**Choose FastAPI or Flask if:** your main goal is learning the intricacies of
making an inference server.

## Workflow Builders

### ComfyUI

ComfyUI is the closest comparable to Roboflow Workflows. It's a node-based
pipeline builder with its roots in generative image and video models. It
has a UI for chaining models and applying logic.

While there is some overlap in functionality (for example, there is a Stable Diffusion
Workflows Block and a YOLO ComfyUI Node), but the ecosystem of nodes and the
community focus of Comfy is squarely centered around generative models while
Inference is focused on interfacing with the real-world.

**Chose ComfyUI if:** you're using generative image and video models like Flux and
Stable Diffusion and don't need to use custom fine-tuned models in your pipeline
to do things like selectively replacing specific objects.

### Node-RED

Node-RED is a low-code development platform for connecting devices, APIs, and
services. It is widely used in home-automation use-cases because of its friendly
interface for non-technical users. It provides a graphical interface for
designing workflows and supports custom nodes which allows integration with
ML models (primarily via external servers).

But because it's not designed for machine learning tasks it can struggle with
high-performance, compute-heavy tasks and isn't well suited for computer vision
use-cases.

**Choose Node-RED:** (possibly in conjunction with Inference via a custom node)
if its wide selection of integrations with IoT sensors and other tools is a big
unlock for your project.

## Edge Deployment

### Edge Impulse

Edge Impulse is a platform focused on deploying ML models to very low-power edge
devices and embedded systems. It supports both vision and other models like audio,
time-series, and signal processing. Edge Impulse is uniquely good at working with
microcontrollers and has SDKs for single-board computers and mobile devices.

The design focus on TinyML makes it less suited for high-resource, general-purpose
tasks like video processing and running modern, state-of-the-art ML models. It also
requires some familiarity with embedded systems. It does not offer an equivalent to
Workflows to create complex logic and integrate with other systems and typically
requires custom coding your application logic to run on the embedded board.

**Chose Edge Impulse if:** you're working on an IoT or wearable device that's not
capable of running more powerful models, framework, and logic.

### NVIDIA DeepStream

DeepStream is NVIDIA's platform for building highly optimized video processing
pipelines accelerated by NVIDIA's hardware, taking full advantage of TensorRT
for accelerated inference and CUDA for parallel processing. It targets many of
the same business problems as Inference, including monitoring security cameras,
smart cities, and industrial IoT.

DeepStream has a reputation for being difficult to use with a steep learning curve.
It requires familiarity with NVIDIA tooling and while it is highly configurable,
it's also highly complex. It's focused on video processing, without deep integrations
with other tooling. DeepStream is not open source; ensure that the license is
suitable for your project.

**Choose DeepStream if:** you're an expert willing to invest a lot of time and effort
into optimizing a single project and high throughput is your primary objective.
