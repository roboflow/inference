# Inference Features

Inference aspires to be a one-stop shop for all of your computer vision needs
(and where the feature you seek is not in stock, it's straightforward to
extend the functionality to suit your needs). A non-exhaustive list of
features is captured here.

## Model Serving

The core of Inference is centered around serving computer vision models. It
implements architectures for tasks like Object Detection, Image Classification,
Instance Segmentation, Keypoint Detection, Image Embedding, OCR,
Visual Question Answering, Gaze Detection, and more.

## Image Processing

A model is only as good as the image it receives. As the old adage says,
"Garbage In, Garbage Out", which is why Inference includes and applies the
same image pre- and post-processing methods that models use during training.
It also applies these steps efficiently to avoid unnecessary latency.

## Video Stream Management

To do video inference properly requires attention to the image ingestion
pipeline. Inference spawns separate threads to process video streams so that
your model is never left hanging and always gets the most recent frame possible.

## Workflows

If models are the brains and cameras are the eyes, Workflows are the nervous
system and Workflow Blocks are the body's other organs and tools. By declaring
a computation graph, Workflows efficiently pipe and parallelize data through
models, logic, integrations, and custom code. There are already over 100
different Workflow Blocks included with new ones contributed regularly.

## Server

Inference exposes an HTTP server for interfacing with its functionality. This
lets you use it as a micro-service in a larger system and programmatically
command its operations.

## SDK

The included Python Client makes it easy to interact with the server from
your applications.

## CLI

The Command Line Interface provides convenience methods for starting the server,
running benchmarks, cloud deployment, and testing.

## Speed

Built-in optimizations like automatic parallelization via multiprocessing,
hardware acceleration, and dynamic batching help you get the most out of your
hardware. With the optional TensorRT flag you can also take advantage of
quantization and device-specific layer fusion optimizations on supported GPUs.
This lets you run more streams and process with higher resolution and
lower latency.

## Offline Cache

By pulling down models and Workflow definitions and storing them locally,
the Inference Server can operate in offline mode.

## Insights

Inference can connect with Roboflow's platform to help you monitor and improve
your models by uploading outlier data, accessing stats and telemetry, and
tying into downstream data sinks and business intelligence suites.

## Roboflow Integration

With over 100k fine-tuned models available, Roboflow Universe is the largest
repository of computer vision models in the world. It also creates the
most popular platform for training custom computer vision models. Inference's
optional integration with the Roboflow platform supercharges CV applications
by providing access to the best and most relevant problems for solving
computer vision problems.

## Portability

We support running Inference on a myriad of devices from MacOS development
machines to beefy cloud servers to tiny edge devices and AI appliances. Simply
swap out the Docker tag and the same code you're running locally can be
deployed to another platform.

## Extensibility

As an open source project, Inference can be forked and extended to add new
models, Workflow Blocks, backends, and more. We also support using
Dynamic Python Blocks to seamlessly bridge the gaps between blocks.

## Cutting Edge Updates

With over 60 releases last year, Inference is constantly being updated.
It often adopts new state of the art models within days of their release
and new functionality is added weekly.

## Stability and Scalability

Inference has a suite of over 350 tests that run on every Pull Request across
each supported target. It's used in production by many of the world's largest
companies and is backed by the creator of the industry standard computer
vision platform. It's powered billions of inferences across hundreds of
thousands of models. It also powers Roboflow's core model serving
infrastructure and products.

## Security

We undergo an annual pen test and SOC 2 Type II audit and have an infrastructure
and security team working to mitigate issues and respond to vulnerabilities and
issues flagged by automated scans and external parties.

## Support

Roboflow has a fully staffed customer support and professional services
organization available for enterprise customers.