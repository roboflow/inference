# Architecture

Inference is best run in server mode (it also supports a native Python interface;
though see [why we recommend Docker](#why-docker)). You interact with it via a REST API
(most often through [the Python SDK](/inference_helpers/inference_sdk/)) or a web
browser (for example, the [Roboflow Platform's UI](https://app.roboflow.com) optionally
provides a frontend for interfacing with a locally hosted Inference Server).

It orchestrates getting predictions through a model (or series of models) and
processing the results. Inference automatically does multithreading to parallelize
workloads and efficiently utilize the available GPU and CPU cores. It also dynamically
adapts to variable processing rates when consuming video streams to ensure its host
machine is not overloaded.

Inference talks to services in the cloud to retrieve model weights and
Workflow definitions and keeps track of model results for later evaluation, but
all of the computation is done locally which means it can run offline.

One Inference Server can handle multiple clients and streams.

## Inference as a Microservice

The most common way to use Inference is as a small part of a larger system. It's
producing a `response` that is consumed by downstream code. This response sometimes
represents the prediction from a model (for example, a set of `Detections` containing
objects' categorization, location, and size in an image) but it could also represent
the result of post-processing logic (like the pass/fail state of an inspection), an
aggregation (like the count of unique objects seen over the past hour), or a visualization.

For image workloads, the input is passed in as a parameter and the response is
returned synchronously.

<div class="imageContainer">
    <div><img src="/images/architecture/microservice.svg" /></div>
    <div class="caption">Inference as a Microservice</div>
</div>

In the case of video streams, a visual agent
(called [an `InferencePipeline`](/workflows/video_processing/overview/))
is started and runs in a loop until terminated. Responses are polled or subscribed
to by the client application for display or processing.

<div class="imageContainer">
    <div><img src="/images/architecture/pipeline.svg" /></div>
    <div class="caption">InferencePipeline</div>
</div>

Example microservice use-cases:

* Tagging of user-uploaded images to a website
* Determining if a machine is setup correctly before allowing it to turn on
* Blurring faces in a video
* Detecting mismatched wiring in a finished circuit board
* Inspecting a manufactured good to ensure it matches the spec
* Validating that an object is defect and blemish free
* Counting the number of pills in an image

## Inference as an Appliance

Inference can also be treated as an autonomous agent that continuously consumes and
processes a video stream and performs downstream actions (like updating a database,
sending notifications, firing webhooks, or signaling hardware). In this paradigm,
the full logic of the system is defined in a Workflow and the output is pushed to
external systems.

<div class="imageContainer">
    <div><img src="/images/architecture/appliance.svg" /></div>
    <div class="caption">Inference as an Appliance</div>
</div>

Example appliance use-cases:

* Stopping a conveyor belt if a jam has occurred
* Collecting highway traffic analytics
* Flagging suspicious activity in a security camera feed
* Updating an inventory system as vehicles enter or leave a yard
* Sounding an alarm when a scrap heap overflows
* Cataloguing retail customers' wait time over the course of a day

## Why Docker

We _highly_ recommend using our Docker container to interface with Inference. Machine
learning dependencies are sensitive to minor changes to their environment; if they
are not isolated into a deterministic environment, your system is likely to break
when you update your operating system or drivers, update the dependencies of your
application code, apply security patches, or setup a new machine. We ensure that
library versions are compatible with each other, packages are compiled to take 
advantage of the GPU, and security patches are applied.

Additionally, using Docker gives you portability. You can decide later to serve
multiple clients from a single large server or to upgrade from a CPU to a GPU
without having to refactor your application code.
