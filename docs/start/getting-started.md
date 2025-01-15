# Choosing a Deployment Method

There are three primary ways to deploy Inference:

* Cloud Hosting - using Roboflow's managed compute.
* Self Hosting - on your own hardware or edge devices.
* Bring Your Own Cloud - with a VM or cluster.

Each has pros and cons and which one you should choose depends on your particular
use-case and organizational constraints.

## Cloud Hosting

By far the easiest way to get started is with Roboflow's managed services. You can
jump straight to building without having to setup any infrastructure. It's often
the front-door to using Inference even for those who know they will eventually want
to self host.

There are two cloud hosted offerings with different targeted use-cases, capabilities,
and pricing models.

### Serverless Hosted API

The [Serverless Hosted API](/managed/serverless.md) supports running Workflows on
pre-trained & fine-tuned models, chaining models, basic logic, visualizations, and
external integrations.

It supports cloud-hosted VLMs like ChatGPT and Anthropic Claude, but does not support
running heavy models like Florence-2 or SAM 2. It also does not support streaming
video.

The Serverless API scales down to zero when you're not using it (and up to infinity
under load) with quick (a couple of seconds) cold-start time. You pay per model
inference with no minimums. Roboflow's free tier credits may be used.

### Dedicated Deployments

[Dedicated Deployments](/managed/dedicated.md) are single-tenant virtual machines that
are allocated for your exclusive use. They can optionally be configured with a GPU
and used in development mode (where you may be evicted if capacity is needed for a
higher priority task & are limited to 3-hour sessions) or production mode (guaranteed
capacity and no session time limit).

On a Dedicated Deployment, you can stream video, run custom Python code, access
heavy foundation models like SAM 2, Florence-2, and Paligemma (including your fine-tunes
of those models), and install additional dependencies. They are much higher performance
machines than the instances backing the Serverless Hosted API.

Dedicated Deployments are only available on accounts with an active subscription (and
are not available on the free trial). Scale-up time is on the order of a minute or
two. They are billed hourly.

## Self Hosting

Running at the edge is a core priority and focus area of Inference. For many use-cases
latency matters, bandwidth is limited, interfacing with local devices is key, and
resiliency to Internet outages is mandatory.

Running locally on a development machine, an AI computer, or an edge device is as simple
as starting a Docker container.

Detailed [installation instructions and device-specific performance tips are here](/install/index.md).

## Bring Your Own Cloud

Sometimes enterprise compliance policies regarding sensitive data requires running
workloads on-premises. This is supported via
[self-hosting on your own cloud](/install/cloud/index.md).

--8<-- "docs/install/using-your-new-server.md"

See [more example Workflows](/workflows/gallery/index.md)
or [start building](/workflows/create_and_run.md).