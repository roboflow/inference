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

## Using Your New Server

Once you hav a server running, you can access it via [its API](/api.md) or using
[the Python SDK](/inference_helpers/inference_sdk.md). You can also use it to build Workflows
using [the Roboflow Platform UI](https://docs.roboflow.com/workflows/create-a-workflow).

=== "Python SDK"
    ### Install the SDK

    ```bash
    pip install inference-sdk
    ```

    ### Run a workflow

    This code runs [an example model comparison Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSHhIODdZR0FGUWhaVmtOVWNEeVUiLCJ3b3Jrc3BhY2VJZCI6IlhySm9BRVFCQkFPc2ozMmpYZ0lPIiwidXNlcklkIjoiNXcyMFZ6UU9iVFhqSmhUanE2a2FkOXVicm0zMyIsImlhdCI6MTczNTIzNDA4Mn0.AA78pZnlivFs5pBPVX9cMigFAOIIMZk0dA4gxEF5tj4)
    on an Inference Server running on your local machine:

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

=== "Node.js"
    From a JavaScript app, hit your new server with an HTTP request.

    ```js
    const response = await fetch('http://localhost:9001/infer/workflows/roboflow-docs/model-comparison', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            // api_key: "<YOUR API KEY>" // optional to access your private data and models
            inputs: {
                "image": {
                    "type": "url",
                    "value": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
                },
                "model1": "yolov8n-640",
                "model2": "yolov11n-640"
            }
        })
    });

    const result = await response.json();
    console.log(result);
    ```

    !!! Warning
        Be careful not to expose your API Key to external users
        (in other words: don't use this snippet in a public-facing front-end app).

=== "HTTP / cURL"
    Using the server's API you can access it from any other client application.
    From the command line using cURL:

    ```bash
    curl -X POST "http://localhost:9001/infer/workflows/roboflow-docs/model-comparison" \
    -H "Content-Type: application/json" \
    -d '{
        "api_key": "<YOUR API KEY -- REMOVE THIS LINE IF NOT FILLING>",
        "inputs": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
            },
            "model1": "yolov8n-640",
            "model2": "yolov11n-640"
        }
    }'
    ```

    !!! Tip
        ChatGPT is really good at converting snippets like this into other languages.
        If you need help, try pasting it in and asking it to translate it to your
        language of choice.

See [more example Workflows](/workflows/gallery/index.md)
or [start building](/workflows/create_and_run.md).