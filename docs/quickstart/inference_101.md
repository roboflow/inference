Inference is designed to fit every computer vision use case. Within Inference, there are several different interfaces and your use case will likely dictate which interface you should choose. The interfaces included in Inference are:

- [Native Python API](#native-python-api)
- [HTTP API](#http-api)
- [Inference Pipeline](#inference-pipeline)
- [Advanced Interfaces (Enterprise)](#advanced-usage-interfaces)

## Native Python API

The native python API is the most simple and involves accessing the base package APIs directly. Going this route, you will import Inference modules directly into your python code. You will load models, run inference, and handle the results all within your own logic. You will also need to manage the dependencies within your python environment. If you are creating a simple app or just testing, the native Python API is a great place to start.

[See Docs and Examples](/using_inference/native_python_api/)

## HTTP API

The HTTP API is a helpful way to treat your machine learning models as their own microservice. With this interface, you will run a docker container and make requests over HTTP. The requests contain all of the information Inference needs to run a computer vision model including the model ID, the input image data, and any configurable parameters used during processing (e.g. confidence threshold).

Running Inference this way can have several advantages:

- No Dependency Management: When running Inference within one of Roboflow's published Inference Server containers, all the dependencies are built and isolated so they wont interfere with other dependencies in your code.
- Microservice Architecture: Running Inference as a separate service allows you to operate and maintain your computer vision models separate from other logic within your software, including scaling up and down to meet dynamic loads.
- Roboflow Hosted API: Roboflow hosts a powerful and infinitely scalable version of the Roboflow Inference Server. This makes it even easier to integrate computer vision models into your software without adding any maintenance burden. And, since the Roboflow hosted APIs are running using the Inference package, it's easy to switch between using a hosted server and an on prem server without having to reinvent your client code.
- Non-Python Clients: Running Inference within and HTTP server allows you to interact with it from any language you prefer.

!!! Hint

    Checkout the [Inference SDK](https://inference.roboflow.com/inference_sdk) for a ready made client that makes using the HTTP interface even easier.

[See Docs and Examples](/using_inference/http_api/)

## Inference Pipeline

The Inference Pipeline interface is made for streaming and is likely the best route to go for real time use cases. It is an asynchronous interface that can consume many different video sources including local devices (like webcams), RTSP video streams, video files, etc. With this interface, you define the source of a video stream and sinks. Sinks are methods that operate on the results of inferring on a single frame. The Inference package has several built in sinks but also gives you the ability to write custom sinks for integrating with the rest of your software.

[See Docs and Examples](/using_inference/inference_pipeline/)

## Advanced Usage & Interfaces

There are several advanced interfaces that enhance the capabilities of the base Inference package.

- Active Learning: Active learning helps improve your model over time by contributing real world data back to your Roboflow dataset in real time. [Docs and Examples](/advanced/active_learning.md)
- Parallel HTTP API\*: A highly parallel server capable of accepting requests from many different clients and batching them dynamically in real time to make the most efficient use of the host hardware. [Docs and Examples](/advanced/parallel_http_api.md)
- Stream Manager API\*: An API for starting, stopping, and managing Inference Pipeline instances. This interfaces combines the advantages of running Inference realtime on a stream while also fitting nicely into a microservice architecture. [Docs and Examples](/advanced/stream_management_api.md)

\*For Roboflow enterprise users only. Contact sales@roboflow.com to learn more.
