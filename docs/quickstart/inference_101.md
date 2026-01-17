There are three ways to run Inference:

- [Using the Python SDK (Images and Videos)](#using-the-python-sdk-images-and-videos)
- [Using the Python HTTP SDK (Images)](#python-http-sdk)
- [Using the HTTP SDK (Images for all languages)](#http-sdk)

We document every method in the "Inputs" section of the Inference documentation.

Below, we talk about when you would want to use each method.

## Using the Python SDK (Images and Videos)

![](https://media.roboflow.com/inference/python-integration.png)

You can use the Python SDK to run models on images and videos directly using the Inference code, without using Docker.

Any code example that imports from `inference.models` uses the model directly.

To use the Python SDK, you need to install:

```bash
pip install inference
```

## Python HTTP SDK

You can use the Python HTTP SDK to run models using Inference with Docker.

Any code example that imports from `inference_sdk` uses the HTTP API.

To use this method, you will need an Inference server running, or you can use the Roboflow endpoint for your model.

### Self-Hosted Inference Server

You can set up and install and Inference server using:

```bash
pip install inference
inference server start
```

### Roboflow Hosted API

First, run:

```bash
pip install inference inference-sdk
```

Then, use your Roboflow hosted API endpoint to access your model. You can find this in the Deploy tab of your Roboflow model.

## HTTP SDK

You can deploy your model with Inference and Docker and use the API in any programming language (i.e. Swift, Node.js, and more).

To use this method, you will need an Inference server running. You can set up and install and Inference server using:


### Self-Hosted Inference Server

You can set up and install and Inference server using:

```bash
pip install inference
inference server start
```

### Roboflow Hosted API

Use your Roboflow hosted API endpoint to access your model. You can find this in the Deploy tab of your Roboflow model.

## Benefits of Using Inference Over HTTP

![](https://media.roboflow.com/inference/http-api.png)
![](https://media.roboflow.com/inference/http-api-roboflow.png)

You can run Inference directly from your codebase or using a HTTP microservice deployed with Docker.

Running Inference this way can have several advantages:

- **No Dependency Management:** When running Inference within one of Roboflow's published Inference Server containers, all the dependencies are built and isolated so they wont interfere with other dependencies in your code.
- **Microservice Architecture:** Running Inference as a separate service allows you to operate and maintain your computer vision models separate from other logic within your software, including scaling up and down to meet dynamic loads.
- **Roboflow Hosted API:** Roboflow hosts a powerful and infinitely scalable version of the Roboflow Inference Server. This makes it even easier to integrate computer vision models into your software without adding any maintenance burden. And, since the Roboflow hosted APIs are running using the Inference package, it's easy to switch between using a hosted server and an on prem server without having to reinvent your client code.
- **Non-Python Clients:** Running Inference within an HTTP server allows you to interact with it from any language you prefer.

## Advanced Usage & Interfaces

There are several advanced interfaces that enhance the capabilities of the base Inference package.

- **Active Learning**: Active learning helps improve your model over time by contributing real world data back to your Roboflow dataset in real time.
- **Parallel HTTP API**: A highly parallel server capable of accepting requests from many different clients and batching them dynamically in real time to make the most efficient use of the host hardware. [Docs and Examples](/enterprise/parallel_processing.md)
- **Stream Manager API**: An API for starting, stopping, and managing Inference Pipeline instances. This interfaces combines the advantages of running Inference realtime on a stream while also fitting nicely into a microservice architecture. [Docs and Examples](../enterprise/stream_management_api.md)

To learn more, [contact the Roboflow sales team](https://roboflow.com/sales).
