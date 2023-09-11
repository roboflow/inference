# Python Inference

You can run inference on models in the Roboflow Inference Server directly using the `inference` Python package. This allows you to interface with models without running the Inference HTTP server.

In this guide, we will show how to run inference on object detection, classification, and segmentation models using the `inference` package.

Let's begin!

## Step 1: Install Roboflow Inference

First, we need to install Roboflow Inference. The command to install Roboflow Inference depends on the device on whihc you are running inference. Here are the available packages:

- `inference`: x86 CPU
- `inference[gpu]`: NVIDIA GPU devices
- `inference[arm]`: ARM CPU
- `inference[jetson]`: NVIDIA Jetson
- `inference[trt]`: TensorRT devices

Run the relevant command on your device. Once you have installed the Python package, you can start running inference.

## Step 2: Choose a Model

At the moment, you can only run inference on models trained on Roboflow. Support for bringing your own models is being actively worked on. This guide will be updated when such support is available.

