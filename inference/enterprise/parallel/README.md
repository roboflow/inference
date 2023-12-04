# Parallel Robofolow Inference server

Introducing the highly concurrent implementation of Roboflow Inference Server! This version of the server accepts and processes requests asynchronously, running the web server, preprocessing, auto batching, inference, and post processing all in separate threads to increase server fps throughput. Separate requests to the same model will be batched on the fly as allowed by `$MAX_BATCH_SIZE`, and then response handling will occurr independently. Images are passed via python's SharedMemory module to maximize throughput.

> ⚠️ Currently, only Object Detection, Instance Segmentation, and Classification models are supported by this module. Core models are not enabled

## How Do I use it?
You can build the server using `./inference/enterprise/parallel/build.sh` and run it using `./inference/enterprise/parallel/run.sh`

We provide a container at dockerhub that you can pull using `docker pull roboflow/roboflow-inference-server-gpu-parallel:latest`. If you are pulling a pinned tag, be sure to change the `$TAG` variable in `run.sh`.

This is a drop in replacement for the old server, so you can send requests using the [same API calls](https://inference.roboflow.com/quickstart/http_inference/#step-2-run-inference) you were using previously

> ⚠️ We require a Roboflow Enterprise License to use this in production. See LICENSE.txt for details

## Performance
We measure and report performance across a variety of different task types, by selecting random models found on Roboflow Universe.

### Methodology

The following metrics are taken on a machine with 8 cores and 1 T4 gpu. The fps metrics reflect best out of 3 trials.

### Results
| Workspace | Model | Model Type | split | 0.9.5.rc fps| 0.9.5.parallel fps |
| ----------|------ | ----------- |------|-------------| -------------------|
| senior-design-project-j9gpp | nbafootage/3| object-detection | train | 30.2 fps | 44.03 fps |
| niklas-bommersbach-jyjff   | dart-scorer/8| object-detection | train | 26.6 fps | 47.0 fps |
| geonu  | water-08xpr/1 | instance-segmentation | valid | 2.3 fps | 2.5 fps |
| university-of-bradford | detecting-drusen_1/2 | instance-segmentation | train | 2.3 fps | 2.4 fps |
| fy-project-y9ecd | cataract-detection-viwsu/2 | classification | train | 48.5 fps | 65.4 fps |
| hesunyu | project-bltpu/1 | classification | train | 44.6 fps | 57.7 fps |